"""LangGraph supervisor for the complete tabular interpretability pipeline."""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
from anthropic import Anthropic
from langgraph.graph import END, START, StateGraph

from eda_agent import Finding, run_eda_agent, severity_violation_rate
from evidence_integration import attach_evidence_verdicts_to_claims
from evidence_validator import resolve_evidence_value
from evaluation_agent import (
    MetricChoice,
    deterministic_metric_choice,
    evaluate_selected_models,
    run_metric_choice_agent,
)
from layer1_persistence import compute_layer1_metrics, rollup_claim_verdicts
from layer2_critic import compute_layer2_metrics, run_layer2_critic
from model_selection_agent import ModelChoice, run_model_selection_agent
from preprocessing_agent import (
    ColumnPlan,
    _make_plan_from_default,
    derive_column_plan_defaults,
    override_rate,
    run_preprocessing_agent,
)
from preprocessing_pipeline import build_preprocessing_pipeline
from profile_dataset import profile_dataset


@dataclass
class StageError:
    stage: str
    code: str
    message: str
    attempt: int | None = None


@dataclass
class PipelineState:
    run_id: str
    dataset_path: str
    target_column: str
    task_type: str
    raw_profile: dict | None = None
    eda_findings: list = field(default_factory=list)
    column_plans: list = field(default_factory=list)
    preprocessing_pipeline: object | None = None
    model_choices: list = field(default_factory=list)
    evaluation_results: list = field(default_factory=list)
    all_claims: list = field(default_factory=list)
    all_evidence_verdicts: list = field(default_factory=list)
    all_layer2_verdicts: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    llm_call_count: int = 0
    call_cap_used: int | None = None
    metric_choice: object | None = None
    stratification_degraded: bool = False
    stage_status: dict[str, str] = field(default_factory=dict)
    completed: bool = False
    aborted: bool = False


class RunAbortedCallCap(RuntimeError):
    pass


class LLMAPIExhausted(RuntimeError):
    pass


def compute_call_cap(n_columns: int) -> int:
    """Return a dataset-width-aware LLM call allowance for one pipeline run."""
    if n_columns < 1:
        raise ValueError("n_columns must be positive")
    # This lands 15%-28% above Milestone 9's clean-run estimates while
    # retaining isolated Layer-2 coverage for every claim.
    base = 28
    per_column = 1.12
    return int(base + per_column * n_columns)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if value.__class__.__module__.startswith("sklearn"):
        transformer = getattr(value, "named_steps", {}).get("preprocess")
        return {
            "repr": repr(value),
            "is_fitted": bool(transformer is not None and hasattr(transformer, "transformers_")),
        }
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)


class _InstrumentedMessages:
    def __init__(self, supervisor: "PipelineSupervisor", stage: str, retry_context: str):
        self.supervisor = supervisor
        self.stage = stage
        self.retry_context = retry_context

    def create(self, **kwargs):
        request = deepcopy(kwargs)
        if self.retry_context:
            request["messages"] = deepcopy(request.get("messages", []))
            if request["messages"]:
                content = request["messages"][-1].get("content", "")
                request["messages"][-1]["content"] = f"{content}\n\nRETRY CORRECTION:\n{self.retry_context}"

        delays = (2, 4, 8)
        for api_attempt in range(4):
            self.supervisor._enforce_call_cap(self.stage)
            self.supervisor.state.llm_call_count += 1
            started = time.perf_counter()
            try:
                response = self.supervisor.base_client.messages.create(**request)
            except Exception as error:
                latency = time.perf_counter() - started
                self.supervisor._record_prompt(self.stage, request, None, latency, error)
                if api_attempt == 3:
                    raise LLMAPIExhausted(str(error)) from error
                self.supervisor._error(self.stage, "LLM_API_RETRY", str(error), api_attempt + 1)
                self.supervisor.sleep_fn(delays[api_attempt])
                continue
            latency = time.perf_counter() - started
            self.supervisor._record_prompt(self.stage, request, response, latency, None)
            return response
        raise AssertionError("unreachable")


class _InstrumentedClient:
    def __init__(self, supervisor: "PipelineSupervisor", stage: str, retry_context: str = ""):
        self.messages = _InstrumentedMessages(supervisor, stage, retry_context)


class PipelineSupervisor:
    """Execute existing stage agents and critics through a compiled LangGraph."""

    def __init__(
        self,
        state: PipelineState,
        *,
        runs_dir: str | Path = "runs",
        client: Any | None = None,
        model: str | None = None,
        critic_model: str | None = None,
        agent_temperature: float = 0.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.state = state
        self.run_dir = Path(runs_dir) / state.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.attempt_started_at = datetime.now(timezone.utc)
        self.attempt_id = (
            f"{self.attempt_started_at.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
        )
        self.attempt_dir = self.run_dir / "attempts" / self.attempt_id
        self.active_stage: str | None = None
        self.base_client = client or Anthropic()
        self.model = model
        self.critic_model = critic_model
        self.agent_temperature = agent_temperature
        self.sleep_fn = sleep_fn
        self.prompt_records: list[dict] = []
        self.df: pd.DataFrame | None = None
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        nodes = [
            ("profile_dataset", self._stage_node("profile_dataset", self._profile_node)),
            ("eda_agent", self._stage_node("eda_agent", self._eda_node)),
            ("critic_eda", self._stage_node("critic_eda", lambda state: self._critic_node(state, "eda", state.eda_findings))),
            ("preprocess_agent", self._stage_node("preprocess_agent", self._preprocess_node)),
            ("critic_preprocess", self._stage_node("critic_preprocess", lambda state: self._critic_node(state, "preprocess", state.column_plans))),
            ("model_select_agent", self._stage_node("model_select_agent", self._model_select_node)),
            ("critic_model_selection", self._stage_node("critic_model_selection", lambda state: self._critic_node(state, "model_selection", state.model_choices))),
            ("evaluate_agent", self._stage_node("evaluate_agent", self._evaluation_node)),
            ("critic_evaluation", self._stage_node("critic_evaluation", lambda state: self._critic_node(state, "evaluation", [state.metric_choice] if state.metric_choice else []))),
            ("synthesis", self._stage_node("synthesis", self._synthesis_node)),
        ]
        for name, node in nodes:
            graph.add_node(name, node)
        graph.add_edge(START, nodes[0][0])
        for (left, _), (right, _) in zip(nodes, nodes[1:]):
            graph.add_edge(left, right)
        graph.add_edge(nodes[-1][0], END)
        return graph.compile()

    def _stage_node(self, stage: str, operation: Callable[[PipelineState], PipelineState]):
        def wrapped(state: PipelineState) -> PipelineState:
            self.active_stage = stage
            return operation(state)

        return wrapped

    def run(self) -> PipelineState:
        self._write_attempt_manifest("running")
        try:
            result = self.graph.invoke(self.state)
            self.state = result if isinstance(result, PipelineState) else PipelineState(**result)
        except BaseException as error:
            crash_dump = self._write_crash_dump(error)
            self._write_attempt_manifest("crashed", crash_dump=crash_dump)
            raise
        self._write_attempt_manifest("completed" if self.state.completed else "aborted")
        return self.state

    def _attempt_payload(self, status: str, crash_dump: Path | None = None) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "run_id": self.state.run_id,
            "status": status,
            "started_at": self.attempt_started_at.isoformat(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "active_stage": self.active_stage,
            "partial_counts": {
                "prompts": len(self.prompt_records),
                "claims": len(self.state.all_claims),
                "evidence_verdicts": len(self.state.all_evidence_verdicts),
                "layer2_verdicts": len(self.state.all_layer2_verdicts),
            },
            "crash_dump": str(crash_dump.relative_to(self.run_dir)) if crash_dump else None,
        }

    def _write_attempt_manifest(self, status: str, crash_dump: Path | None = None) -> None:
        self.attempt_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.attempt_dir / "attempt.json", self._attempt_payload(status, crash_dump))

    def _write_crash_dump(self, error: BaseException) -> Path:
        crash_dump = self.run_dir / f"crash_dump_{self.attempt_id}.json"
        original_traceback = traceback.format_exc()
        payload = {
            "attempt": self._attempt_payload("crashed", crash_dump=crash_dump),
            "exception": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": original_traceback,
            },
            "partial_state": _jsonable(self.state),
            "prompt_records": _jsonable(self.prompt_records),
        }
        try:
            self._write_json(crash_dump, payload)
        except Exception as dump_error:
            fallback_dump = self.run_dir / f"crash_dump_{self.attempt_id}_minimal.json"
            self._write_json(
                fallback_dump,
                {
                    "attempt": self._attempt_payload("crashed"),
                    "exception": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": original_traceback,
                    },
                    "crash_dump_error": f"{type(dump_error).__name__}: {dump_error}",
                },
            )
            return fallback_dump
        return crash_dump

    def _enforce_call_cap(self, stage: str) -> None:
        if self.state.call_cap_used is None:
            raise RuntimeError("LLM call cap was not initialized during profiling")
        if self.state.llm_call_count >= self.state.call_cap_used:
            self.state.aborted = True
            self.state.stage_status[stage] = "aborted_call_cap"
            self._error(stage, "RUN_ABORTED_CALL_CAP", "LLM call cap reached before API invocation")
            raise RunAbortedCallCap("RUN_ABORTED_CALL_CAP")

    def _error(self, stage: str, code: str, message: str, attempt: int | None = None) -> None:
        self.state.errors.append(StageError(stage, code, message, attempt))

    def _record_prompt(self, stage: str, prompt: dict, response: Any, latency: float, error: Exception | None) -> None:
        usage = getattr(response, "usage", None)
        self.prompt_records.append(
            {
                "stage": stage,
                "prompt": _jsonable(prompt),
                "raw_response": _jsonable(response) if response is not None else None,
                "tokens": {
                    "input": getattr(usage, "input_tokens", None),
                    "output": getattr(usage, "output_tokens", None),
                },
                "latency_seconds": latency,
                "error": str(error) if error else None,
            }
        )

    def _client(self, stage: str, retry_context: str = "") -> _InstrumentedClient:
        return _InstrumentedClient(self, stage, retry_context)

    def _invalid_evidence_keys(self, claims: list) -> list[str]:
        invalid: set[str] = set()
        for claim in claims:
            for reference in claim.evidence:
                try:
                    resolve_evidence_value(reference.metric_key, self.state.raw_profile or {})
                except KeyError:
                    invalid.add(reference.metric_key)
        return sorted(invalid)

    def _run_stage(self, stage: str, operation: Callable[[Any], Any], claims_from_output: Callable[[Any], list], fallback: Callable[[], Any]):
        retry_context = ""
        for attempt in (1, 2):
            if self.state.aborted:
                return None
            try:
                output = operation(self._client(stage, retry_context))
                claims = claims_from_output(output)
                invalid_keys = self._invalid_evidence_keys(claims)
                if invalid_keys:
                    raise ValueError(f"EvidenceRef metric_key(s) not in raw_profile: {invalid_keys}")
                return output
            except RunAbortedCallCap:
                return None
            except LLMAPIExhausted as error:
                self._error(stage, "LLM_API_RETRIES_EXHAUSTED", str(error), 4)
                return fallback()
            except Exception as error:
                code = "STAGE_SCHEMA_OR_EVIDENCE_RETRY" if attempt == 1 else "STAGE_RETRY_EXHAUSTED"
                self._error(stage, code, str(error), attempt)
                if attempt == 1:
                    retry_context = f"Previous output failed validation: {error}. Correct this exact error."
        return fallback()

    def _accumulate(self, claims: list) -> None:
        self.state.all_claims.extend(claims)
        for claim in claims:
            self.state.all_evidence_verdicts.extend(getattr(claim, "evidence_verdicts", None) or [])

    def _profile_node(self, state: PipelineState) -> PipelineState:
        self.state = state
        if state.aborted:
            return state
        self.df = pd.read_csv(state.dataset_path)
        state.call_cap_used = compute_call_cap(self.df.shape[1])
        state.raw_profile = profile_dataset(self.df, state.target_column, state.task_type)
        state.stage_status["profile"] = "completed"
        return state

    def _eda_node(self, state: PipelineState) -> PipelineState:
        self.state = state

        def fallback():
            self._error("eda", "FALLBACK_EDA_EMPTY", "Proceeding with zero EDA findings")
            state.stage_status["eda"] = "fallback_empty"
            return []

        output = self._run_stage(
            "eda",
            lambda client: run_eda_agent(
                state.raw_profile or {},
                model=self.model,
                temperature=self.agent_temperature,
                client=client,
            ),
            lambda value: value if isinstance(value, list) and all(isinstance(item, Finding) for item in value) else (_ for _ in ()).throw(ValueError("EDA output schema invalid")),
            fallback,
        )
        state.eda_findings = output or []
        if output is not None and state.stage_status.get("eda") != "fallback_empty":
            state.stage_status["eda"] = "completed"
        self._accumulate(state.eda_findings)
        return state

    def _preprocess_node(self, state: PipelineState) -> PipelineState:
        self.state = state
        if state.aborted:
            return state

        def fallback():
            defaults = derive_column_plan_defaults(state.raw_profile or {}, state.target_column, list(self.df.columns))
            plans = [_make_plan_from_default(default, state.raw_profile or {}, index) for index, default in enumerate(defaults, 1)]
            for plan in plans:
                plan.justification = ""
                plan.confidence = 0.0
                plan.override_reason = ""
                plan.is_override = False
            attach_evidence_verdicts_to_claims(plans, state.raw_profile or {})
            pipeline = build_preprocessing_pipeline(plans, state.raw_profile or {})
            self._error("preprocess", "FALLBACK_PREPROCESS_RULE_DERIVED", "Used deterministic preprocessing plans")
            state.stage_status["preprocess"] = "fallback_rule_derived"
            return plans, pipeline

        output = self._run_stage(
            "preprocess",
            lambda client: run_preprocessing_agent(
                state.raw_profile or {},
                self.df,
                state.target_column,
                model=self.model,
                temperature=self.agent_temperature,
                client=client,
            ),
            lambda value: value[0] if isinstance(value, tuple) and len(value) == 2 and all(isinstance(item, ColumnPlan) for item in value[0]) else (_ for _ in ()).throw(ValueError("Preprocessing output schema invalid")),
            fallback,
        )
        if output is not None:
            state.column_plans, state.preprocessing_pipeline = output
            if hasattr(state.preprocessing_pipeline.named_steps["preprocess"], "transformers_"):
                raise RuntimeError("Supervisor received a fitted preprocessing pipeline")
            if state.stage_status.get("preprocess") != "fallback_rule_derived":
                state.stage_status["preprocess"] = "completed"
            self._accumulate(state.column_plans)
        return state

    def _model_select_node(self, state: PipelineState) -> PipelineState:
        self.state = state
        if state.aborted:
            return state

        def fallback():
            self._error("model_selection", "FALLBACK_MODEL_SELECT_UNHANDLED", "No model-selection fallback is defined")
            state.stage_status["model_selection"] = "fallback_unhandled"
            return []

        dropped_columns = [plan.column for plan in state.column_plans if plan.action == "drop"]
        retained_columns = [plan.column for plan in state.column_plans if plan.action != "drop"]
        output = self._run_stage(
            "model_selection",
            lambda client: run_model_selection_agent(
                state.raw_profile or {},
                state.task_type,
                retained_columns=retained_columns,
                dropped_columns=dropped_columns,
                model=self.model,
                temperature=self.agent_temperature,
                client=client,
            ),
            lambda value: value if isinstance(value, list) and len(value) == 4 and all(isinstance(item, ModelChoice) for item in value) else (_ for _ in ()).throw(ValueError("Model selection output schema invalid")),
            fallback,
        )
        state.model_choices = output or []
        if state.model_choices:
            state.stage_status["model_selection"] = "completed"
            self._accumulate(state.model_choices)
        return state

    def _evaluation_node(self, state: PipelineState) -> PipelineState:
        self.state = state
        if state.aborted:
            return state
        if not state.model_choices:
            state.stage_status["evaluation"] = "skipped_no_models"
            return state

        def fallback():
            self._error("evaluation", "FALLBACK_EVALUATION_UNHANDLED", "Metric justification unavailable; deterministic CV continues")
            state.stage_status["evaluation"] = "fallback_metric_only"
            return None

        dropped_columns = [plan.column for plan in state.column_plans if plan.action == "drop"]
        retained_columns = [plan.column for plan in state.column_plans if plan.action != "drop"]
        choice = self._run_stage(
            "evaluation",
            lambda client: run_metric_choice_agent(
                state.raw_profile or {},
                state.task_type,
                retained_columns=retained_columns,
                dropped_columns=dropped_columns,
                model=self.model,
                temperature=self.agent_temperature,
                client=client,
            ),
            lambda value: [value] if isinstance(value, MetricChoice) else (_ for _ in ()).throw(ValueError("Evaluation claim schema invalid")),
            fallback,
        )
        state.metric_choice = choice
        if choice is not None:
            self._accumulate([choice])
        primary, secondary = (
            (choice.primary_metric, choice.secondary_metrics)
            if choice is not None
            else deterministic_metric_choice(state.raw_profile or {}, state.task_type)
        )
        state.evaluation_results, state.stratification_degraded = evaluate_selected_models(
            state.model_choices,
            state.preprocessing_pipeline,
            self.df,
            state.target_column,
            state.task_type,
            primary,
            secondary,
        )
        if state.stage_status.get("evaluation") != "fallback_metric_only":
            state.stage_status["evaluation"] = "completed"
        return state

    def _critic_node(self, state: PipelineState, stage: str, claims: list) -> PipelineState:
        self.state = state
        if state.aborted or not claims:
            state.stage_status[f"critic_{stage}"] = "skipped" if not state.aborted else "aborted"
            return state
        try:
            verdicts = run_layer2_critic(
                rollup_claim_verdicts(claims),
                claims,
                state.raw_profile or {},
                model=self.critic_model,
                client=self._client(f"critic_{stage}"),
            )
            state.all_layer2_verdicts.extend(verdicts)
            state.stage_status[f"critic_{stage}"] = "completed"
        except RunAbortedCallCap:
            state.stage_status[f"critic_{stage}"] = "aborted_call_cap"
        except Exception as error:
            self._error(f"critic_{stage}", "CRITIC_ERROR_OBSERVED", str(error))
            state.stage_status[f"critic_{stage}"] = "error_nonblocking"
        return state

    def _synthesis_node(self, state: PipelineState) -> PipelineState:
        self.state = state
        state.completed = not state.aborted
        state.stage_status["synthesis"] = "completed" if not state.aborted else "aborted_run_persisted"
        self._persist(state)
        return state

    def _persist(self, state: PipelineState) -> None:
        claim_verdicts = rollup_claim_verdicts(state.all_claims)
        metrics = compute_layer1_metrics(
            state.all_evidence_verdicts,
            claim_verdicts,
            severity_violation_rate(state.eda_findings),
        )
        stage_by_claim = {claim.claim_id: claim.stage for claim in state.all_claims}
        metrics.update(compute_layer2_metrics(state.all_layer2_verdicts, stage_by_claim))
        metrics.update(
            {
                "override_rate": override_rate(state.column_plans),
                "columns_dropped": sum(plan.action == "drop" for plan in state.column_plans),
                "columns_kept": sum(plan.action != "drop" for plan in state.column_plans),
                "preprocess_claims": len(state.column_plans),
                "models_evaluated": len(state.evaluation_results),
                "stratification_degraded": state.stratification_degraded,
                "llm_call_count": state.llm_call_count,
                "call_cap_used": state.call_cap_used,
                "run_completed": state.completed,
                "run_aborted": state.aborted,
                "stage_fallbacks": {
                    stage: status
                    for stage, status in sorted(state.stage_status.items())
                    if status.startswith("fallback_")
                },
            }
        )
        if state.evaluation_results:
            minimize = state.evaluation_results[0].primary_metric in {"mae", "rmse"}
            best = (min if minimize else max)(state.evaluation_results, key=lambda result: result.primary_metric_mean)
            metrics["best_model_id"] = best.model_id
            metrics["best_primary_metric_value"] = best.primary_metric_mean
        else:
            metrics["best_model_id"] = None
            metrics["best_primary_metric_value"] = None

        self._write_json(self.run_dir / "state.json", state)
        self._write_jsonl(self.run_dir / "claims.jsonl", state.all_claims)
        self._write_jsonl(self.run_dir / "evidence_verdicts.jsonl", state.all_evidence_verdicts)
        self._write_jsonl(self.run_dir / "claim_verdicts.jsonl", claim_verdicts)
        self._write_jsonl(self.run_dir / "layer2_verdicts.jsonl", state.all_layer2_verdicts)
        self._write_jsonl(self.run_dir / "evaluation_results.jsonl", state.evaluation_results)
        self._write_jsonl(self.run_dir / "column_plans.jsonl", state.column_plans)
        self._write_jsonl(self.run_dir / "prompts.jsonl", self.prompt_records)
        self._write_jsonl(self.run_dir / "errors.jsonl", state.errors)
        self._write_json(self.run_dir / "metrics.json", metrics)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n")


def run_supervised_pipeline(
    dataset_path: str,
    target_column: str,
    task_type: str,
    run_id: str,
    *,
    runs_dir: str | Path = "runs",
    client: Any | None = None,
    model: str | None = None,
    critic_model: str | None = None,
    agent_temperature: float = 0.0,
    initial_llm_call_count: int = 0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> PipelineState:
    state = PipelineState(run_id, dataset_path, target_column, task_type, llm_call_count=initial_llm_call_count)
    return PipelineSupervisor(
        state,
        runs_dir=runs_dir,
        client=client,
        model=model,
        critic_model=critic_model,
        agent_temperature=agent_temperature,
        sleep_fn=sleep_fn,
    ).run()
