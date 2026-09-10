"""Executable normal and failure-path tests for the LangGraph supervisor."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from pipeline_supervisor import compute_call_cap, run_supervised_pipeline
from preprocessing_agent import derive_column_plan_defaults
from profile_dataset import profile_dataset


DATASET = "/Users/akhilgorla/Downloads/mini_dataset.csv"


class FakeAnthropic:
    def __init__(self, profile, *, bad_eda_once=False, bad_eda_twice=False, critic_score=3):
        self.profile = profile
        self.bad_eda_once = bad_eda_once
        self.bad_eda_twice = bad_eda_twice
        self.critic_score = critic_score
        self.eda_calls = 0
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        tool_name = kwargs["tool_choice"]["name"]
        payload = getattr(self, f"_{tool_name}")()
        block = SimpleNamespace(type="tool_use", name=tool_name, input=payload)
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        return SimpleNamespace(content=[block], usage=usage)

    def _record_eda_findings(self):
        self.eda_calls += 1
        if self.bad_eda_twice and self.eda_calls <= 2:
            return {"findings": []}
        invalid = self.bad_eda_once and self.eda_calls == 1
        findings = []
        for index in range(5):
            metric_key = "nonexistent_retry_key" if invalid and index == 0 else "n_rows"
            claimed_value = 999 if invalid and index == 0 else self.profile["n_rows"]
            findings.append({
                "claim_id": f"eda_{index + 1:03d}", "stage": "eda",
                "decision": f"Inspect profile item {index + 1}.",
                "justification": "The cited profile value supports inspection.",
                "evidence": [{"metric_key": metric_key, "claimed_value": claimed_value}],
                "confidence": 0.8, "severity": "info",
            })
        return {"findings": findings}

    def _record_preprocessing_plans(self):
        plans = []
        for default in derive_column_plan_defaults(self.profile, "target"):
            plans.append({
                "column": default.column,
                "justification": "Use the deterministic preprocessing rule.",
                "evidence": [{"metric_key": key, "claimed_value": self.profile[key]} for key in default.evidence_keys],
                "confidence": 1.0, "override_reason": "",
                "imputation": default.imputation, "encoding": default.encoding,
                "scaling": default.scaling,
            })
        return {"plans": plans}

    def _record_model_choices(self):
        rows = [("logreg", 1, False), ("rf", 2, False), ("knn", 3, False), ("gbt", None, True)]
        return {"choices": [{
            "model_id": model_id, "rank": rank, "is_rejection": rejected,
            "decision": ("Reject " if rejected else "Select ") + model_id,
            "justification": "Dataset shape, signal structure, and interpretability tradeoffs support this choice.",
            "evidence": [{"metric_key": "n_rows", "claimed_value": self.profile["n_rows"]}, {"metric_key": "col:age:dtype", "claimed_value": self.profile["col:age:dtype"]}],
            "confidence": 0.8,
        } for model_id, rank, rejected in rows]}

    def _record_metric_justification(self):
        return {
            "decision": "Use ROC-AUC.",
            "justification": "The target balance supports the deterministic metric.",
            "evidence": [{"metric_key": "target_class_balance", "claimed_value": self.profile["target_class_balance"]}],
            "confidence": 1.0,
        }

    def _record_layer2_verdict(self):
        return {
            "claim_id": "ignored", "layer2_score": self.critic_score,
            "discrepancy_note": "forced score zero" if self.critic_score == 0 else "",
            "critic_confidence": 1.0,
        }


class FlakyAnthropic(FakeAnthropic):
    def __init__(self, profile):
        super().__init__(profile)
        self.failures_remaining = 3

    def create(self, **kwargs):
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise TimeoutError("injected timeout")
        return super().create(**kwargs)


class PreprocessFallbackAnthropic(FakeAnthropic):
    def create(self, **kwargs):
        if kwargs["tool_choice"]["name"] == "record_preprocessing_plans":
            self.requests.append(kwargs)
            block = SimpleNamespace(
                type="tool_use",
                name="record_preprocessing_plans",
                input={},
            )
            return SimpleNamespace(
                content=[block],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            )
        return super().create(**kwargs)


def _profile():
    return profile_dataset(pd.read_csv(DATASET), "target", "binary")


def _run(fake, run_id, initial_count=0):
    root = tempfile.mkdtemp(prefix=f"{run_id}_")
    state = run_supervised_pipeline(DATASET, "target", "binary", run_id, runs_dir=root, client=fake, initial_llm_call_count=initial_count, sleep_fn=lambda _: None)
    return state, Path(root) / run_id


def test_1_normal_run():
    state, run_dir = _run(FakeAnthropic(_profile()), "normal")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    prompts = [json.loads(line) for line in (run_dir / "prompts.jsonl").read_text().splitlines()]
    model_prompt = next(row["prompt"] for row in prompts if row["stage"] == "model_selection")
    evaluation_prompt = next(row["prompt"] for row in prompts if row["stage"] == "evaluation")
    dropped = [plan.column for plan in state.column_plans if plan.action == "drop"]
    retained = [plan.column for plan in state.column_plans if plan.action != "drop"]
    model_prompt_text = json.dumps(model_prompt)
    evaluation_prompt_text = json.dumps(evaluation_prompt)
    model_metric_keys = model_prompt["tools"][0]["input_schema"]["properties"]["choices"]["items"]["properties"]["evidence"]["items"]["properties"]["metric_key"]["enum"]
    model_evidence_keys = [reference.metric_key for choice in state.model_choices for reference in choice.evidence]
    assert state.completed and not state.aborted
    assert state.raw_profile["n_cols"] == 11
    assert state.call_cap_used == compute_call_cap(11) == 40
    assert metrics["call_cap_used"] == state.call_cap_used
    assert metrics["stage_fallbacks"] == {}
    assert all(column in model_prompt_text and column in evaluation_prompt_text for column in dropped + retained)
    assert "ORIGINAL dataset" in model_prompt_text and "RETAINED columns only" in model_prompt_text
    assert "ORIGINAL dataset" in evaluation_prompt_text and "RETAINED columns only" in evaluation_prompt_text
    assert "n_cols" not in model_metric_keys and "n_cols" not in model_evidence_keys
    assert not any(key.startswith(f"col:{column}:") for column in dropped for key in model_metric_keys)
    assert not any(key.startswith(f"col:{column}:") for column in dropped for key in model_evidence_keys)
    assert all(state.stage_status.get(stage) == "completed" for stage in ["profile", "eda", "preprocess", "model_selection", "evaluation", "synthesis"])
    assert len((run_dir / "claims.jsonl").read_text().splitlines()) == metrics["total_claims"]
    assert all((run_dir / name).exists() for name in ["state.json", "claims.jsonl", "evidence_verdicts.jsonl", "claim_verdicts.jsonl", "layer2_verdicts.jsonl", "metrics.json", "prompts.jsonl", "errors.jsonl"])
    row_counts = {
        name: len((run_dir / name).read_text().splitlines())
        for name in ["claims.jsonl", "evidence_verdicts.jsonl", "claim_verdicts.jsonl", "layer2_verdicts.jsonl", "prompts.jsonl", "errors.jsonl"]
    }
    print("NORMAL", {"completed": state.completed, "total_claims": metrics["total_claims"], "dropped": dropped, "retained": retained, "model_evidence_keys": model_evidence_keys, "row_counts": row_counts})


def test_2_invalid_evidence_retries_and_succeeds():
    state, run_dir = _run(FakeAnthropic(_profile(), bad_eda_once=True), "retry")
    prompts = [json.loads(line) for line in (run_dir / "prompts.jsonl").read_text().splitlines()]
    errors = [json.loads(line) for line in (run_dir / "errors.jsonl").read_text().splitlines()]
    retry_prompts = [row for row in prompts if row["stage"] == "eda" and "RETRY CORRECTION" in json.dumps(row["prompt"])]
    assert state.completed and state.stage_status["eda"] == "completed"
    assert retry_prompts and "nonexistent_retry_key" in json.dumps(retry_prompts[0]["prompt"])
    assert any(row["code"] == "STAGE_SCHEMA_OR_EVIDENCE_RETRY" for row in errors)
    print("RETRY", {"completed": state.completed, "eda_calls": 2, "retry_contains": "nonexistent_retry_key", "error_codes": [row["code"] for row in errors]})


def test_3_schema_failure_twice_uses_eda_fallback():
    state, run_dir = _run(FakeAnthropic(_profile(), bad_eda_twice=True), "fallback")
    errors = [json.loads(line) for line in (run_dir / "errors.jsonl").read_text().splitlines()]
    codes = [row["code"] for row in errors]
    assert state.completed and state.eda_findings == []
    assert state.stage_status["eda"] == "fallback_empty"
    assert state.stage_status["preprocess"] == "completed" and state.stage_status["evaluation"] == "completed"
    assert "FALLBACK_EDA_EMPTY" in codes
    print("FALLBACK", {"completed": state.completed, "eda_findings": 0, "preprocess": state.stage_status["preprocess"], "evaluation": state.stage_status["evaluation"], "error_codes": codes})


def test_preprocess_fallback_is_visible_in_metrics():
    state, run_dir = _run(PreprocessFallbackAnthropic(_profile()), "preprocess_fallback")
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert state.completed
    assert state.stage_status["preprocess"] == "fallback_rule_derived"
    assert metrics["stage_fallbacks"] == {"preprocess": "fallback_rule_derived"}
    print("STAGE_FALLBACKS", metrics["stage_fallbacks"])


def test_4_critic_score_zero_never_gates():
    state, _ = _run(FakeAnthropic(_profile(), critic_score=0), "critic_zero")
    assert state.completed and state.stage_status["evaluation"] == "completed"
    assert state.all_layer2_verdicts and all(verdict.layer2_score == 0 for verdict in state.all_layer2_verdicts)
    print("CRITIC_ZERO", {"completed": state.completed, "scores": sorted(set(verdict.layer2_score for verdict in state.all_layer2_verdicts)), "evaluation": state.stage_status["evaluation"], "models_evaluated": len(state.evaluation_results)})


def test_5_call_cap_aborts_before_next_call():
    call_cap = compute_call_cap(11)
    state, run_dir = _run(object(), "call_cap", initial_count=call_cap)
    errors = [json.loads(line) for line in (run_dir / "errors.jsonl").read_text().splitlines()]
    assert state.aborted and not state.completed and state.llm_call_count == call_cap
    assert [row["code"] for row in errors] == ["RUN_ABORTED_CALL_CAP"]
    print("CALL_CAP", {"aborted": state.aborted, "completed": state.completed, "llm_call_count": state.llm_call_count, "error_codes": [row["code"] for row in errors]})


def test_api_timeout_retries_use_exact_backoff_then_succeed():
    delays = []
    root = tempfile.mkdtemp(prefix="api_retry_")
    state = run_supervised_pipeline(DATASET, "target", "binary", "api_retry", runs_dir=root, client=FlakyAnthropic(_profile()), sleep_fn=delays.append)
    codes = [error.code for error in state.errors]
    assert state.completed and delays == [2, 4, 8]
    assert codes.count("LLM_API_RETRY") == 3
    print("API_RETRY", {"completed": state.completed, "delays": delays, "retry_events": codes.count("LLM_API_RETRY")})


def test_agent_temperature_does_not_change_critic_temperature():
    fake = FakeAnthropic(_profile())
    root = tempfile.mkdtemp(prefix="temperature_07_")
    state = run_supervised_pipeline(
        DATASET,
        "target",
        "binary",
        "temperature_07",
        runs_dir=root,
        client=fake,
        agent_temperature=0.7,
        sleep_fn=lambda _: None,
    )
    temperatures_by_tool = {
        request["tool_choice"]["name"]: request["temperature"]
        for request in fake.requests
    }
    assert state.completed
    assert temperatures_by_tool["record_eda_findings"] == 0.7
    assert temperatures_by_tool["record_preprocessing_plans"] == 0.7
    assert temperatures_by_tool["record_model_choices"] == 0.7
    assert temperatures_by_tool["record_metric_justification"] == 0.7
    assert temperatures_by_tool["record_layer2_verdict"] == 0
    print("TEMPERATURE", temperatures_by_tool)
