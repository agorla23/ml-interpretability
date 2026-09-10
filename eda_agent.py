"""LLM-backed EDA finding generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from anthropic import Anthropic

from evidence_integration import attach_evidence_verdicts_to_claims
from evidence_validator import Claim, EvidenceRef, EvidenceVerdict
from layer1_persistence import append_layer1_claims, persist_layer1_run, rollup_claim_verdicts
from layer2_critic import persist_layer2_run, run_layer2_critic
from evaluation_agent import evaluate_selected_models, persist_evaluation_results, run_metric_choice_agent
from model_selection_agent import ModelChoice, run_model_selection_agent
from preprocessing_agent import ColumnPlan, run_preprocessing_agent
from preprocessing_persistence import persist_preprocessing_run
from profile_dataset import profile_dataset


Severity = Literal["info", "warn", "critical"]
CRITICAL_SEVERITY_METRICS = {
    "leakage_suspect",
    "missingness_leakage_suspect",
    "is_constant",
    "target:is_imbalanced",
    "has_high_missing",
}
WARN_SEVERITY_METRICS = {
    "is_multicollinear",
    "has_moderate_missing",
    "is_skewed",
    "missingness_informative",
}


@dataclass
class Finding(Claim):
    severity: Severity
    llm_stated_severity: Severity
    rule_derived_severity: Severity | None = None
    severity_rule_violation: bool | None = None
    evidence_verdicts: list[EvidenceVerdict] | None = None


SYSTEM_PROMPT = """You are an EDA agent for supervised tabular ML workflows.

Produce 5-12 Finding objects that identify the most important data quality, modeling, and leakage risks in the provided flat raw_profile dictionary.

Severity rules:
- critical: leakage_suspect, is_constant, target:is_imbalanced, has_high_missing
- warn: is_multicollinear, has_moderate_missing, is_skewed, missingness_informative
- info: everything else

Additional precedence rule: missingness_leakage_suspect is critical, overriding the "everything else" rule above.

missingness_target_corr measures association between a column's missingness indicator and the target. missingness_cluster_corr_max only measures coincidence with OTHER columns' missingness and must not be described as target association.

Do not compute new statistics. Only interpret and connect the provided profile keys. Every finding must cite at least one piece of evidence (a metric_key that exists in the profile you were given, and the value you understood it to be).

Use claim_id values in sequential order starting at eda_001. Every finding must use stage="eda". Keep decisions concise and actionable."""


EDA_FINDINGS_TOOL = {
    "name": "record_eda_findings",
    "description": "Record structured EDA findings derived only from the provided raw_profile.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["findings"],
        "properties": {
            "findings": {
                "type": "array",
                "minItems": 5,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "claim_id",
                        "stage",
                        "decision",
                        "justification",
                        "evidence",
                        "confidence",
                        "severity",
                    ],
                    "properties": {
                        "claim_id": {
                            "type": "string",
                            "pattern": "^eda_[0-9]{3}$",
                        },
                        "stage": {"type": "string", "const": "eda"},
                        "decision": {"type": "string"},
                        "justification": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["metric_key", "claimed_value"],
                                "properties": {
                                    "metric_key": {"type": "string"},
                                    "claimed_value": {
                                        "type": ["number", "string", "boolean", "null"],
                                    },
                                },
                            },
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["info", "warn", "critical"],
                        },
                    },
                },
            }
        },
    },
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _extract_tool_input(message: Any) -> dict[str, Any]:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "record_eda_findings":
            return block.input
    raise ValueError("Claude response did not include the required record_eda_findings tool call")


def _coerce_finding(payload: dict[str, Any], sequence_number: int) -> Finding:
    evidence = [
        EvidenceRef(
            metric_key=ref["metric_key"],
            claimed_value=ref.get("claimed_value"),
        )
        for ref in payload["evidence"]
    ]
    llm_stated_severity = payload["severity"]
    return Finding(
        claim_id=f"eda_{sequence_number:03d}",
        stage="eda",
        decision=payload["decision"],
        justification=payload["justification"],
        evidence=evidence,
        confidence=float(payload["confidence"]),
        severity=llm_stated_severity,
        llm_stated_severity=llm_stated_severity,
    )


def _metric_rule_name(metric_key: str) -> str:
    if metric_key == "target:is_imbalanced":
        return metric_key
    return metric_key.rsplit(":", 1)[-1]


def _severity_from_evidence(evidence: list[EvidenceRef]) -> Severity:
    metric_rule_names = {
        _metric_rule_name(ref.metric_key)
        for ref in evidence
        if ref.claimed_value is True
    }
    if metric_rule_names & CRITICAL_SEVERITY_METRICS:
        return "critical"
    if metric_rule_names & WARN_SEVERITY_METRICS:
        return "warn"
    return "info"


def _annotate_severity_rules(findings: list[Finding]) -> None:
    for finding in findings:
        rule_derived_severity = _severity_from_evidence(finding.evidence)
        finding.rule_derived_severity = rule_derived_severity
        finding.severity_rule_violation = finding.llm_stated_severity != rule_derived_severity
        finding.severity = rule_derived_severity


def severity_violation_rate(findings: list[Finding]) -> float | None:
    if not findings:
        return None
    violations = sum(1 for finding in findings if finding.severity_rule_violation)
    return violations / len(findings)


def run_eda_agent(
    raw_profile: dict,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    client: Anthropic | None = None,
) -> list[Finding]:
    """Generate EDA findings in one Claude tool-use call and validate evidence.

    The evidence verification step deliberately delegates to
    ``attach_evidence_verdicts_to_claims`` so Layer-1 evidence logic stays in the
    tested validator module.
    """
    anthropic_client = client or Anthropic()
    model_name = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    profile_json = json.dumps(raw_profile, default=_json_default, sort_keys=True)

    message = anthropic_client.messages.create(
        model=model_name,
        max_tokens=4000,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        tools=[EDA_FINDINGS_TOOL],
        tool_choice={"type": "tool", "name": "record_eda_findings"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Generate all EDA findings in a single tool call from this raw_profile JSON:\n"
                    f"{profile_json}"
                ),
            }
        ],
    )
    tool_input = _extract_tool_input(message)
    finding_payloads = tool_input["findings"]
    if not 5 <= len(finding_payloads) <= 12:
        raise ValueError(f"Expected 5-12 findings, got {len(finding_payloads)}")

    findings = [
        _coerce_finding(payload, sequence_number)
        for sequence_number, payload in enumerate(finding_payloads, start=1)
    ]
    _annotate_severity_rules(findings)
    attach_evidence_verdicts_to_claims(findings, raw_profile)
    return findings


def _print_findings(findings: list[Finding]) -> None:
    rate = severity_violation_rate(findings)
    print(f"severity_violation_rate: {rate:.4f}" if rate is not None else "severity_violation_rate: null")
    print()
    for finding in findings:
        print(f"{finding.claim_id} [{finding.severity}]")
        print(f"llm_stated_severity: {finding.llm_stated_severity}")
        print(f"rule_derived_severity: {finding.rule_derived_severity}")
        print(f"severity_rule_violation: {finding.severity_rule_violation}")
        print(f"decision: {finding.decision}")
        print(f"justification: {finding.justification}")
        print("evidence_verdicts:")
        for verdict in finding.evidence_verdicts or []:
            print(f"  {verdict}")
        print()


def _print_layer2_verdicts(layer2_verdicts) -> None:
    print("layer2_verdicts:")
    for verdict in layer2_verdicts:
        print(
            f"{verdict.claim_id}: "
            f"score={verdict.layer2_score} "
            f"note={verdict.discrepancy_note!r}"
        )
    print()


def _print_column_plans(column_plans: list[ColumnPlan]) -> None:
    print("column_plans:")
    for plan in column_plans:
        print(
            f"{plan.column}: "
            f"action={plan.action} "
            f"imputation={plan.imputation} "
            f"encoding={plan.encoding} "
            f"scaling={plan.scaling} "
            f"rule_derived_action={plan.rule_derived_action} "
            f"is_override={plan.is_override} "
            f"override_reason={plan.override_reason!r}"
        )
    print()


def _print_unfitted_pipeline(pipeline) -> None:
    column_transformer = pipeline.named_steps["preprocess"]
    exposes_fitted_transformers = hasattr(column_transformer, "transformers_")
    if exposes_fitted_transformers:
        raise RuntimeError("Preprocessing agent returned a fitted ColumnTransformer")
    print("pipeline_repr:")
    print(repr(pipeline))
    print(f"column_transformer_has_transformers_: {exposes_fitted_transformers}")
    print("pipeline_state: unfitted")
    print()


def _print_model_choices(choices: list[ModelChoice]) -> None:
    print("model_choices:")
    for choice in choices:
        label = "rejected" if choice.is_rejection else f"rank={choice.rank}"
        print(f"{choice.model_id} ({label}): {choice.justification}")
    print()


def _print_evaluation_results(results) -> None:
    print("evaluation_results:")
    for result in results:
        print(
            f"{result.model_id} rank={result.rank} "
            f"{result.primary_metric}={result.primary_metric_mean:.4f} +/- {result.primary_metric_std:.4f} "
            f"train_val_gap={result.train_val_gap:.4f} "
            f"overfitting_flag={result.overfitting_flag}"
        )
        for metric_name, summary in result.secondary_metrics.items():
            print(f"  {metric_name}={summary['mean']:.4f} +/- {summary['std']:.4f}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Claude-backed EDA agent on a CSV file.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="/Users/akhilgorla/Downloads/mini_dataset.csv",
        help="CSV path to profile before running the EDA agent.",
    )
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--task-type", default="binary")
    parser.add_argument("--model", default=None)
    parser.add_argument("--critic-model", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-layer2", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set in this shell. Run:\n"
            "  export ANTHROPIC_API_KEY='your_real_key_here'",
            file=sys.stderr,
        )
        return 2

    df = pd.read_csv(args.csv_path)
    raw_profile = profile_dataset(df, args.target_column, args.task_type)
    findings = run_eda_agent(raw_profile, model=args.model)
    run_id, run_dir, metrics = persist_layer1_run(
        findings,
        severity_violation_rate=severity_violation_rate(findings),
        run_id=args.run_id,
    )
    column_plans, preprocessing_pipeline = run_preprocessing_agent(
        raw_profile,
        df,
        args.target_column,
        model=args.model,
    )
    preprocessing_claim_verdicts, metrics = persist_preprocessing_run(
        run_dir,
        column_plans,
    )
    dropped_columns = [plan.column for plan in column_plans if plan.action == "drop"]
    retained_columns = [plan.column for plan in column_plans if plan.action != "drop"]
    model_choices = run_model_selection_agent(
        raw_profile,
        args.task_type,
        retained_columns=retained_columns,
        dropped_columns=dropped_columns,
        model=args.model,
    )
    model_claim_verdicts, metrics = append_layer1_claims(run_dir, model_choices)
    metric_choice = run_metric_choice_agent(
        raw_profile,
        args.task_type,
        retained_columns=retained_columns,
        dropped_columns=dropped_columns,
        model=args.model,
    )
    metric_claim_verdicts, metrics = append_layer1_claims(run_dir, [metric_choice])
    evaluation_results, stratification_degraded = evaluate_selected_models(
        model_choices,
        preprocessing_pipeline,
        df,
        args.target_column,
        args.task_type,
        metric_choice.primary_metric,
        metric_choice.secondary_metrics,
    )
    metrics = persist_evaluation_results(run_dir, evaluation_results, stratification_degraded)

    print(f"run_id: {run_id}")
    print(f"run_dir: {run_dir}")
    _print_column_plans(column_plans)
    _print_unfitted_pipeline(preprocessing_pipeline)
    _print_model_choices(model_choices)
    print(f"primary_metric: {metric_choice.primary_metric}")
    print(f"metric_justification: {metric_choice.justification}")
    print()
    _print_evaluation_results(evaluation_results)

    if args.task_type in {"binary", "multiclass", "classification"} and any(
        result.primary_metric_mean >= 0.98 for result in evaluation_results
    ):
        print("LEAKAGE_RED_FLAG: a model reported a near-perfect primary CV score; stopping for review.", file=sys.stderr)
        return 1

    if not args.skip_layer2:
        eda_claim_verdicts = rollup_claim_verdicts(findings)
        eda_layer2_verdicts = run_layer2_critic(
            eda_claim_verdicts,
            findings,
            raw_profile,
            model=args.critic_model,
        )
        persist_layer2_run(run_dir, eda_layer2_verdicts)

        preprocessing_layer2_verdicts = run_layer2_critic(
            preprocessing_claim_verdicts,
            column_plans,
            raw_profile,
            model=args.critic_model,
        )
        metrics = persist_layer2_run(
            run_dir,
            preprocessing_layer2_verdicts,
            append=True,
        )
        model_layer2_verdicts = run_layer2_critic(
            model_claim_verdicts,
            model_choices,
            raw_profile,
            model=args.critic_model,
        )
        persist_layer2_run(run_dir, model_layer2_verdicts, append=True)
        evaluation_layer2_verdicts = run_layer2_critic(
            metric_claim_verdicts,
            [metric_choice],
            raw_profile,
            model=args.critic_model,
        )
        metrics = persist_layer2_run(run_dir, evaluation_layer2_verdicts, append=True)
        print("preprocessing_layer2_verdicts:")
        for verdict in preprocessing_layer2_verdicts:
            print(
                f"{verdict.claim_id}: "
                f"score={verdict.layer2_score} "
                f"note={verdict.discrepancy_note!r}"
            )
        print()
        print("milestone7_layer2_verdicts:")
        for verdict in model_layer2_verdicts + evaluation_layer2_verdicts:
            print(f"{verdict.claim_id}: score={verdict.layer2_score} note={verdict.discrepancy_note!r}")
        print()

    print("metrics:")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print()

    _print_findings(findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
