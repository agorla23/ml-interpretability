"""Persistence and aggregate metrics for preprocessing plans."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evidence_validator import EvidenceVerdict
from layer1_persistence import (
    ClaimVerdict,
    compute_layer1_metrics,
    flatten_evidence_verdicts,
    rollup_claim_verdicts,
)
from preprocessing_agent import ColumnPlan, override_rate


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _column_plan_record(plan: ColumnPlan) -> dict[str, Any]:
    return {
        "claim_id": plan.claim_id,
        "stage": plan.stage,
        "decision": plan.decision,
        "justification": plan.justification,
        "evidence": [asdict(reference) for reference in plan.evidence],
        "confidence": plan.confidence,
        "column": plan.column,
        "action": plan.action,
        "imputation": plan.imputation,
        "encoding": plan.encoding,
        "scaling": plan.scaling,
        "rule_derived_action": plan.rule_derived_action,
        "is_override": plan.is_override,
        "override_reason": plan.override_reason,
    }


def _read_evidence_verdicts(path: Path) -> list[EvidenceVerdict]:
    if not path.exists():
        return []
    return [
        EvidenceVerdict(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_claim_verdicts(path: Path) -> list[ClaimVerdict]:
    if not path.exists():
        return []
    return [
        ClaimVerdict(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, rows) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), default=_json_default, sort_keys=True) + "\n")


def persist_preprocessing_run(
    run_dir: str | Path,
    column_plans: list[ColumnPlan],
) -> tuple[list[ClaimVerdict], dict]:
    """Persist plans, append Layer-1 rows, and recompute combined metrics."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    plans_path = run_path / "column_plans.jsonl"
    with plans_path.open("w", encoding="utf-8") as handle:
        for plan in column_plans:
            handle.write(
                json.dumps(_column_plan_record(plan), default=_json_default, sort_keys=True)
                + "\n"
            )

    evidence_path = run_path / "evidence_verdicts.jsonl"
    claims_path = run_path / "claim_verdicts.jsonl"
    existing_evidence = _read_evidence_verdicts(evidence_path)
    existing_claims = _read_claim_verdicts(claims_path)
    preprocessing_evidence = flatten_evidence_verdicts(column_plans)
    preprocessing_claims = rollup_claim_verdicts(column_plans)

    _append_jsonl(evidence_path, preprocessing_evidence)
    _append_jsonl(claims_path, preprocessing_claims)

    metrics_path = run_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    combined_layer1_metrics = compute_layer1_metrics(
        existing_evidence + preprocessing_evidence,
        existing_claims + preprocessing_claims,
        metrics.get("severity_violation_rate"),
    )
    metrics.update(combined_layer1_metrics)
    metrics.update(
        {
            "override_rate": override_rate(column_plans),
            "columns_dropped": sum(plan.action == "drop" for plan in column_plans),
            "columns_kept": sum(plan.action != "drop" for plan in column_plans),
            "preprocess_claims": len(column_plans),
        }
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return preprocessing_claims, metrics
