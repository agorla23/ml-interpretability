"""Re-score historical preprocessing overrides without rerunning a pipeline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from anthropic import Anthropic

from evidence_validator import EvidenceRef, EvidenceVerdict
from layer2_critic import _call_layer2_critic
from preprocessing_agent import ColumnPlan, _final_override_justification


DEFAULT_RUN_IDS = (
    "m9_wine_quality_ab_t00_r1",
    "m9_wine_quality_ab_t00_r2",
    "m9_wine_quality_ab_t07_r1",
)
DEFAULT_CLAIM_IDS = ("prep_006", "prep_007", "prep_010")
DEFAULT_OUTPUT = Path("rescore_prep_override_fix.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _plan_from_record(record: dict[str, Any]) -> ColumnPlan:
    return ColumnPlan(
        claim_id=record["claim_id"],
        stage=record["stage"],
        decision=record["decision"],
        justification=record["justification"],
        evidence=[EvidenceRef(**item) for item in record["evidence"]],
        confidence=float(record["confidence"]),
        column=record["column"],
        action=record["action"],
        imputation=record["imputation"],
        encoding=record["encoding"],
        scaling=record["scaling"],
        rule_derived_action=record["rule_derived_action"],
        override_reason=record["override_reason"],
        is_override=bool(record["is_override"]),
        evidence_verdicts=[
            EvidenceVerdict(**item) for item in record.get("evidence_verdicts", [])
        ],
    )


def _records_by_claim_id(path: Path) -> dict[str, dict[str, Any]]:
    return {record["claim_id"]: record for record in _read_jsonl(path)}


def _corrected_claim(record: dict[str, Any]) -> tuple[ColumnPlan, str]:
    plan = _plan_from_record(record)
    old_justification = plan.justification
    before = asdict(plan)
    plan.justification = _final_override_justification(plan)
    after = asdict(plan)

    before.pop("justification")
    after.pop("justification")
    if before != after:
        raise AssertionError("Historical claim reconstruction changed fields besides justification")
    return plan, old_justification


def rescore_historical_overrides(
    *,
    run_ids: Iterable[str] = DEFAULT_RUN_IDS,
    claim_ids: Iterable[str] = DEFAULT_CLAIM_IDS,
    runs_root: str | Path = "runs",
    output_path: str | Path = DEFAULT_OUTPUT,
    client: Anthropic | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Re-score selected claims and atomically write a separate JSONL artifact.

    The existing single-claim critic is deliberately used unchanged. Its
    temperature remains fixed at zero, including for claims created during a
    stage-agent run whose temperature was 0.7.
    """
    root = Path(runs_root)
    selected_claim_ids = tuple(claim_ids)
    anthropic_client = client or Anthropic()
    model_name = model or os.environ.get("ANTHROPIC_CRITIC_MODEL") or os.environ.get(
        "ANTHROPIC_MODEL", "claude-sonnet-4-5"
    )
    results: list[dict[str, Any]] = []

    for run_id in run_ids:
        run_dir = root / run_id
        plans = _records_by_claim_id(run_dir / "column_plans.jsonl")
        old_verdicts = _records_by_claim_id(run_dir / "layer2_verdicts.jsonl")
        raw_profile = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))[
            "raw_profile"
        ]

        for claim_id in selected_claim_ids:
            if claim_id not in plans:
                raise KeyError(f"{claim_id!r} is missing from {run_dir / 'column_plans.jsonl'}")
            if claim_id not in old_verdicts:
                raise KeyError(f"{claim_id!r} is missing from {run_dir / 'layer2_verdicts.jsonl'}")

            corrected_claim, old_justification = _corrected_claim(plans[claim_id])
            verdict = _call_layer2_critic(
                corrected_claim,
                raw_profile,
                client=anthropic_client,
                model=model_name,
            )
            results.append(
                {
                    "run_id": run_id,
                    "claim_id": claim_id,
                    "old_justification": old_justification,
                    "new_justification": corrected_claim.justification,
                    "old_layer2_score": int(old_verdicts[claim_id]["layer2_score"]),
                    "new_layer2_score": verdict.layer2_score,
                    "new_discrepancy_note": verdict.discrepancy_note,
                }
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    temporary_output.write_text(
        "".join(json.dumps(result, sort_keys=True) + "\n" for result in results),
        encoding="utf-8",
    )
    temporary_output.replace(output)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model")
    args = parser.parse_args()

    results = rescore_historical_overrides(
        runs_root=args.runs_root,
        output_path=args.output,
        model=args.model,
    )
    for result in results:
        print(json.dumps(result, sort_keys=True))
    print(f"wrote {len(results)} rescored claims to {args.output}")


if __name__ == "__main__":
    main()
