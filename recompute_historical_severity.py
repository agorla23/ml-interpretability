"""Recompute historical EDA severity violations without modifying run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eda_agent import _severity_from_evidence
from evidence_validator import EvidenceRef


DATASETS = ("titanic", "wine_quality", "adult_income")


def _load_eda_claims(run_dir: Path) -> list[dict[str, Any]]:
    claims = [
        json.loads(line)
        for line in (run_dir / "claims.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    return [claim for claim in claims if claim.get("stage") == "eda"]


def recompute_run(run_dir: Path) -> dict[str, str | float | int]:
    """Return stored and value-aware violation rates for one completed run."""
    claims = _load_eda_claims(run_dir)
    if not claims:
        raise ValueError(f"No EDA claims found in {run_dir / 'claims.jsonl'}")

    old_violations = sum(bool(claim["severity_rule_violation"]) for claim in claims)
    corrected_violations = 0
    for claim in claims:
        evidence = [EvidenceRef(**item) for item in claim["evidence"]]
        corrected_severity = _severity_from_evidence(evidence)
        corrected_violations += claim["llm_stated_severity"] != corrected_severity

    return {
        "run_id": run_dir.name,
        "eda_claims": len(claims),
        "old_severity_violation_rate": old_violations / len(claims),
        "corrected_severity_violation_rate": corrected_violations / len(claims),
    }


def recompute_completed_runs(runs_dir: str | Path = "runs") -> list[dict[str, str | float | int]]:
    """Recompute the four Arm A/B runs for each selected benchmark dataset."""
    root = Path(runs_dir)
    rows: list[dict[str, str | float | int]] = []
    for dataset in DATASETS:
        run_dirs = sorted(root.glob(f"m9_{dataset}_ab_*"))
        if len(run_dirs) != 4:
            raise ValueError(f"Expected 4 completed {dataset} runs, found {len(run_dirs)}")
        rows.extend(recompute_run(run_dir) for run_dir in run_dirs)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    print("run_id\told_severity_violation_rate\tcorrected_severity_violation_rate")
    for row in recompute_completed_runs(args.runs_dir):
        print(
            f"{row['run_id']}\t"
            f"{row['old_severity_violation_rate']:.6f}\t"
            f"{row['corrected_severity_violation_rate']:.6f}"
        )


if __name__ == "__main__":
    main()
