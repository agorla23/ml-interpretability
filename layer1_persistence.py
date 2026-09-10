"""Persistence and aggregate metrics for Layer-1 evidence verdicts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from evidence_validator import EvidenceVerdict


Layer1Code = Literal["FABRICATED_EVIDENCE", "EVIDENCE_MISSTATED", "EVIDENCE_ACCURATE"]


@dataclass
class ClaimVerdict:
    claim_id: str
    stage: str
    citation_layer1_codes: list[str]
    claim_layer1_code: Layer1Code


def rollup_claim_verdict(finding) -> ClaimVerdict:
    citation_codes = [
        verdict.layer1_code
        for verdict in (getattr(finding, "evidence_verdicts", None) or [])
    ]
    if "FABRICATED_EVIDENCE" in citation_codes:
        claim_code: Layer1Code = "FABRICATED_EVIDENCE"
    elif "EVIDENCE_MISSTATED" in citation_codes:
        claim_code = "EVIDENCE_MISSTATED"
    else:
        claim_code = "EVIDENCE_ACCURATE"

    return ClaimVerdict(
        claim_id=finding.claim_id,
        stage=finding.stage,
        citation_layer1_codes=citation_codes,
        claim_layer1_code=claim_code,
    )


def rollup_claim_verdicts(findings) -> list[ClaimVerdict]:
    return [rollup_claim_verdict(finding) for finding in findings]


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_layer1_metrics(
    evidence_verdicts: list[EvidenceVerdict],
    claim_verdicts: list[ClaimVerdict],
    severity_violation_rate: float | None,
) -> dict:
    total_citations = len(evidence_verdicts)
    total_claims = len(claim_verdicts)
    fabricated_citations = sum(
        verdict.layer1_code == "FABRICATED_EVIDENCE"
        for verdict in evidence_verdicts
    )
    misstated_citations = sum(
        verdict.layer1_code == "EVIDENCE_MISSTATED"
        for verdict in evidence_verdicts
    )
    fabricated_claims = sum(
        verdict.claim_layer1_code == "FABRICATED_EVIDENCE"
        for verdict in claim_verdicts
    )
    misstated_claims = sum(
        verdict.claim_layer1_code == "EVIDENCE_MISSTATED"
        for verdict in claim_verdicts
    )

    return {
        "total_citations": total_citations,
        "total_claims": total_claims,
        "citation_fabrication_rate": _rate(fabricated_citations, total_citations),
        "citation_misstatement_rate": _rate(misstated_citations, total_citations),
        "claim_fabrication_rate": _rate(fabricated_claims, total_claims),
        "claim_misstatement_rate": _rate(misstated_claims, total_claims),
        "severity_violation_rate": severity_violation_rate,
    }


def flatten_evidence_verdicts(findings) -> list[EvidenceVerdict]:
    evidence_verdicts: list[EvidenceVerdict] = []
    for finding in findings:
        evidence_verdicts.extend(getattr(finding, "evidence_verdicts", None) or [])
    return evidence_verdicts


def persist_layer1_run(
    findings,
    *,
    severity_violation_rate: float | None,
    run_id: str | None = None,
    runs_dir: str | Path = "runs",
) -> tuple[str, Path, dict]:
    resolved_run_id = run_id or uuid4().hex
    run_dir = Path(runs_dir) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    evidence_verdicts = flatten_evidence_verdicts(findings)
    claim_verdicts = rollup_claim_verdicts(findings)
    metrics = compute_layer1_metrics(
        evidence_verdicts,
        claim_verdicts,
        severity_violation_rate,
    )

    _write_jsonl(run_dir / "evidence_verdicts.jsonl", evidence_verdicts)
    _write_jsonl(run_dir / "claim_verdicts.jsonl", claim_verdicts)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return resolved_run_id, run_dir, metrics


def append_layer1_claims(run_dir: str | Path, claims) -> tuple[list[ClaimVerdict], dict]:
    """Append claim verdict rows and recompute Layer-1 metrics across stages."""
    run_path = Path(run_dir)
    evidence_path = run_path / "evidence_verdicts.jsonl"
    claims_path = run_path / "claim_verdicts.jsonl"
    existing_evidence = _read_evidence_verdicts(evidence_path)
    existing_claims = _read_claim_verdicts(claims_path)
    new_evidence = flatten_evidence_verdicts(claims)
    new_claims = rollup_claim_verdicts(claims)
    _append_jsonl(evidence_path, new_evidence)
    _append_jsonl(claims_path, new_claims)

    metrics_path = run_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics.update(
        compute_layer1_metrics(
            existing_evidence + new_evidence,
            existing_claims + new_claims,
            metrics.get("severity_violation_rate"),
        )
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return new_claims, metrics


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def _append_jsonl(path: Path, rows) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def _read_evidence_verdicts(path: Path) -> list[EvidenceVerdict]:
    if not path.exists():
        return []
    return [EvidenceVerdict(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_claim_verdicts(path: Path) -> list[ClaimVerdict]:
    if not path.exists():
        return []
    return [ClaimVerdict(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
