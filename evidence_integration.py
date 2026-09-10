"""Helpers for attaching Layer-1 evidence verdicts to agent claims."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from evidence_validator import Claim, EvidenceVerdict, validate_evidence


def _as_validator_claim(claim: Claim | dict[str, Any]) -> Claim | SimpleNamespace:
    if not isinstance(claim, dict):
        return claim

    evidence = [
        SimpleNamespace(**ref) if isinstance(ref, dict) else ref
        for ref in claim.get("evidence", [])
    ]
    return SimpleNamespace(**{**claim, "evidence": evidence})


def attach_evidence_verdicts(claim: Claim | dict[str, Any], raw_profile: dict) -> list[EvidenceVerdict]:
    """Validate a claim's evidence and store verdicts alongside the claim.

    Future agents should call this immediately after producing any Finding or
    Claim that has an ``evidence`` list. Dataclass-style claims receive an
    ``evidence_verdicts`` attribute; dict-style claims receive an
    ``"evidence_verdicts"`` key. The returned list is the same object stored on
    the claim and can later be serialized to ``verdicts.jsonl``.
    """
    verdicts = validate_evidence(_as_validator_claim(claim), raw_profile)

    if isinstance(claim, dict):
        claim["evidence_verdicts"] = verdicts
    else:
        setattr(claim, "evidence_verdicts", verdicts)

    return verdicts


def attach_evidence_verdicts_to_claims(
    claims: list[Claim | dict[str, Any]], raw_profile: dict
) -> list[list[EvidenceVerdict]]:
    """Validate and attach evidence verdicts for a batch of agent claims."""
    return [attach_evidence_verdicts(claim, raw_profile) for claim in claims]
