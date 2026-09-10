"""Layer-2 LLM critic for claim reasoning support."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from evidence_validator import resolve_evidence_value
from layer1_persistence import ClaimVerdict


@dataclass
class Layer2Verdict:
    claim_id: str
    layer2_score: int
    discrepancy_note: str
    critic_confidence: float


CRITIC_SYSTEM_PROMPT = """You are a Layer-2 critic. You evaluate whether cited evidence supports a single EDA claim's stated decision.

You are NOT checking whether the cited evidence is factually accurate --
assume every number you are given is correct. That has already been
verified. You are ONLY evaluating the reasoning that connects the evidence
to the decision.

Work through these questions IN ORDER. Stop at the first one that applies.

QUESTION 1 — Does the justification mischaracterize what any cited evidence
actually shows?
This means the number is correct but the words describing it are false:
calling a near-zero correlation "strong," describing moderate missingness
as "severe," treating a weak signal as decisive. If yes, score 0 and stop.
A false description of the evidence invalidates the reasoning regardless of
whether the decision happens to be correct.

QUESTION 2 — Does the decision move in the WRONG DIRECTION for this column?
The evidence is described honestly, but the action taken is not the kind of
action this evidence calls for: dropping a column when the evidence calls
for transforming it, imputing when the evidence calls for dropping, keeping
when the evidence calls for removal. If the direction is wrong, score 1 and
stop — even if the error is mild or the decision sounds reasonable.

QUESTION 3 — The direction is right. Is the decision complete, and is the
cited evidence the evidence that actually supports it?
- Score 3 ONLY if BOTH are true: the decision fully addresses what the
  evidence calls for, AND the specific evidence cited is directly the
  evidence that justifies it.
- Score 2 if the direction is right but EITHER the decision is incomplete
  or partial (it addresses part of what the evidence calls for but misses
  something the evidence points to), OR the cited evidence is thin,
  indirect, or only tangentially related to the decision even though a
  stronger justification was available.

Summary:
0 = mischaracterizes what the evidence shows (description failure)
1 = evidence described honestly, but wrong direction (judgment failure)
2 = right direction, but incomplete OR thinly evidenced
3 = right direction, complete, and directly evidenced"""


LAYER2_TOOL = {
    "name": "record_layer2_verdict",
    "description": "Record a structured Layer-2 verdict for one isolated claim.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "claim_id",
            "layer2_score",
            "discrepancy_note",
            "critic_confidence",
        ],
        "properties": {
            "claim_id": {"type": "string"},
            "layer2_score": {"type": "integer", "enum": [0, 1, 2, 3]},
            "discrepancy_note": {"type": "string", "maxLength": 300},
            "critic_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
    },
}


def build_critic_context(claim, raw_profile: dict) -> dict:
    """Return only cited profile entries for a single claim's evidence keys."""
    return {
        ref.metric_key: resolve_evidence_value(ref.metric_key, raw_profile)
        for ref in claim.evidence
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _extract_tool_input(message: Any) -> dict[str, Any]:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "record_layer2_verdict":
            return block.input
    raise ValueError("Claude response did not include the required record_layer2_verdict tool call")


def _finding_by_claim_id(findings) -> dict[str, Any]:
    return {finding.claim_id: finding for finding in findings}


def _call_layer2_critic(claim, raw_profile: dict, *, client: Anthropic, model: str) -> Layer2Verdict:
    context = build_critic_context(claim, raw_profile)
    isolated_payload = {
        "decision": claim.decision,
        "justification": claim.justification,
        "evidence_profile_subset": context,
    }
    message = client.messages.create(
        model=model,
        max_tokens=800,
        temperature=0,
        system=CRITIC_SYSTEM_PROMPT,
        tools=[LAYER2_TOOL],
        tool_choice={"type": "tool", "name": "record_layer2_verdict"},
        messages=[
            {
                "role": "user",
                "content": json.dumps(isolated_payload, default=_json_default, sort_keys=True),
            }
        ],
    )
    tool_input = _extract_tool_input(message)
    layer2_score = int(tool_input["layer2_score"])
    discrepancy_note = "" if layer2_score == 3 else str(tool_input["discrepancy_note"])[:300]
    return Layer2Verdict(
        claim_id=claim.claim_id,
        layer2_score=layer2_score,
        discrepancy_note=discrepancy_note,
        critic_confidence=float(tool_input["critic_confidence"]),
    )


def run_layer2_critic(
    claim_verdicts: list[ClaimVerdict],
    findings: list,
    raw_profile: dict,
    *,
    model: str | None = None,
    client: Anthropic | None = None,
) -> list[Layer2Verdict]:
    """Score Layer-1-accurate claims with isolated one-claim critic calls."""
    anthropic_client = client or Anthropic()
    model_name = model or os.environ.get("ANTHROPIC_CRITIC_MODEL") or os.environ.get(
        "ANTHROPIC_MODEL",
        "claude-sonnet-4-5",
    )
    findings_by_id = _finding_by_claim_id(findings)
    eligible_claim_ids = [
        verdict.claim_id
        for verdict in claim_verdicts
        if verdict.claim_layer1_code == "EVIDENCE_ACCURATE"
    ]

    layer2_verdicts: list[Layer2Verdict] = []
    for claim_id in eligible_claim_ids:
        if claim_id not in findings_by_id:
            raise KeyError(f"No Claim available for Layer-1 accurate claim_id {claim_id!r}")
        layer2_verdicts.append(
            _call_layer2_critic(
                findings_by_id[claim_id],
                raw_profile,
                client=anthropic_client,
                model=model_name,
            )
        )
    return layer2_verdicts


def persist_layer2_run(
    run_dir: str | Path,
    layer2_verdicts: list[Layer2Verdict],
    *,
    append: bool = False,
) -> dict:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    verdicts_path = run_path / "layer2_verdicts.jsonl"
    existing_verdicts = _read_layer2_verdicts(verdicts_path) if append else []
    if append:
        _append_jsonl(verdicts_path, layer2_verdicts)
    else:
        _write_jsonl(verdicts_path, layer2_verdicts)
    combined_verdicts = existing_verdicts + layer2_verdicts

    metrics_path = run_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    claim_stages = _read_claim_stages(run_path / "claim_verdicts.jsonl")
    metrics.update(compute_layer2_metrics(combined_verdicts, claim_stages))
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def compute_layer2_metrics(
    layer2_verdicts: list[Layer2Verdict],
    claim_stages: dict[str, str] | None = None,
) -> dict:
    count = len(layer2_verdicts)
    mean_score = sum(verdict.layer2_score for verdict in layer2_verdicts) / count if count else None
    mischaracterizations = sum(verdict.layer2_score == 0 for verdict in layer2_verdicts)
    wrong_direction = sum(verdict.layer2_score == 1 for verdict in layer2_verdicts)
    scores_by_stage: dict[str, list[int]] = {}
    if claim_stages is not None:
        scores_by_stage = {stage: [] for stage in sorted(set(claim_stages.values()))}
        for verdict in layer2_verdicts:
            stage = claim_stages.get(verdict.claim_id)
            if stage is not None:
                scores_by_stage.setdefault(stage, []).append(verdict.layer2_score)

    return {
        "mean_layer2_score": mean_score,
        "mean_layer2_score_by_stage": {
            stage: sum(scores) / len(scores) if scores else None
            for stage, scores in scores_by_stage.items()
        },
        "layer2_claims_scored": count,
        "mischaracterization_rate": mischaracterizations / count if count else None,
        "wrong_direction_rate": wrong_direction / count if count else None,
        "layer2_scale": "0-3",
    }


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def _append_jsonl(path: Path, rows) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def _read_layer2_verdicts(path: Path) -> list[Layer2Verdict]:
    if not path.exists():
        return []
    return [
        Layer2Verdict(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_claim_stages(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        record["claim_id"]: record["stage"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
    }
