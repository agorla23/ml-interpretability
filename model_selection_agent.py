"""Structured LLM model-choice claims over the fixed Milestone 7 model zoo."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic

from evidence_integration import attach_evidence_verdicts_to_claims
from evidence_validator import Claim, EvidenceRef, EvidenceVerdict


CLASSIFICATION_MODELS = ("logreg", "rf", "gbt", "knn")
REGRESSION_MODELS = ("linreg", "rf", "gbt", "knn")


@dataclass
class ModelChoice(Claim):
    model_id: str
    rank: int | None
    is_rejection: bool
    evidence_verdicts: list[EvidenceVerdict] | None = None


MODEL_SELECTION_SYSTEM_PROMPT = """You select models for a supervised tabular dataset.

Use only the supplied raw_profile and preprocessing context for reasoning. Select
exactly three permitted models and rank them 1, 2, and 3, then reject the one
remaining permitted model. Each claim must
cite at least two provided profile keys. Each justification must address: data-shape
fit (n_rows, retained-column count, and retained dtype mix), signal-structure fit (corr_with_target and
multicollinearity flags), and the interpretability versus performance tradeoff.
Do not compute new statistics or propose tuning. The model zoo and task-specific
permitted IDs are fixed in the user payload.

You may ONLY cite metric_keys that appear in the raw_profile dict you were
given. This includes facts you may already know from elsewhere in the pipeline
(e.g. the task type used to choose the model zoo): if a fact is not a key in
raw_profile, do not cite it as evidence, even if it is true. Every cited key
must directly support the model-selection decision; do not cite a profile key
merely because it exists."""


def _preprocessing_context(dropped_columns: list[str], retained_columns: list[str]) -> str:
    return f"""The raw_profile you are given describes the ORIGINAL dataset, before
preprocessing. The following columns have ALREADY BEEN DROPPED by the
preprocessing stage and are NOT part of the data your model will actually
be trained on: {dropped_columns}. Do not cite flags belonging to dropped
columns as live concerns, evidence, or justification for anything -- they
no longer apply. Only these columns remain in the modeling dataset:
{retained_columns}. Base your reasoning about data shape, signal
structure, and feature counts on the RETAINED columns only.

The retained pre-encoding column count is {len(retained_columns)}. This count is
prompt-only context, not a raw_profile metric, so do not cite it as an
EvidenceRef. Do not cite raw_profile[\"n_cols\"] for feature-count reasoning:
it is the original dataframe column count and does not describe the modeling
features or one-hot-expanded matrix."""


def _allowed_model_evidence(raw_profile: dict, retained_columns: list[str]) -> dict:
    return {
        key: value
        for key, value in raw_profile.items()
        if key != "n_cols"
        and (
            not key.startswith("col:")
            or any(key.startswith(f"col:{column}:") for column in retained_columns)
        )
    }


MODEL_SELECTION_TOOL = {
    "name": "record_model_choices",
    "description": "Record exactly three ranked model selections and one rejection.",
    "input_schema": {
        "type": "object", "additionalProperties": False, "required": ["choices"],
        "properties": {"choices": {"type": "array", "minItems": 4, "maxItems": 4, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["model_id", "rank", "is_rejection", "decision", "justification", "evidence", "confidence"],
            "properties": {
                "model_id": {"type": "string"}, "rank": {"type": ["integer", "null"]},
                "is_rejection": {"type": "boolean"}, "decision": {"type": "string"},
                "justification": {"type": "string"}, "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {"type": "array", "minItems": 2, "items": {"type": "object", "additionalProperties": False,
                    "required": ["metric_key", "claimed_value"], "properties": {"metric_key": {"type": "string"}, "claimed_value": {"type": ["number", "string", "boolean", "object", "null"]}}}}
            }
        }}}
    }
}


def permitted_model_ids(task_type: str) -> tuple[str, ...]:
    if task_type in {"binary", "multiclass", "classification"}:
        return CLASSIFICATION_MODELS
    if task_type == "regression":
        return REGRESSION_MODELS
    raise ValueError(f"Unsupported task_type: {task_type!r}")


def _extract_tool_input(message: Any) -> dict[str, Any]:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "record_model_choices":
            return block.input
    raise ValueError("Claude response did not include record_model_choices")


def run_model_selection_agent(
    raw_profile: dict,
    task_type: str,
    *,
    retained_columns: list[str],
    dropped_columns: list[str],
    model: str | None = None,
    temperature: float = 0.0,
    client: Anthropic | None = None,
) -> list[ModelChoice]:
    """Generate four validated model-choice claims in one structured LLM call."""
    permitted = permitted_model_ids(task_type)
    allowed_profile = _allowed_model_evidence(raw_profile, retained_columns)
    anthropic_client = client or Anthropic()
    tool = deepcopy(MODEL_SELECTION_TOOL)
    evidence_key_schema = tool["input_schema"]["properties"]["choices"]["items"]["properties"]["evidence"]["items"]["properties"]["metric_key"]
    evidence_key_schema["enum"] = list(allowed_profile)
    message = anthropic_client.messages.create(
        model=model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"), max_tokens=4000, temperature=temperature,
        system=f"{MODEL_SELECTION_SYSTEM_PROMPT}\n\n{_preprocessing_context(dropped_columns, retained_columns)}", tools=[tool],
        tool_choice={"type": "tool", "name": "record_model_choices"},
        messages=[{"role": "user", "content": json.dumps({"task_type": task_type, "permitted_model_ids": permitted, "dropped_columns": dropped_columns, "retained_columns": retained_columns, "retained_column_count": len(retained_columns), "allowed_evidence_keys": list(allowed_profile), "raw_profile": raw_profile}, default=lambda v: v.item() if hasattr(v, "item") else str(v), sort_keys=True)}],
    )
    payloads = _extract_tool_input(message)["choices"]
    invalid_evidence_keys = {
        ref["metric_key"]
        for item in payloads
        for ref in item["evidence"]
        if ref["metric_key"] not in allowed_profile
    }
    if invalid_evidence_keys:
        raise ValueError(f"Model selection cited stale or unavailable keys: {sorted(invalid_evidence_keys)}")
    if len(payloads) != 4 or {item["model_id"] for item in payloads} != set(permitted):
        raise ValueError("Model selection must cover each permitted model exactly once")
    selected = [item for item in payloads if not item["is_rejection"]]
    rejected = [item for item in payloads if item["is_rejection"]]
    if len(selected) != 3 or len(rejected) != 1 or {item["rank"] for item in selected} != {1, 2, 3} or rejected[0]["rank"] is not None:
        raise ValueError("Model selection must contain ranks 1-3 and one unranked rejection")
    choices = [
        ModelChoice(
            claim_id=f"model_select_{index:03d}", stage="model_selection", model_id=item["model_id"], rank=item["rank"], is_rejection=bool(item["is_rejection"]),
            decision=item["decision"], justification=item["justification"],
            evidence=[EvidenceRef(ref["metric_key"], ref.get("claimed_value")) for ref in item["evidence"]], confidence=float(item["confidence"]),
        ) for index, item in enumerate(payloads, start=1)
    ]
    attach_evidence_verdicts_to_claims(choices, raw_profile)
    return choices
