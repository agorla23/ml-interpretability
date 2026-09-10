"""Deterministic preprocessing rules plus LLM-written plan justifications."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from anthropic import Anthropic

from evidence_integration import attach_evidence_verdicts_to_claims
from evidence_validator import Claim, EvidenceRef, EvidenceVerdict

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


Action = Literal[
    "drop",
    "impute",
    "encode",
    "scale",
    "impute_and_encode",
    "impute_and_scale",
    "passthrough",
]
Imputation = Literal["mean", "median", "mode", "knn", "constant", "none"]
Encoding = Literal["onehot", "target", "ordinal", "none"]
Scaling = Literal["standard", "minmax", "robust", "none"]

OVERRIDE_CAP = 3
PREPROCESSING_BATCH_SIZE = 15


def preprocessing_batch_count(n_columns: int) -> int:
    if n_columns < 0:
        raise ValueError("n_columns must be non-negative")
    return (n_columns + PREPROCESSING_BATCH_SIZE - 1) // PREPROCESSING_BATCH_SIZE


@dataclass
class ColumnPlan(Claim):
    column: str
    action: Action
    imputation: Imputation
    encoding: Encoding
    scaling: Scaling
    rule_derived_action: str
    override_reason: str
    is_override: bool
    evidence_verdicts: list[EvidenceVerdict] | None = None


@dataclass
class RuleDefault:
    column: str
    effective_type: Literal["numeric", "categorical"]
    action: Action
    imputation: Imputation
    encoding: Encoding
    scaling: Scaling
    drop_reason: str
    evidence_keys: list[str]


PREPROCESSING_SYSTEM_PROMPT = """You are a preprocessing-plan explanation agent.

The preprocessing strategy is already chosen by deterministic rules. Do not choose a new strategy by default. For each provided column default, write a concise justification that cites profile evidence keys and their values.

missingness_informative means the column's absence is associated with the target. missingness_cluster_corr_max instead measures whether its absence coincides with other columns' missingness and does not establish target relevance. A non-drop column reaching informative-missingness imputation is guaranteed not to be missingness_leakage_suspect because that higher-severity condition is handled earlier by an ordered drop rule.

You may optionally override imputation, encoding, or scaling only. You may never override a drop action. Use at most three overrides across the whole run. Every plan must cite at least one evidence item whose metric_key exists in the supplied profile subset."""


PREPROCESSING_TOOL = {
    "name": "record_preprocessing_plans",
    "description": "Record preprocessing plan justifications and optional non-drop overrides.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["plans"],
        "properties": {
            "plans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "column",
                        "justification",
                        "evidence",
                        "confidence",
                        "override_reason",
                        "imputation",
                        "encoding",
                        "scaling",
                    ],
                    "properties": {
                        "column": {"type": "string"},
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
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "override_reason": {"type": "string"},
                        "imputation": {
                            "type": "string",
                            "enum": ["mean", "median", "mode", "knn", "constant", "none"],
                        },
                        "encoding": {
                            "type": "string",
                            "enum": ["onehot", "target", "ordinal", "none"],
                        },
                        "scaling": {
                            "type": "string",
                            "enum": ["standard", "minmax", "robust", "none"],
                        },
                    },
                },
            }
        },
    },
}


def _profile_key(column: str, metric: str) -> str:
    return f"col:{column}:{metric}"


def _profile_value(raw_profile: dict, column: str, metric: str, default: Any = None) -> Any:
    return raw_profile.get(_profile_key(column, metric), default)


def _is_categorical_dtype_string(dtype: str) -> bool:
    return dtype == "object" or dtype == "category" or dtype == "bool"


def _column_order_from_profile(raw_profile: dict) -> list[str]:
    columns: list[str] = []
    for key in raw_profile:
        if not key.startswith("col:"):
            continue
        _, column, metric = key.split(":", 2)
        if metric == "dtype" and column not in columns:
            columns.append(column)
    return columns


def _duplicate_drop_columns(raw_profile: dict, columns: list[str]) -> set[str]:
    index_by_column = {column: index for index, column in enumerate(columns)}
    duplicate_groups: list[set[str]] = []
    for column in columns:
        duplicate_of = _profile_value(raw_profile, column, "duplicate_of")
        if duplicate_of is None:
            continue
        group = {column, duplicate_of}
        for existing_group in duplicate_groups:
            if existing_group & group:
                existing_group.update(group)
                break
        else:
            duplicate_groups.append(group)

    to_drop: set[str] = set()
    for group in duplicate_groups:
        ordered_group = sorted(group, key=lambda col: index_by_column.get(col, len(columns)))
        to_drop.update(ordered_group[1:])
    return to_drop


def _effective_type(raw_profile: dict, column: str) -> Literal["numeric", "categorical"]:
    dtype = str(_profile_value(raw_profile, column, "dtype", ""))
    is_categorical = _is_categorical_dtype_string(dtype)
    return (
        "categorical"
        if is_categorical or bool(_profile_value(raw_profile, column, "is_numeric_but_categorical", False))
        else "numeric"
    )


def _imputation_choice(raw_profile: dict, column: str, effective_type: str) -> Imputation:
    """Choose imputation after ordered leakage and deletion rules have passed.

    A column reaching this function with ``missingness_informative=True`` is
    guaranteed not to be ``missingness_leakage_suspect``: the latter is dropped
    before imputation. Constant imputation and an indicator therefore remain the
    policy for target-informative but non-extreme missingness.
    """
    if float(_profile_value(raw_profile, column, "missing_pct", 0.0) or 0.0) == 0.0:
        return "none"

    missingness_informative = bool(_profile_value(raw_profile, column, "missingness_informative", False))
    if effective_type == "numeric":
        if missingness_informative:
            return "constant"
        if bool(_profile_value(raw_profile, column, "is_skewed", False)):
            return "median"
        return "mean"

    if missingness_informative:
        return "constant"
    if float(_profile_value(raw_profile, column, "top_category_pct", 0.0) or 0.0) > 50.0:
        return "mode"
    return "constant"


def _encoding_choice(raw_profile: dict, column: str, effective_type: str) -> Encoding:
    if effective_type == "numeric":
        return "none"
    nunique = int(_profile_value(raw_profile, column, "nunique", 0) or 0)
    if nunique == 2:
        return "ordinal"
    if nunique <= 15:
        return "onehot"
    return "target"


def _scaling_choice(raw_profile: dict, column: str, effective_type: str) -> Scaling:
    if effective_type == "categorical":
        return "none"
    if bool(_profile_value(raw_profile, column, "is_zero_inflated", False)):
        return "robust"
    if float(_profile_value(raw_profile, column, "iqr_outlier_pct", 0.0) or 0.0) > 5.0:
        return "robust"
    return "standard"


def _action_from_transforms(imputation: Imputation, encoding: Encoding, scaling: Scaling) -> Action:
    has_imputation = imputation != "none"
    has_encoding = encoding != "none"
    has_scaling = scaling != "none"
    if has_imputation and has_encoding:
        return "impute_and_encode"
    if has_imputation and has_scaling:
        return "impute_and_scale"
    if has_imputation:
        return "impute"
    if has_encoding:
        return "encode"
    if has_scaling:
        return "scale"
    return "passthrough"


def derive_column_plan_defaults(
    raw_profile: dict,
    target_column: str,
    columns: list[str] | None = None,
) -> list[RuleDefault]:
    columns = columns or _column_order_from_profile(raw_profile)
    duplicate_drops = _duplicate_drop_columns(raw_profile, columns)
    defaults: list[RuleDefault] = []

    for column in columns:
        if column == target_column:
            continue

        effective_type = _effective_type(raw_profile, column)
        evidence_keys = [_profile_key(column, "dtype"), _profile_key(column, "nunique")]

        drop_reason = ""
        if bool(_profile_value(raw_profile, column, "leakage_suspect", False)):
            drop_reason = "leakage_suspect"
            evidence_keys.append(_profile_key(column, "leakage_suspect"))
        elif bool(_profile_value(raw_profile, column, "missingness_leakage_suspect", False)):
            drop_reason = "missingness_leakage_suspect"
            evidence_keys.extend(
                [
                    _profile_key(column, "missingness_leakage_suspect"),
                    _profile_key(column, "missingness_target_corr"),
                ]
            )
        elif bool(_profile_value(raw_profile, column, "is_constant", False)):
            drop_reason = "is_constant"
            evidence_keys.append(_profile_key(column, "is_constant"))
        elif bool(_profile_value(raw_profile, column, "is_low_variance", False)):
            drop_reason = "is_low_variance"
            evidence_keys.append(_profile_key(column, "is_low_variance"))
        elif bool(_profile_value(raw_profile, column, "is_id_like", False)):
            drop_reason = "is_id_like"
            evidence_keys.append(_profile_key(column, "is_id_like"))
        elif column in duplicate_drops:
            drop_reason = "is_duplicate_of_other_column"
            evidence_keys.extend(
                [
                    _profile_key(column, "is_duplicate_of_other_column"),
                    _profile_key(column, "duplicate_of"),
                ]
            )
        elif (
            bool(_profile_value(raw_profile, column, "has_high_missing", False))
            and not bool(_profile_value(raw_profile, column, "missingness_informative", False))
        ):
            drop_reason = "has_high_missing"
            evidence_keys.extend(
                [
                    _profile_key(column, "has_high_missing"),
                    _profile_key(column, "missingness_informative"),
                ]
            )

        if drop_reason:
            defaults.append(
                RuleDefault(
                    column=column,
                    effective_type=effective_type,
                    action="drop",
                    imputation="none",
                    encoding="none",
                    scaling="none",
                    drop_reason=drop_reason,
                    evidence_keys=list(dict.fromkeys(evidence_keys)),
                )
            )
            continue

        imputation = _imputation_choice(raw_profile, column, effective_type)
        encoding = _encoding_choice(raw_profile, column, effective_type)
        scaling = _scaling_choice(raw_profile, column, effective_type)
        evidence_keys.extend(
            [
                _profile_key(column, "missing_pct"),
                _profile_key(column, "missingness_informative"),
                _profile_key(column, "is_skewed"),
                _profile_key(column, "top_category_pct"),
                _profile_key(column, "iqr_outlier_pct"),
                _profile_key(column, "is_zero_inflated"),
                _profile_key(column, "nonzero_pct"),
                _profile_key(column, "is_numeric_but_categorical"),
            ]
        )
        defaults.append(
            RuleDefault(
                column=column,
                effective_type=effective_type,
                action=_action_from_transforms(imputation, encoding, scaling),
                imputation=imputation,
                encoding=encoding,
                scaling=scaling,
                drop_reason="",
                evidence_keys=[key for key in dict.fromkeys(evidence_keys) if key in raw_profile],
            )
        )

    return defaults


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _extract_tool_input(message: Any) -> dict[str, Any]:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "record_preprocessing_plans":
            return block.input
    raise ValueError("Claude response did not include the required record_preprocessing_plans tool call")


def _default_justification(default: RuleDefault, raw_profile: dict) -> tuple[str, list[EvidenceRef]]:
    evidence = [
        EvidenceRef(metric_key=key, claimed_value=raw_profile[key])
        for key in default.evidence_keys
        if key in raw_profile
    ]
    if default.action == "drop":
        justification = f"Deterministic drop rule matched first: {default.drop_reason}."
    else:
        justification = (
            f"Deterministic rules treat {default.column!r} as {default.effective_type}; "
            f"imputation={default.imputation}, encoding={default.encoding}, scaling={default.scaling}."
        )
    return justification, evidence


def _make_plan_from_default(
    default: RuleDefault,
    raw_profile: dict,
    sequence_number: int,
) -> ColumnPlan:
    justification, evidence = _default_justification(default, raw_profile)
    return ColumnPlan(
        claim_id=f"prep_{sequence_number:03d}",
        stage="preprocess",
        decision=f"{default.action} {default.column}",
        justification=justification,
        evidence=evidence,
        confidence=1.0,
        column=default.column,
        action=default.action,
        imputation=default.imputation,
        encoding=default.encoding,
        scaling=default.scaling,
        rule_derived_action=default.action,
        override_reason="",
        is_override=False,
    )


def _final_override_justification(plan: ColumnPlan) -> str:
    """Describe the transforms that will actually be applied after an override."""
    return (
        f"Final preprocessing plan for {plan.column!r}: action={plan.action}, "
        f"imputation={plan.imputation}, encoding={plan.encoding}, "
        f"scaling={plan.scaling}."
    )


def _apply_llm_payloads(
    plans: list[ColumnPlan],
    payloads: list[dict[str, Any]],
    raw_profile: dict,
) -> list[str]:
    logs: list[str] = []
    payload_by_column = {payload["column"]: payload for payload in payloads}
    overrides_used = 0

    for plan in plans:
        payload = payload_by_column.get(plan.column)
        if payload is None:
            continue

        plan.justification = payload["justification"]
        plan.evidence = [
            EvidenceRef(metric_key=ref["metric_key"], claimed_value=ref.get("claimed_value"))
            for ref in payload["evidence"]
        ]
        plan.confidence = float(payload["confidence"])

        proposed = {
            "imputation": payload["imputation"],
            "encoding": payload["encoding"],
            "scaling": payload["scaling"],
        }
        current = {
            "imputation": plan.imputation,
            "encoding": plan.encoding,
            "scaling": plan.scaling,
        }
        transform_changed = proposed != current
        if not transform_changed:
            if payload["override_reason"]:
                plan.justification = payload["override_reason"]
            continue

        if plan.action == "drop":
            logs.append(f"DROP_OVERRIDE_REJECTED:{plan.column}")
            continue
        if overrides_used >= OVERRIDE_CAP:
            logs.append(f"OVERRIDE_CAP_REACHED:{plan.column}")
            continue

        plan.imputation = proposed["imputation"]
        plan.encoding = proposed["encoding"]
        plan.scaling = proposed["scaling"]
        plan.action = _action_from_transforms(plan.imputation, plan.encoding, plan.scaling)
        plan.override_reason = payload["override_reason"]
        plan.is_override = True
        plan.justification = _final_override_justification(plan)
        overrides_used += 1

    attach_evidence_verdicts_to_claims(plans, raw_profile)
    return logs


def override_rate(plans: list[ColumnPlan]) -> float | None:
    if not plans:
        return None
    return sum(plan.is_override for plan in plans) / len(plans)


def update_metrics_with_override_rate(run_dir: str | Path, plans: list[ColumnPlan]) -> dict:
    metrics_path = Path(run_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics["override_rate"] = override_rate(plans)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def run_preprocessing_agent(
    raw_profile: dict,
    df: pd.DataFrame,
    target_column: str,
    *,
    columns: list[str] | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    client: Anthropic | None = None,
) -> tuple[list[ColumnPlan], Pipeline]:
    """Return validated column plans and an unfitted preprocessing Pipeline."""
    if target_column not in df.columns:
        raise KeyError(f"Target column {target_column!r} is not present in df")

    selected_columns = columns or list(df.columns)
    defaults = derive_column_plan_defaults(raw_profile, target_column, selected_columns)
    plans = [
        _make_plan_from_default(default, raw_profile, index)
        for index, default in enumerate(defaults, start=1)
    ]

    anthropic_client = client or Anthropic()
    model_name = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    batches = [
        defaults[start:start + PREPROCESSING_BATCH_SIZE]
        for start in range(0, len(defaults), PREPROCESSING_BATCH_SIZE)
    ]
    payloads: list[dict[str, Any]] = []
    for batch_index, batch_defaults in enumerate(batches, start=1):
        profile_subset = {
            key: raw_profile[key]
            for default in batch_defaults
            for key in default.evidence_keys
            if key in raw_profile
        }
        message = anthropic_client.messages.create(
            model=model_name,
            max_tokens=5000,
            temperature=temperature,
            system=PREPROCESSING_SYSTEM_PROMPT,
            tools=[PREPROCESSING_TOOL],
            tool_choice={"type": "tool", "name": "record_preprocessing_plans"},
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "rule_defaults": [default.__dict__ for default in batch_defaults],
                            "profile_subset": profile_subset,
                            "override_cap": OVERRIDE_CAP,
                            "batch_index": batch_index,
                            "batch_count": len(batches),
                            "override_budget_scope": "global across all batches",
                        },
                        default=_json_default,
                        sort_keys=True,
                    ),
                }
            ],
        )
        batch_payloads = _extract_tool_input(message)["plans"]
        expected_columns = [default.column for default in batch_defaults]
        returned_columns = [payload["column"] for payload in batch_payloads]
        if returned_columns != expected_columns:
            raise ValueError(
                f"Preprocessing batch {batch_index}/{len(batches)} returned columns "
                f"{returned_columns}; expected {expected_columns}"
            )
        if any(not str(payload["justification"]).strip() for payload in batch_payloads):
            raise ValueError(
                f"Preprocessing batch {batch_index}/{len(batches)} returned an empty justification"
            )
        if any(float(payload["confidence"]) <= 0.0 for payload in batch_payloads):
            raise ValueError(
                f"Preprocessing batch {batch_index}/{len(batches)} returned non-positive confidence"
            )
        payloads.extend(batch_payloads)

    logs = _apply_llm_payloads(plans, payloads, raw_profile)

    # Construction only: fitting is reserved for the evaluation-stage CV loop.
    from preprocessing_pipeline import build_preprocessing_pipeline

    pipeline = build_preprocessing_pipeline(plans, raw_profile)
    pipeline.preprocessing_logs = logs
    return plans, pipeline
