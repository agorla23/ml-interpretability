"""Deterministic CV evaluation and structured metric-justification claims."""

from __future__ import annotations

import json
import os
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from anthropic import Anthropic
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score, recall_score, precision_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from evidence_integration import attach_evidence_verdicts_to_claims
from evidence_validator import Claim, EvidenceRef, EvidenceVerdict
from model_selection_agent import ModelChoice


@dataclass
class MetricChoice(Claim):
    primary_metric: str
    secondary_metrics: list[str]
    evidence_verdicts: list[EvidenceVerdict] | None = None


@dataclass
class EvaluationResult:
    model_id: str
    rank: int
    primary_metric: str
    primary_metric_mean: float
    primary_metric_std: float
    secondary_metrics: dict[str, dict[str, float]]
    train_score_mean: float
    val_score_mean: float
    train_val_gap: float
    overfitting_flag: bool
    usable_folds: int


METRIC_SYSTEM_PROMPT = """You justify a deterministic evaluation metric choice.
Do not choose a different metric. Use only the supplied raw_profile and
preprocessing context for reasoning, and cite at least one supplied profile key.
For imbalanced binary classification,
explain correctly that accuracy can be misleading because majority-class
predictions can look accurate while missing the minority class.

You may ONLY cite metric_keys that appear in the raw_profile dict you were
given. This includes facts you may already know from elsewhere in the pipeline
(e.g. the task type used to select cross-validation strategy) -- if a fact is
not a key in raw_profile, do not cite it as evidence, even if it is true.
Reference it in your reasoning without an EvidenceRef, or omit it."""


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
EvidenceRef. Do not cite raw_profile[\"n_cols\"] for feature-count reasoning."""

METRIC_TOOL = {
    "name": "record_metric_justification", "description": "Record a metric-choice claim.",
    "input_schema": {"type": "object", "additionalProperties": False,
        "required": ["decision", "justification", "evidence", "confidence"],
        "properties": {"decision": {"type": "string"}, "justification": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "evidence": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False,
                "required": ["metric_key", "claimed_value"], "properties": {"metric_key": {"type": "string"}, "claimed_value": {"type": ["number", "string", "boolean", "object", "null"]}}}}}}
}


def deterministic_metric_choice(raw_profile: dict, task_type: str) -> tuple[str, list[str]]:
    if task_type == "binary":
        if bool(raw_profile.get("target:is_imbalanced", False)):
            return "pr_auc", ["f1", "roc_auc", "precision", "recall"]
        return "roc_auc", ["f1", "accuracy"]
    if task_type in {"multiclass", "classification"}:
        return "macro_f1", ["balanced_accuracy", "per_class_f1"]
    if task_type == "regression":
        if float(raw_profile.get("target_skew", 0.0) or 0.0) > 1.0:
            return "mae", ["rmse", "r2"]
        return "rmse", ["mae", "r2"]
    raise ValueError(f"Unsupported task_type: {task_type!r}")


def _extract_tool_input(message: Any) -> dict[str, Any]:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "record_metric_justification":
            return block.input
    raise ValueError("Claude response did not include record_metric_justification")


def run_metric_choice_agent(
    raw_profile: dict,
    task_type: str,
    *,
    retained_columns: list[str],
    dropped_columns: list[str],
    model: str | None = None,
    temperature: float = 0.0,
    client: Anthropic | None = None,
) -> MetricChoice:
    primary, secondary = deterministic_metric_choice(raw_profile, task_type)
    relevant_keys = ["target:is_imbalanced", "target_class_balance", "target_skew", "target_dtype", "n_rows"]
    subset = {key: raw_profile[key] for key in relevant_keys if key in raw_profile}
    anthropic_client = client or Anthropic()
    tool = deepcopy(METRIC_TOOL)
    evidence_key_schema = tool["input_schema"]["properties"]["evidence"]["items"]["properties"]["metric_key"]
    evidence_key_schema["enum"] = list(subset)
    message = anthropic_client.messages.create(
        model=model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"), max_tokens=1200, temperature=temperature,
        system=f"{METRIC_SYSTEM_PROMPT}\n\n{_preprocessing_context(dropped_columns, retained_columns)}", tools=[tool], tool_choice={"type": "tool", "name": "record_metric_justification"},
        messages=[{"role": "user", "content": json.dumps({"task_type": task_type, "primary_metric": primary, "secondary_metrics": secondary, "dropped_columns": dropped_columns, "retained_columns": retained_columns, "retained_column_count": len(retained_columns), "allowed_evidence_keys": list(subset), "raw_profile": subset}, default=lambda v: v.item() if hasattr(v, "item") else str(v), sort_keys=True)}],
    )
    item = _extract_tool_input(message)
    invalid_evidence_keys = {
        ref["metric_key"] for ref in item["evidence"] if ref["metric_key"] not in subset
    }
    if invalid_evidence_keys:
        raise ValueError(f"Metric justification cited keys outside raw_profile: {sorted(invalid_evidence_keys)}")
    claim = MetricChoice("evaluate_001", "evaluation", item["decision"], item["justification"], [EvidenceRef(ref["metric_key"], ref.get("claimed_value")) for ref in item["evidence"]], float(item["confidence"]), primary, secondary)
    attach_evidence_verdicts_to_claims([claim], raw_profile)
    return claim


def _estimator(model_id: str, task_type: str):
    classification = task_type in {"binary", "multiclass", "classification"}
    if model_id == "logreg" and classification: return LogisticRegression(max_iter=1000, C=1.0)
    if model_id == "linreg" and not classification: return Ridge(alpha=1.0)
    if model_id == "rf": return RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1) if classification else RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1)
    if model_id == "gbt": return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1) if classification else XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1)
    if model_id == "knn": return KNeighborsClassifier(n_neighbors=15) if classification else KNeighborsRegressor(n_neighbors=15)
    raise ValueError(f"Model {model_id!r} is not valid for {task_type!r}")


def _metric_fn(name: str, task_type: str, encoded_labels: dict[str, int] | None = None):
    def score(estimator, X, y):
        y = np.asarray(y)
        if name in {"roc_auc", "pr_auc"}:
            if len(np.unique(y)) < 2: return np.nan
            probabilities = estimator.predict_proba(X)
            values = probabilities[:, 1]
            return roc_auc_score(y, values) if name == "roc_auc" else average_precision_score(y, values)
        prediction = estimator.predict(X)
        if name == "f1": return f1_score(y, prediction, zero_division=0)
        if name == "accuracy": return accuracy_score(y, prediction)
        if name == "precision": return precision_score(y, prediction, zero_division=0)
        if name == "recall": return recall_score(y, prediction, zero_division=0)
        if name == "macro_f1": return f1_score(y, prediction, average="macro", zero_division=0)
        if name == "balanced_accuracy": return balanced_accuracy_score(y, prediction)
        if name.startswith("per_class_f1:"):
            original_label = name.split(":", 1)[1]
            if encoded_labels is None or original_label not in encoded_labels:
                raise ValueError(f"No encoded class label is available for {original_label!r}")
            return f1_score(y, prediction, labels=[encoded_labels[original_label]], average="macro", zero_division=0)
        if name == "mae": return mean_absolute_error(y, prediction)
        if name == "rmse": return root_mean_squared_error(y, prediction)
        if name == "r2": return r2_score(y, prediction)
        raise ValueError(f"Unsupported metric {name}")
    return score


def _summary(values: np.ndarray) -> tuple[float, float, int]:
    usable = values[~np.isnan(values)]
    return (float(np.mean(usable)), float(np.std(usable)), len(usable)) if len(usable) else (float("nan"), float("nan"), 0)


def evaluate_selected_models(choices: list[ModelChoice], preprocessing_pipeline: Pipeline, df, target_column: str, task_type: str, primary_metric: str, secondary_metrics: list[str]) -> tuple[list[EvaluationResult], bool]:
    """Evaluate selected models; the only estimator fits occur inside cross_validate."""
    X, y = df.drop(columns=[target_column]), df[target_column]
    classification = task_type in {"binary", "multiclass", "classification"}
    encoded_labels: dict[str, int] | None = None
    original_labels: list[Any] = []
    if classification:
        # XGBoost requires labels numbered 0..n_classes-1. Label names carry no
        # predictive information, so fitting this mapping before CV is safe.
        label_encoder = LabelEncoder()
        original_labels = list(label_encoder.fit(y).classes_)
        encoded_labels = {
            str(label): index
            for index, label in enumerate(original_labels)
        }
        y = pd.Series(label_encoder.transform(y), index=y.index, name=y.name)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if classification else KFold(n_splits=5, shuffle=True, random_state=42)
    degraded = classification and int(y.value_counts().min()) < 5
    metric_names = [primary_metric] + secondary_metrics
    if task_type in {"multiclass", "classification"} and "per_class_f1" in metric_names:
        metric_names.remove("per_class_f1")
        metric_names.extend([f"per_class_f1:{label}" for label in original_labels])
    scoring = {name: _metric_fn(name, task_type, encoded_labels) for name in metric_names}
    results: list[EvaluationResult] = []
    for choice in sorted((item for item in choices if not item.is_rejection), key=lambda item: item.rank or 99):
        full = Pipeline([("preprocess", preprocessing_pipeline), ("model", _estimator(choice.model_id, task_type))])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scores = cross_validate(full, X, y, cv=cv, scoring=scoring, return_train_score=True, error_score="raise")
        degraded = degraded or any("least populated class" in str(w.message).lower() for w in caught)
        val_mean, val_std, usable = _summary(np.asarray(scores[f"test_{primary_metric}"], dtype=float))
        train_mean, _, _ = _summary(np.asarray(scores[f"train_{primary_metric}"], dtype=float))
        secondary = {name: {"mean": _summary(np.asarray(scores[f"test_{name}"], dtype=float))[0], "std": _summary(np.asarray(scores[f"test_{name}"], dtype=float))[1]} for name in metric_names if name != primary_metric}
        results.append(EvaluationResult(choice.model_id, choice.rank or 0, primary_metric, val_mean, val_std, secondary, train_mean, val_mean, train_mean - val_mean, bool(train_mean - val_mean > 0.10), usable))
    return results, degraded


def persist_evaluation_results(run_dir: str | Path, results: list[EvaluationResult], stratification_degraded: bool) -> dict:
    run_path = Path(run_dir)
    with (run_path / "evaluation_results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results: handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
    metrics_path = run_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = min(results, key=lambda r: r.primary_metric_mean) if results and results[0].primary_metric in {"mae", "rmse"} else max(results, key=lambda r: r.primary_metric_mean)
    metrics.update({"models_evaluated": len(results), "best_model_id": best.model_id, "best_primary_metric_value": best.primary_metric_mean, "stratification_degraded": stratification_degraded, "evaluation_logs": ["STRATIFICATION_DEGRADED"] if stratification_degraded else []})
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics
