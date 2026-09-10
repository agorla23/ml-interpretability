"""Regression coverage for multiclass targets whose labels do not start at zero."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import evaluation_agent
from evaluation_agent import evaluate_selected_models
from evidence_validator import EvidenceRef
from model_selection_agent import ModelChoice


class EncodedLabelProbeClassifier(ClassifierMixin, BaseEstimator):
    """Predict the first transformed feature while recording CV fit labels."""

    observed_fit_labels: list[tuple[int, ...]] = []

    def fit(self, X, y):
        type(self).observed_fit_labels.append(tuple(sorted(np.unique(y).tolist())))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0].astype(int)


def _choice() -> ModelChoice:
    return ModelChoice(
        claim_id="model_select_001",
        stage="model_selection",
        model_id="gbt",
        rank=1,
        is_rejection=False,
        decision="SELECT",
        justification="Regression test choice.",
        evidence=[EvidenceRef("n_rows", 1)],
        confidence=1.0,
    )


def _passthrough_pipeline() -> Pipeline:
    return Pipeline(
        [("preprocess", ColumnTransformer([("numeric", "passthrough", ["prediction_code"])]))]
    )


def _assert_identity_label_evaluation(
    monkeypatch,
    labels: np.ndarray,
    expected_labels: tuple[int, ...],
    checked_label: int,
) -> None:
    prediction_codes = labels.copy()
    prediction_codes[::4] = (prediction_codes[::4] + 1) % len(expected_labels)
    frame = pd.DataFrame({"prediction_code": prediction_codes, "target": labels})
    EncodedLabelProbeClassifier.observed_fit_labels = []
    monkeypatch.setattr(
        evaluation_agent,
        "_estimator",
        lambda model_id, task_type: EncodedLabelProbeClassifier(),
    )

    multiclass = len(expected_labels) > 2
    primary_metric = "macro_f1" if multiclass else "f1"
    secondary_metrics = ["balanced_accuracy", "per_class_f1"] if multiclass else []
    results, degraded = evaluate_selected_models(
        [_choice()],
        _passthrough_pipeline(),
        frame,
        "target",
        "multiclass" if multiclass else "binary",
        primary_metric,
        secondary_metrics,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    expected_f1 = np.mean(
        [
            f1_score(
                labels[validation_index],
                prediction_codes[validation_index],
                labels=[checked_label],
                average="macro",
                zero_division=0,
            )
            for _, validation_index in cv.split(frame[["prediction_code"]], labels)
        ]
    )
    reported_f1 = (
        results[0].secondary_metrics[f"per_class_f1:{checked_label}"]["mean"]
        if multiclass
        else results[0].primary_metric_mean
    )

    assert degraded is False
    assert EncodedLabelProbeClassifier.observed_fit_labels == [expected_labels] * 5
    assert np.isclose(reported_f1, expected_f1)
    print(
        {
            "source_labels": labels.tolist(),
            "fit_labels": EncodedLabelProbeClassifier.observed_fit_labels,
            "reported_metric": f"per_class_f1:{checked_label}",
            "reported_f1": reported_f1,
            "independent_f1": float(expected_f1),
        }
    )


def test_wine_quality_labels_are_encoded_for_xgboost_and_reported_as_original_values() -> None:
    frame = pd.read_csv("benchmarks/wine_quality_red.csv")
    target = "quality"
    features = [column for column in frame.columns if column != target]
    preprocessing = Pipeline(
        [("preprocess", ColumnTransformer([("numeric", "passthrough", features)]))]
    )
    results, degraded = evaluate_selected_models(
        [_choice()],
        preprocessing,
        frame,
        target,
        "multiclass",
        "macro_f1",
        ["balanced_accuracy", "per_class_f1"],
    )

    assert degraded is False
    assert len(results) == 1
    assert results[0].primary_metric == "macro_f1"
    assert results[0].usable_folds == 5
    assert set(results[0].secondary_metrics) == {
        "balanced_accuracy",
        "per_class_f1:3",
        "per_class_f1:4",
        "per_class_f1:5",
        "per_class_f1:6",
        "per_class_f1:7",
        "per_class_f1:8",
    }


def test_contiguous_multiclass_labels_use_an_identity_mapping(monkeypatch) -> None:
    # Deliberately encounter class 2 first: LabelEncoder must still sort to 0, 1, 2.
    labels = np.tile(np.array([2, 0, 1]), 20)
    _assert_identity_label_evaluation(monkeypatch, labels, (0, 1, 2), checked_label=1)


def test_contiguous_binary_labels_use_an_identity_mapping(monkeypatch) -> None:
    # Deliberately encounter class 1 first to cover the real Adult/Home Credit shape.
    labels = np.tile(np.array([1, 0]), 30)
    _assert_identity_label_evaluation(monkeypatch, labels, (0, 1), checked_label=1)
