"""Leakage-safe preprocessing Pipeline construction.

Build an unfitted sklearn Pipeline / ColumnTransformer that can be passed to
``cross_val_score`` or ``cross_validate``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from preprocessing_agent import ColumnPlan


@dataclass(frozen=True)
class PipelineColumnSpec:
    imputation: str
    encoding: str
    scaling: str
    add_indicator: bool
    stringify_numeric_categories: bool


class NumericCategoricalStringifier(BaseEstimator, TransformerMixin):
    """Convert numeric category values to strings while preserving missing values."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return frame.apply(
            lambda series: series.map(
                lambda value: str(value) if pd.notna(value) else np.nan
            )
        )


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """K-fold target encoder that avoids target leakage during fit_transform.

    ``fit_transform`` is called by sklearn only on the training subset for the
    current outer CV fold. Inside that training subset, this encoder creates
    out-of-fold encodings: each row is encoded using mappings fit on the other
    inner folds only. ``transform`` then applies mappings learned from the full
    outer-training subset to the held-out fold.
    """

    def __init__(self, n_splits: int = 5, smoothing: float = 10.0, random_state: int = 0):
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

    def fit(self, X, y):
        # FIT BOUNDARY: fit is called by the sklearn Pipeline on the current
        # training fold only; the outer validation fold is not visible here.
        frame = self._to_frame(X)
        target = pd.Series(y).reset_index(drop=True)
        self.global_mean_ = float(target.mean())
        self.category_maps_ = [
            self._fit_column_mapping(
                frame.iloc[:, index],
                target,
                self.global_mean_,
            )
            for index in range(frame.shape[1])
        ]
        return self

    def fit_transform(self, X, y=None, **fit_params):
        # FIT BOUNDARY: fit_transform receives only the current outer-training
        # fold. Inner K-fold mappings below are fit on inner-training splits and
        # applied to their matching inner-validation split.
        if y is None:
            raise ValueError("KFoldTargetEncoder requires y during fit_transform")

        frame = self._to_frame(X)
        target = pd.Series(y).reset_index(drop=True)
        encoded = pd.DataFrame(index=frame.index)
        n_splits = min(self.n_splits, len(frame))
        if n_splits < 2:
            raise ValueError(
                "KFoldTargetEncoder requires at least 2 rows for out-of-fold encoding"
            )

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for column_index in range(frame.shape[1]):
            encoded_column = pd.Series(np.nan, index=frame.index, dtype=float)
            for inner_train_idx, inner_valid_idx in kfold.split(frame):
                inner_prior = float(target.iloc[inner_train_idx].mean())
                mapping = self._fit_column_mapping(
                    frame.iloc[inner_train_idx, column_index],
                    target.iloc[inner_train_idx],
                    inner_prior,
                )
                encoded_column.iloc[inner_valid_idx] = (
                    frame.iloc[inner_valid_idx, column_index]
                    .map(mapping)
                    .fillna(inner_prior)
                )
            encoded[column_index] = encoded_column.fillna(float(target.mean()))

        self.fit(frame, target)
        return encoded.to_numpy()

    def transform(self, X):
        frame = self._to_frame(X)
        encoded = pd.DataFrame(index=frame.index)
        for column_index, mapping in enumerate(self.category_maps_):
            encoded[column_index] = (
                frame.iloc[:, column_index]
                .map(mapping)
                .fillna(self.global_mean_)
                .astype(float)
            )
        return encoded.to_numpy()

    def _fit_column_mapping(
        self,
        values: pd.Series,
        target: pd.Series,
        prior: float,
    ) -> pd.Series:
        stats = (
            pd.DataFrame({"value": values.values, "target": target.values})
            .groupby("value")["target"]
            .agg(["mean", "count"])
        )
        weight = stats["count"] / (stats["count"] + self.smoothing)
        return prior * (1.0 - weight) + stats["mean"] * weight

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        return pd.DataFrame(X)


def build_preprocessing_pipeline(column_plans: list[ColumnPlan], raw_profile: dict) -> Pipeline:
    """Build an unfitted leakage-safe preprocessing pipeline.

    Pass the returned Pipeline as the estimator preprocessing step inside
    ``cross_val_score`` / ``cross_validate``. sklearn clones this object and
    calls ``fit`` only on each training fold.
    """
    column_transformer = ColumnTransformer(
        transformers=_build_transformers(column_plans, raw_profile),
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline(
        steps=[
            # FIT BOUNDARY: ColumnTransformer.fit is invoked only when the
            # enclosing sklearn estimator Pipeline is fit on a training fold.
            ("preprocess", column_transformer),
        ]
    )


def _build_transformers(column_plans: list[ColumnPlan], raw_profile: dict) -> list[tuple[str, Pipeline, list[str]]]:
    grouped: dict[PipelineColumnSpec, list[str]] = defaultdict(list)
    for plan in column_plans:
        if plan.action == "drop":
            continue
        add_indicator = bool(raw_profile.get(f"col:{plan.column}:missingness_informative", False))
        spec = PipelineColumnSpec(
            imputation=plan.imputation,
            encoding=plan.encoding,
            scaling=plan.scaling,
            add_indicator=add_indicator,
            stringify_numeric_categories=(
                plan.action == "impute_and_encode"
                and plan.encoding == "onehot"
                and bool(raw_profile.get(f"col:{plan.column}:is_numeric_but_categorical", False))
            ),
        )
        grouped[spec].append(plan.column)

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    for index, (spec, columns) in enumerate(grouped.items(), start=1):
        transformers.append((f"prep_{index}", _pipeline_for_spec(spec), columns))
    return transformers


def _pipeline_for_spec(spec: PipelineColumnSpec) -> Pipeline:
    steps: list[tuple[str, Any]] = []

    if spec.stringify_numeric_categories:
        steps.append(("stringify_numeric_categories", NumericCategoricalStringifier()))

    if spec.imputation != "none":
        steps.append(("imputer", _imputer_for_spec(spec)))

    if spec.encoding != "none":
        steps.append(("encoder", _encoder_for_spec(spec)))

    if spec.scaling != "none":
        steps.append(("scaler", _scaler_for_spec(spec)))

    if not steps:
        steps.append(("passthrough", "passthrough"))

    return Pipeline(steps=steps)


def _imputer_for_spec(spec: PipelineColumnSpec) -> SimpleImputer:
    if spec.imputation == "constant":
        fill_value = -999 if spec.encoding == "none" else "__MISSING__"
        return SimpleImputer(strategy="constant", fill_value=fill_value, add_indicator=spec.add_indicator)
    if spec.imputation == "mean":
        return SimpleImputer(strategy="mean", add_indicator=spec.add_indicator)
    if spec.imputation == "median":
        return SimpleImputer(strategy="median", add_indicator=spec.add_indicator)
    if spec.imputation == "mode":
        return SimpleImputer(strategy="most_frequent", add_indicator=spec.add_indicator)
    if spec.imputation == "knn":
        raise NotImplementedError("KNN imputation is not wired until 6B review approval")
    raise ValueError(f"Unsupported imputation strategy: {spec.imputation}")


def _encoder_for_spec(spec: PipelineColumnSpec):
    if spec.encoding == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if spec.encoding == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if spec.encoding == "target":
        return KFoldTargetEncoder(n_splits=5, smoothing=10)
    raise ValueError(f"Unsupported encoding strategy: {spec.encoding}")


def _scaler_for_spec(spec: PipelineColumnSpec):
    if spec.scaling == "standard":
        return StandardScaler()
    if spec.scaling == "minmax":
        return MinMaxScaler()
    if spec.scaling == "robust":
        return RobustScaler()
    raise ValueError(f"Unsupported scaling strategy: {spec.scaling}")
