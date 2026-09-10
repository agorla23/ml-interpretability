"""Dataset profiling utilities for tabular ML workflows."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Tunable thresholds: revisit after observing missingness-to-target behavior on
# the remaining benchmark datasets.
MISSINGNESS_INFORMATIVE_CORR_THRESHOLD = 0.3
MISSINGNESS_LEAKAGE_CORR_THRESHOLD = 0.7
ZERO_INFLATED_ZERO_FRACTION_THRESHOLD = 0.5


def _as_float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _is_numeric(series: pd.Series) -> bool:
    # Pandas considers bool numeric, but IQR/VIF statistics require quantities.
    return bool(
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    )


def _is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )


def _is_whole_number_numeric(series: pd.Series) -> bool:
    clean = series.dropna()
    if clean.empty:
        return False
    return bool((clean % 1 == 0).all())


def _is_sequential(series: pd.Series) -> bool:
    if not _is_numeric(series) or not _is_whole_number_numeric(series):
        return False

    unique_values = np.sort(series.dropna().unique())
    if len(unique_values) < 3:
        return False

    diffs = np.diff(unique_values)
    return bool((diffs > 0).all() and np.allclose(diffs, diffs[0]))


def _pearson_corr(left: pd.Series, right: pd.Series) -> float | None:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 2:
        return None
    if pair.iloc[:, 0].nunique(dropna=True) <= 1 or pair.iloc[:, 1].nunique(dropna=True) <= 1:
        return None
    return _as_float_or_none(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson"))


def _cramers_v(left: pd.Series, right: pd.Series) -> float | None:
    pair = pd.concat([left, right], axis=1).dropna()
    if pair.empty:
        return None
    table = pd.crosstab(pair.iloc[:, 0], pair.iloc[:, 1])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None

    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n <= 0:
        return None

    phi2 = chi2 / n
    rows, cols = table.shape

    # Bergsma/Wicher bias correction keeps Cramer's V stable on sparse tables.
    phi2_corr = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / (n - 1)) if n > 1 else 0.0
    rows_corr = rows - ((rows - 1) ** 2) / (n - 1) if n > 1 else rows
    cols_corr = cols - ((cols - 1) ** 2) / (n - 1) if n > 1 else cols
    denom = min(cols_corr - 1, rows_corr - 1)
    if denom <= 0:
        return None

    return _as_float_or_none(math.sqrt(phi2_corr / denom))


def _correlation_with_target(column: pd.Series, target: pd.Series, target_is_numeric: bool) -> float | None:
    if target_is_numeric and _is_numeric(column):
        return _pearson_corr(column, target)
    return _cramers_v(column, target)


def _max_abs_corr_other(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, float | None]:
    if len(numeric_columns) < 2:
        return {column: None for column in df.columns}

    corr = df[numeric_columns].corr(method="pearson").abs()
    values: dict[str, float | None] = {column: None for column in df.columns}
    for column in numeric_columns:
        others = corr.loc[column].drop(labels=[column]).dropna()
        values[column] = _as_float_or_none(others.max()) if not others.empty else None
    return values


def _vif_by_column(
    df: pd.DataFrame,
    numeric_columns: list[str],
    duplicate_of: dict[str, str | None],
    target_column: str,
) -> dict[str, float | None]:
    values: dict[str, float | None] = {column: None for column in df.columns}
    if len(numeric_columns) < 2:
        return values

    column_order = {column: index for index, column in enumerate(df.columns)}
    numeric_column_set = set(numeric_columns)

    def duplicate_representative(column: str) -> str:
        group = {column}
        if duplicate_of[column] is not None:
            group.add(duplicate_of[column])
        for other_column, matching_column in duplicate_of.items():
            if matching_column == column and other_column in numeric_column_set:
                group.add(other_column)
        return min(group, key=column_order.get)

    representative_columns = [
        column
        for column in numeric_columns
        if column != target_column and duplicate_representative(column) == column
    ]
    if len(representative_columns) < 2:
        return values

    numeric = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    usable_columns = [
        column
        for column in representative_columns
        if numeric[column].notna().any() and numeric[column].nunique(dropna=True) > 1
    ]
    if len(usable_columns) < 2:
        return values

    matrix = numeric[usable_columns].fillna(numeric[usable_columns].median())
    matrix = matrix.astype(float)
    matrix = (matrix - matrix.mean()) / matrix.std(ddof=0)
    for index, column in enumerate(usable_columns):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                values[column] = _as_float_or_none(variance_inflation_factor(matrix.to_numpy(), index))
        except (FloatingPointError, RuntimeWarning, ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            values[column] = None
    return values


def _missingness_cluster_corr_max(df: pd.DataFrame) -> dict[str, float | None]:
    """Return max correlation with other columns' missingness indicators."""
    values: dict[str, float | None] = {column: None for column in df.columns}
    if len(df.columns) < 2:
        return values

    missing = df.isna().astype(float)
    for column in df.columns:
        if missing[column].nunique(dropna=True) <= 1:
            continue
        best: float | None = None
        for other in df.columns:
            if other == column or missing[other].nunique(dropna=True) <= 1:
                continue
            corr = _pearson_corr(missing[column], missing[other])
            if corr is not None:
                best = max(best or 0.0, abs(corr))
        values[column] = best
    return values


def _numeric_ranges_nearly_identical(left: pd.Series, right: pd.Series) -> bool:
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if pair.empty:
        return False

    left_min = pair.iloc[:, 0].min()
    left_max = pair.iloc[:, 0].max()
    right_min = pair.iloc[:, 1].min()
    right_max = pair.iloc[:, 1].max()
    return bool(
        np.isclose(left_min, right_min, rtol=1e-3, atol=1e-8)
        and np.isclose(left_max, right_max, rtol=1e-3, atol=1e-8)
    )


def _duplicate_of_by_column(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, str | None]:
    duplicate_of: dict[str, str | None] = {column: None for column in df.columns}
    numeric_column_set = set(numeric_columns)

    for column in df.columns:
        series = df[column]
        for other_column in df.columns:
            if other_column == column:
                continue

            other_series = df[other_column]
            if series.equals(other_series):
                duplicate_of[column] = other_column
                break

            both_numeric = column in numeric_column_set and other_column in numeric_column_set
            if not both_numeric:
                continue

            corr = _pearson_corr(series, other_series)
            if (
                corr is not None
                and corr > 0.999
                and _numeric_ranges_nearly_identical(series, other_series)
            ):
                duplicate_of[column] = other_column
                break

    return duplicate_of


def profile_dataset(df: pd.DataFrame, target_column: str, task_type: str) -> dict:
    """Return a flat profile dictionary for a supervised tabular dataset.

    Threshold rationale:
    - ``is_high_cardinality`` flags object/category columns with more than 15
      distinct values, a practical warning that one-hot encoding may create a
      wide sparse design matrix or require target/frequency encoding.
    - ``is_id_like`` requires both ``nunique / n_rows > 0.95`` and ID-shaped
      values: integer-valued numeric data or non-numeric categorical strings.
      Uniqueness alone is insufficient because continuous float measurements
      also often have near-100% unique values; requiring integer-like numeric
      values separates identifiers from fractional measurements.
    - ``is_sequential`` marks numeric columns whose sorted unique whole-number
      values have a constant positive step, which is near-conclusive diagnostic
      evidence for row identifiers while leaving non-sequential unique integers
      available for more cautious downstream handling.
    - ``has_high_missing`` uses missingness above 40% to identify columns where
      deletion, strong imputation assumptions, or missingness indicators may
      materially change model behavior.
    - ``has_moderate_missing`` uses 5%-40% to mark columns with enough missing
      data to deserve imputation review while usually retaining usable signal.
    - ``is_multicollinear`` uses VIF > 10, a common regression diagnostic for
      severe linear dependence that can destabilize coefficients and importances.
    - ``is_skewed`` uses absolute skew > 1 to flag strongly asymmetric numeric
      distributions where transforms, robust scalers, or winsorization may help.
    - ``is_zero_inflated`` marks numeric columns when Q1 equals Q3, so the IQR
      outlier rule has collapsed, and strictly more than 50% of non-null values
      are exactly zero. For these columns ``iqr_outlier_pct`` is ``None`` rather
      than a misleading 0.0, while ``nonzero_pct`` records the real nonzero tail.
    - ``leakage_suspect`` uses absolute target association > 0.95 because such
      near-perfect relationships often indicate target leakage, duplicated target
      fields, or post-outcome features.
    - ``missingness_target_corr`` measures association between a column's
      missingness indicator and the target (Pearson for numeric targets and
      Cramer's V for categorical targets, matching ``corr_with_target``). The
      signed Pearson value is retained because its direction is informative.
    - ``missingness_cluster_corr_max`` measures whether this column's missingness
      coincides with OTHER columns' missingness (a data-collection-pattern
      signal), NOT whether missingness relates to the target.
    - ``missingness_informative`` uses absolute ``missingness_target_corr`` > 0.3
      to flag columns whose ABSENCE is itself predictive of the target, distinct
      from ``missingness_cluster_corr_max``, which only measures coincidence with
      other columns' missingness.
    - ``missingness_leakage_suspect`` uses absolute
      ``missingness_target_corr`` > 0.7 because extremely target-linked absence
      can reconstruct the outcome when encoded, often because the reason for
      missingness is downstream of the outcome. This threshold is deliberately
      tunable pending evidence from additional datasets.
    - ``is_low_variance`` uses coefficient of variation below 0.01, excluding
      already-constant columns, to catch numeric features that technically vary
      but have almost no spread relative to their scale.
    - ``is_numeric_but_categorical`` marks numeric columns with at most 10
      unique whole-number values, a common sign that the numbers are category
      codes that should be encoded rather than scaled as continuous quantities.
    - ``is_duplicate_of_other_column`` detects exact duplicates for any dtype,
      plus numeric columns with Pearson correlation above 0.999 and near-identical
      value ranges, so literally redundant columns can be dropped rather than
      treated as ordinary multicollinearity.
    - ``target:is_imbalanced`` marks binary targets with a minority class below
      20%, and multiclass targets with any class below 5%, because those regimes
      often need stratified validation, class weighting, or threshold tuning.

    The returned dictionary is intentionally flat: dataset metrics use their
    plain names, target flags use ``target:...``, and per-column metrics use
    ``col:{name}:...``. Non-applicable metrics are included with ``None``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if target_column not in df.columns:
        raise ValueError(f"target_column {target_column!r} is not in df")

    normalized_task_type = task_type.lower().strip()
    if normalized_task_type in {"binary", "multiclass"}:
        normalized_task_type = "classification"
    if normalized_task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification', 'binary', 'multiclass', or 'regression'")

    n_rows, n_cols = df.shape
    target = df[target_column]
    target_is_numeric = _is_numeric(target)
    numeric_columns = [column for column in df.columns if _is_numeric(df[column])]

    result: dict[str, Any] = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_duplicated_rows": int(df.duplicated().sum()),
        "target_dtype": str(target.dtype),
        "target_class_balance": None,
        "target_skew": None,
        "target:is_imbalanced": None,
    }

    if normalized_task_type == "classification":
        class_balance = target.value_counts(normalize=True, dropna=False).to_dict()
        result["target_class_balance"] = {key: round(float(value), 4) for key, value in class_balance.items()}
        non_missing_counts = target.dropna().value_counts(normalize=True)
        if non_missing_counts.empty:
            result["target:is_imbalanced"] = None
        elif len(non_missing_counts) == 2:
            # Strict <, so exactly 20.0% minority is NOT flagged as imbalanced.
            result["target:is_imbalanced"] = bool(non_missing_counts.min() < 0.20)
        else:
            # Strict <, so exactly 5.0% minority is NOT flagged as imbalanced.
            result["target:is_imbalanced"] = bool((non_missing_counts < 0.05).any())
    else:
        result["target_skew"] = _as_float_or_none(target.skew()) if target_is_numeric else None
        result["target:is_imbalanced"] = None

    max_corr_other = _max_abs_corr_other(df, numeric_columns)
    duplicate_of = _duplicate_of_by_column(df, numeric_columns)
    vif_values = _vif_by_column(df, numeric_columns, duplicate_of, target_column)
    missing_cluster_corr = _missingness_cluster_corr_max(df)

    for column_name in df.columns:
        column = df[column_name]
        prefix = f"col:{column_name}:"
        is_numeric = _is_numeric(column)
        is_categorical = _is_categorical(column)
        nunique = int(column.nunique(dropna=True))
        missing_pct = round(float(column.isna().mean() * 100.0), 2) if n_rows else None
        corr_with_target = (
            None if column_name == target_column else _correlation_with_target(column, target, target_is_numeric)
        )
        missingness_target_corr = (
            None
            if column_name == target_column
            else _correlation_with_target(column.isna().astype(float), target, target_is_numeric)
        )
        skew = _as_float_or_none(column.skew()) if is_numeric else None
        mean = _as_float_or_none(column.mean()) if is_numeric else None
        std = _as_float_or_none(column.std()) if is_numeric else None
        iqr_outlier_pct, is_zero_inflated, nonzero_pct = (
            _iqr_outlier_profile(column) if is_numeric else (None, False, None)
        )
        is_whole_number_numeric = is_numeric and _is_whole_number_numeric(column)
        high_uniqueness = bool(n_rows > 0 and nunique / n_rows > 0.95)
        id_shaped = bool(is_whole_number_numeric or (is_categorical and not is_numeric))

        result[f"{prefix}dtype"] = str(column.dtype)
        result[f"{prefix}nunique"] = nunique
        result[f"{prefix}missing_pct"] = missing_pct
        result[f"{prefix}mean"] = mean
        result[f"{prefix}std"] = std
        result[f"{prefix}min"] = _as_float_or_none(column.min()) if is_numeric else None
        result[f"{prefix}max"] = _as_float_or_none(column.max()) if is_numeric else None
        result[f"{prefix}median"] = _as_float_or_none(column.median()) if is_numeric else None
        result[f"{prefix}skew"] = skew
        result[f"{prefix}kurtosis"] = _as_float_or_none(column.kurtosis()) if is_numeric else None
        result[f"{prefix}iqr_outlier_pct"] = iqr_outlier_pct
        result[f"{prefix}nonzero_pct"] = nonzero_pct
        result[f"{prefix}top_category_pct"] = _top_category_pct(column) if is_categorical else None
        result[f"{prefix}corr_with_target"] = corr_with_target
        result[f"{prefix}max_abs_corr_other"] = max_corr_other[column_name]
        result[f"{prefix}vif"] = vif_values[column_name] if is_numeric else None
        result[f"{prefix}missingness_target_corr"] = missingness_target_corr
        result[f"{prefix}missingness_cluster_corr_max"] = missing_cluster_corr[column_name]
        result[f"{prefix}duplicate_of"] = duplicate_of[column_name]
        result[f"{prefix}is_sequential"] = _is_sequential(column)

        result[f"{prefix}is_high_cardinality"] = bool(nunique > 15 and is_categorical)
        result[f"{prefix}is_constant"] = bool(nunique <= 1)
        result[f"{prefix}is_id_like"] = bool(high_uniqueness and id_shaped)
        result[f"{prefix}has_high_missing"] = bool(missing_pct is not None and missing_pct > 40.0)
        result[f"{prefix}has_moderate_missing"] = bool(
            missing_pct is not None and 5.0 < missing_pct <= 40.0
        )
        result[f"{prefix}is_multicollinear"] = bool(
            result[f"{prefix}vif"] is not None and result[f"{prefix}vif"] > 10.0
        )
        result[f"{prefix}is_skewed"] = bool(skew is not None and abs(skew) > 1.0)
        result[f"{prefix}is_zero_inflated"] = is_zero_inflated
        result[f"{prefix}leakage_suspect"] = (
            None if column_name == target_column else bool(corr_with_target is not None and abs(corr_with_target) > 0.95)
        )
        result[f"{prefix}missingness_informative"] = bool(
            missingness_target_corr is not None
            and abs(missingness_target_corr) > MISSINGNESS_INFORMATIVE_CORR_THRESHOLD
        )
        result[f"{prefix}missingness_leakage_suspect"] = bool(
            missingness_target_corr is not None
            and abs(missingness_target_corr) > MISSINGNESS_LEAKAGE_CORR_THRESHOLD
        )
        result[f"{prefix}is_low_variance"] = bool(
            is_numeric
            and not result[f"{prefix}is_constant"]
            and n_rows > 0
            and nunique / n_rows < 0.5
            and mean is not None
            and std is not None
            and std / (abs(mean) + 1e-8) < 0.01
        )
        result[f"{prefix}is_numeric_but_categorical"] = bool(
            is_numeric
            and nunique <= 10
            and is_whole_number_numeric
        )
        result[f"{prefix}is_duplicate_of_other_column"] = bool(duplicate_of[column_name] is not None)

    return result


def _iqr_outlier_profile(series: pd.Series) -> tuple[float | None, bool, float | None]:
    clean = series.dropna()
    if clean.empty:
        return None, False, None

    zero_fraction = float(clean.eq(0).mean())
    nonzero_pct = round((1.0 - zero_fraction) * 100.0, 2)
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr):
        return None, False, nonzero_pct

    is_zero_inflated = bool(
        q1 == q3 and zero_fraction > ZERO_INFLATED_ZERO_FRACTION_THRESHOLD
    )
    if is_zero_inflated:
        return None, True, nonzero_pct
    if iqr == 0:
        return 0.0, False, nonzero_pct

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_pct = round(float(((clean < lower) | (clean > upper)).mean() * 100.0), 2)
    return outlier_pct, False, nonzero_pct


def _top_category_pct(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return round(float(clean.value_counts(normalize=True).iloc[0] * 100.0), 2)
