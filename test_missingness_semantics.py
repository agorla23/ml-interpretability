"""Regression tests for target-specific and clustered missingness semantics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from preprocessing_agent import derive_column_plan_defaults
from profile_dataset import (
    MISSINGNESS_INFORMATIVE_CORR_THRESHOLD,
    MISSINGNESS_LEAKAGE_CORR_THRESHOLD,
    profile_dataset,
)


TITANIC_DATASET = Path("benchmarks/titanic.csv")
MINI_DATASET = Path("/Users/akhilgorla/Downloads/mini_dataset.csv")


def test_titanic_missingness_uses_target_association_and_drops_boat() -> None:
    frame = pd.read_csv(TITANIC_DATASET)
    profile = profile_dataset(frame, "survived", "binary")

    assert MISSINGNESS_INFORMATIVE_CORR_THRESHOLD == 0.3
    assert MISSINGNESS_LEAKAGE_CORR_THRESHOLD == 0.7
    assert profile["col:boat:missingness_target_corr"] == pytest.approx(-0.9481900695821412)
    assert profile["col:boat:missingness_cluster_corr_max"] == pytest.approx(0.315893712357252)
    assert profile["col:boat:missingness_informative"] is True
    assert profile["col:boat:missingness_leakage_suspect"] is True
    assert not any(key.endswith(":missingness_corr_max") for key in profile)

    defaults = {
        default.column: default
        for default in derive_column_plan_defaults(profile, "survived")
    }
    boat = defaults["boat"]
    body = defaults["body"]
    assert boat.action == "drop"
    assert boat.drop_reason == "missingness_leakage_suspect"
    assert "col:boat:missingness_leakage_suspect" in boat.evidence_keys
    assert "col:boat:missingness_target_corr" in boat.evidence_keys
    assert body.action == "drop"
    assert body.drop_reason == "has_high_missing"


def test_missingness_drop_priority_is_leakage_then_missingness_then_constant() -> None:
    frame = pd.read_csv(TITANIC_DATASET)
    profile = profile_dataset(frame, "survived", "binary")

    profile["col:boat:is_constant"] = True
    defaults = {
        default.column: default
        for default in derive_column_plan_defaults(profile, "survived")
    }
    assert defaults["boat"].drop_reason == "missingness_leakage_suspect"

    profile["col:boat:leakage_suspect"] = True
    defaults = {
        default.column: default
        for default in derive_column_plan_defaults(profile, "survived")
    }
    assert defaults["boat"].drop_reason == "leakage_suspect"


def test_mini_age_old_cluster_signal_is_not_target_informative() -> None:
    frame = pd.read_csv(MINI_DATASET)
    profile = profile_dataset(frame, "target", "binary")

    for column in frame.columns:
        assert profile[f"col:{column}:missingness_leakage_suspect"] is False

    assert profile["col:age:missingness_cluster_corr_max"] == pytest.approx(1.0)
    assert profile["col:age:missingness_target_corr"] == pytest.approx(0.140028008402801)
    assert profile["col:age:missingness_informative"] is False
    assert profile["col:age_duplicate:missingness_informative"] is False


def test_boolean_indicator_is_profiled_as_categorical_not_numeric() -> None:
    frame = pd.DataFrame(
        {
            "DAYS_EMPLOYED": [-1000.0, -500.0, None, -200.0],
            "DAYS_EMPLOYED_ANOM": pd.Series([False, False, True, False], dtype=bool),
            "TARGET": [0, 1, 0, 1],
        }
    )

    profile = profile_dataset(frame, "TARGET", "binary")

    assert profile["col:DAYS_EMPLOYED_ANOM:dtype"] == "bool"
    assert profile["col:DAYS_EMPLOYED_ANOM:iqr_outlier_pct"] is None
    assert profile["col:DAYS_EMPLOYED_ANOM:vif"] is None
    assert profile["col:DAYS_EMPLOYED_ANOM:top_category_pct"] == 75.0
