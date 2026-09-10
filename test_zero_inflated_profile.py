from pathlib import Path

import pandas as pd

from preprocessing_agent import derive_column_plan_defaults
from profile_dataset import ZERO_INFLATED_ZERO_FRACTION_THRESHOLD, profile_dataset


ADULT_DATASET = Path("benchmarks/adult_income.csv")


def test_adult_zero_inflated_columns_are_explicit_and_use_robust_scaling() -> None:
    frame = pd.read_csv(ADULT_DATASET)
    profile = profile_dataset(frame, "income", "binary")
    plans = {
        plan.column: plan
        for plan in derive_column_plan_defaults(profile, "income")
    }

    assert ZERO_INFLATED_ZERO_FRACTION_THRESHOLD == 0.5
    assert profile["col:capital-gain:is_zero_inflated"] is True
    assert profile["col:capital-gain:iqr_outlier_pct"] is None
    assert profile["col:capital-gain:nonzero_pct"] == 8.26
    assert plans["capital-gain"].scaling == "robust"

    assert profile["col:capital-loss:is_zero_inflated"] is True
    assert profile["col:capital-loss:iqr_outlier_pct"] is None
    assert profile["col:capital-loss:nonzero_pct"] == 4.67
    assert plans["capital-loss"].scaling == "robust"

    assert profile["col:fnlwgt:is_zero_inflated"] is False
    assert profile["col:fnlwgt:iqr_outlier_pct"] == 2.97
    assert profile["col:fnlwgt:nonzero_pct"] == 100.0
    assert plans["fnlwgt"].scaling == "standard"


def test_zero_threshold_is_strictly_greater_than_half() -> None:
    frame = pd.DataFrame(
        {
            "exactly_half_zero": [0, 0, 1, 1],
            "target": [0, 1, 0, 1],
        }
    )
    profile = profile_dataset(frame, "target", "binary")

    assert profile["col:exactly_half_zero:is_zero_inflated"] is False
