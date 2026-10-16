"""Tests for local lakehouse gold verification safeguards."""

import pandas as pd
import pytest

from scripts.verify_lakehouse_gold import checked_merge


def test_checked_merge_rejects_pandas_suffixes() -> None:
    left = pd.DataFrame({"run_id": ["run-1"], "metric": [1.0]})
    right = pd.DataFrame({"run_id": ["run-1"], "metric": [2.0]})

    with pytest.raises(ValueError, match="pandas_suffixes=.*metric_x.*metric_y"):
        checked_merge(left, right, context="test join", on="run_id")


def test_checked_merge_rejects_duplicate_columns() -> None:
    left = pd.DataFrame(
        [["run-1", 1.0, 2.0]],
        columns=["run_id", "metric", "metric"],
    )
    right = pd.DataFrame({"run_id": ["run-1"]})

    with pytest.raises(ValueError, match=r"duplicates=\['metric'\]"):
        checked_merge(left, right, context="test join", on="run_id")
