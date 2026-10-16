"""Shared pytest behavior for optional local benchmark datasets."""

from pathlib import Path

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip marked tests when their gitignored benchmark data is unavailable."""
    marker = item.get_closest_marker("requires_benchmark_data")
    if marker is None:
        return

    root = Path(item.config.rootpath)
    missing = [path for path in marker.args if not (root / path).is_file()]
    if missing:
        pytest.skip(
            "benchmark data not available: "
            f"{', '.join(str(path) for path in missing)}; "
            "build it with benchmark_datasets.py"
        )
