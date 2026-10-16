"""Tests for per-run environment provenance capture."""

import json
import platform
from importlib.metadata import version

from mlinterp.pipeline.environment_capture import find_requirements_file, requirement_names
from mlinterp.pipeline.pipeline_supervisor import PipelineState, PipelineSupervisor


def test_pipeline_initialization_writes_environment_snapshot(tmp_path):
    state = PipelineState(
        run_id="environment-test",
        dataset_path="unused.csv",
        target_column="target",
        task_type="binary",
    )

    PipelineSupervisor(state, runs_dir=tmp_path, client=object())

    environment_path = tmp_path / state.run_id / "environment.json"
    payload = json.loads(environment_path.read_text(encoding="utf-8"))
    expected_packages = {
        name: version(name) for name in requirement_names(find_requirements_file())
    }
    assert payload["python_version"] == platform.python_version()
    assert payload["platform"] == platform.platform()
    assert payload["packages"] == expected_packages
