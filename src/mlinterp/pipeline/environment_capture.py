"""Capture the reproducibility environment for newly created pipeline runs."""

from __future__ import annotations

import json
import platform
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def find_requirements_file(start: Path | None = None) -> Path:
    """Walk upward until the repository's requirements file is found."""
    origin = (start or Path(__file__)).resolve()
    for directory in (origin.parent, *origin.parents):
        candidate = directory / "requirements.txt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find requirements.txt above environment_capture.py")


def requirement_names(requirements_path: Path) -> list[str]:
    """Return distribution names from the repository's pinned requirements."""
    names = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = REQUIREMENT_NAME.match(line)
        if match is None:
            raise ValueError(f"Could not parse requirement line: {raw_line!r}")
        names.append(match.group(1))
    return names


def environment_payload(requirements_path: Path | None = None) -> dict:
    """Build a JSON-serializable snapshot of Python and pinned packages."""
    resolved_requirements = requirements_path or find_requirements_file()
    packages = {}
    for name in requirement_names(resolved_requirements):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def write_environment_snapshot(
    run_dir: str | Path, requirements_path: Path | None = None
) -> Path:
    """Write environment.json into a run directory and return its path."""
    destination = Path(run_dir) / "environment.json"
    destination.write_text(
        json.dumps(environment_payload(requirements_path), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return destination
