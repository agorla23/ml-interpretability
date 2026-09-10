"""Fixed Milestone 9 benchmark acquisition, caching, and profile preflight."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import urllib.request
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from profile_dataset import profile_dataset
from pipeline_supervisor import compute_call_cap
from preprocessing_agent import preprocessing_batch_count


DEFAULT_BENCHMARK_DIR = Path("benchmarks")
HOME_CREDIT_ROWS = 50_000
HOME_CREDIT_RANDOM_STATE = 42


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    display_name: str
    task_type: str
    target_column: str
    cache_filename: str
    source: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "home_credit": BenchmarkSpec(
        "home_credit",
        "Home Credit Default Risk",
        "binary",
        "TARGET",
        "home_credit_50k.csv",
        "Kaggle Home Credit application_train.csv; deterministic 50k-row sample",
    ),
    "adult_income": BenchmarkSpec(
        "adult_income",
        "UCI Adult Income",
        "binary",
        "income",
        "adult_income.csv",
        "https://archive.ics.uci.edu/static/public/2/adult.zip",
    ),
    "ames_housing": BenchmarkSpec(
        "ames_housing",
        "Ames Housing",
        "regression",
        "price",
        "ames_housing.csv",
        "https://www.openintro.org/data/csv/ames.csv",
    ),
    "wine_quality": BenchmarkSpec(
        "wine_quality",
        "UCI Wine Quality (red wine)",
        "multiclass",
        "quality",
        "wine_quality_red.csv",
        "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    ),
    "titanic": BenchmarkSpec(
        "titanic",
        "Titanic",
        "binary",
        "survived",
        "titanic.csv",
        "https://www.openml.org/data/get_csv/16826755/phpMYEkMl",
    ),
}


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ml-interpretability-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_zip_csv(zip_path: Path, member_suffix: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(member_suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one {member_suffix!r} member in {zip_path}, found {matches}")
        with archive.open(matches[0]) as handle:
            return pd.read_csv(handle, **kwargs)


def _home_credit_source(explicit_source: str | Path | None) -> Path:
    candidates = [
        explicit_source,
        os.environ.get("HOME_CREDIT_APPLICATION_TRAIN"),
        Path.home() / "Downloads" / "application_train.csv.zip",
        Path.home() / "Downloads" / "application_train.csv",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "Home Credit requires application_train.csv or application_train.csv.zip. "
        "Pass --home-credit-source or set HOME_CREDIT_APPLICATION_TRAIN."
    )


def _build_home_credit(source: Path, source_dir: Path) -> tuple[pd.DataFrame, Path]:
    source_copy = source_dir / source.name
    if source.resolve() != source_copy.resolve() and not source_copy.exists():
        shutil.copy2(source, source_copy)
    stable_source = source_copy if source_copy.exists() else source
    if zipfile.is_zipfile(stable_source):
        frame = _read_zip_csv(stable_source, "application_train.csv", low_memory=False)
    else:
        frame = pd.read_csv(stable_source, low_memory=False)
    if len(frame) < HOME_CREDIT_ROWS:
        raise ValueError(f"Home Credit source has {len(frame)} rows; {HOME_CREDIT_ROWS} are required")
    if len(frame) > HOME_CREDIT_ROWS:
        frame = frame.sample(n=HOME_CREDIT_ROWS, random_state=HOME_CREDIT_RANDOM_STATE).sort_index()
    return frame.reset_index(drop=True), stable_source


def _build_adult(source_dir: Path) -> tuple[pd.DataFrame, Path]:
    source = source_dir / "adult.zip"
    if not source.exists():
        _download(BENCHMARKS["adult_income"].source, source)
    train = _read_zip_csv(
        source,
        "adult.data",
        names=ADULT_COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )
    test = _read_zip_csv(
        source,
        "adult.test",
        names=ADULT_COLUMNS,
        na_values="?",
        skipinitialspace=True,
        skiprows=1,
    )
    frame = pd.concat([train, test], ignore_index=True)
    for column in frame.select_dtypes(include="object"):
        frame[column] = frame[column].str.strip()
    frame["income"] = frame["income"].str.removesuffix(".")
    return frame, source


def _build_ames(source_dir: Path) -> tuple[pd.DataFrame, Path]:
    source = source_dir / "ames.csv"
    if not source.exists():
        _download(BENCHMARKS["ames_housing"].source, source)
    frame = pd.read_csv(source, low_memory=False)
    if "price" not in frame.columns:
        raise ValueError(f"Ames source is missing price; columns begin {list(frame.columns[:8])}")
    return frame, source


def _build_wine(source_dir: Path, explicit_source: str | Path | None) -> tuple[pd.DataFrame, Path]:
    local_candidates = [explicit_source, Path.home() / "Downloads" / "winequality-red.csv"]
    for candidate in local_candidates:
        if candidate and Path(candidate).exists():
            return pd.read_csv(candidate, sep=None, engine="python"), Path(candidate)
    source = source_dir / "wine_quality.zip"
    if not source.exists():
        _download(BENCHMARKS["wine_quality"].source, source)
    return _read_zip_csv(source, "winequality-red.csv", sep=";"), source


def _build_titanic(source_dir: Path) -> tuple[pd.DataFrame, Path]:
    source = source_dir / "titanic.csv"
    if not source.exists():
        _download(BENCHMARKS["titanic"].source, source)
    frame = pd.read_csv(source, na_values="?", low_memory=False)
    for column in ("age", "fare", "body"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, source


def _write_cache(frame: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)


def prepare_benchmark(
    name: str,
    *,
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    home_credit_source: str | Path | None = None,
    wine_source: str | Path | None = None,
    refresh: bool = False,
) -> tuple[BenchmarkSpec, Path]:
    """Download/normalize one fixed benchmark once and return its cached CSV."""
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark {name!r}; choose from {list(BENCHMARKS)}")
    spec = BENCHMARKS[name]
    root = Path(benchmark_dir)
    cache_path = root / spec.cache_filename
    source_dir = root / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return spec, cache_path

    builders: dict[str, Callable[[], tuple[pd.DataFrame, Path]]] = {
        "home_credit": lambda: _build_home_credit(_home_credit_source(home_credit_source), source_dir),
        "adult_income": lambda: _build_adult(source_dir),
        "ames_housing": lambda: _build_ames(source_dir),
        "wine_quality": lambda: _build_wine(source_dir, wine_source),
        "titanic": lambda: _build_titanic(source_dir),
    }
    frame, source_path = builders[name]()
    if spec.target_column not in frame.columns:
        raise ValueError(f"{spec.display_name} is missing target {spec.target_column!r}")
    _write_cache(frame, cache_path)
    metadata = {
        "benchmark": asdict(spec),
        "cache_path": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "n_rows": len(frame),
        "n_cols": len(frame.columns),
    }
    (root / f"{name}.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return spec, cache_path


def load_benchmark(
    name: str,
    *,
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    **prepare_kwargs,
) -> tuple[BenchmarkSpec, pd.DataFrame, Path]:
    spec, path = prepare_benchmark(name, benchmark_dir=benchmark_dir, **prepare_kwargs)
    frame = pd.read_csv(path, low_memory=False)
    return spec, frame, path


def _dtype_issues(frame: pd.DataFrame, spec: BenchmarkSpec) -> list[str]:
    issues: list[str] = []
    if frame.columns.duplicated().any():
        issues.append("duplicate column names")
    if frame[spec.target_column].isna().any():
        issues.append(f"target has {int(frame[spec.target_column].isna().sum())} missing values")
    if spec.task_type == "regression" and not pd.api.types.is_numeric_dtype(frame[spec.target_column]):
        issues.append("regression target is non-numeric")
    for column in frame.select_dtypes(include="object"):
        clean = frame[column].dropna()
        observed_types = {type(value).__name__ for value in clean.head(10_000)}
        if len(observed_types) > 1:
            issues.append(f"{column}: mixed Python value types {sorted(observed_types)}")
        if not clean.empty:
            numeric_ratio = pd.to_numeric(clean.head(10_000), errors="coerce").notna().mean()
            if numeric_ratio > 0.98:
                issues.append(f"{column}: numeric-looking object dtype ({numeric_ratio:.1%})")
    return issues


def preflight_benchmarks(
    *,
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    home_credit_source: str | Path | None = None,
    wine_source: str | Path | None = None,
    refresh: bool = False,
) -> dict:
    """Cache, load, and fully profile all five fixed benchmarks without LLM calls."""
    root = Path(benchmark_dir)
    profiles_dir = root / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for name in BENCHMARKS:
        spec, frame, cache_path = load_benchmark(
            name,
            benchmark_dir=root,
            home_credit_source=home_credit_source,
            wine_source=wine_source,
            refresh=refresh,
        )
        profile_path = profiles_dir / f"{name}.json"
        profile_cache_hit = profile_path.exists() and not refresh
        if profile_cache_hit:
            raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            caught = []
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                raw_profile = profile_dataset(frame, spec.target_column, spec.task_type)
            profile_path.write_text(
                json.dumps(raw_profile, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        n_features = len(frame.columns) - 1
        preprocess_batches = preprocessing_batch_count(n_features)
        report[name] = {
            "display_name": spec.display_name,
            "task_type": spec.task_type,
            "target_column": spec.target_column,
            "n_rows": len(frame),
            "n_cols": len(frame.columns),
            "dtype_counts": {str(dtype): int(count) for dtype, count in frame.dtypes.value_counts().items()},
            "dtype_issues": _dtype_issues(frame, spec),
            "profile_warnings": sorted({str(item.message) for item in caught}),
            "cache_path": str(cache_path),
            "profile_path": str(profile_path),
            "profile_cache_hit": profile_cache_hit,
            "estimated_llm_calls_without_retries": {
                "minimum": n_features + 13 + preprocess_batches,
                "maximum": n_features + 20 + preprocess_batches,
            },
            "call_cap_used": compute_call_cap(len(frame.columns)),
        }
        print(
            f"{name}: rows={len(frame)} cols={len(frame.columns)} "
            f"target={spec.target_column} task={spec.task_type} "
            f"dtype_issues={report[name]['dtype_issues']}"
        )
    report_path = root / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache and profile the fixed Milestone 9 benchmarks.")
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--home-credit-source", default=None)
    parser.add_argument("--wine-source", default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    report = preflight_benchmarks(
        benchmark_dir=args.benchmark_dir,
        home_credit_source=args.home_credit_source,
        wine_source=args.wine_source,
        refresh=args.refresh,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
