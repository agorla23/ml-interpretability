"""Deterministic synthetic-corruption injection for Arm C experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


SYNTHETIC_CORRUPTION_SEED = 20_260_914
SYNTH_CONSTANT = "SYNTH_CONSTANT"
SYNTH_LEAKED = "SYNTH_LEAKED"
SYNTH_HIGH_MISSING = "SYNTH_HIGH_MISSING"
SYNTHETIC_COLUMNS = (SYNTH_CONSTANT, SYNTH_LEAKED, SYNTH_HIGH_MISSING)
TARGET_LEAK_CORRELATION = 0.99
HIGH_MISSING_FRACTION = 0.60


@dataclass(frozen=True)
class SyntheticCorruptionReport:
    seed: int
    columns: dict[str, str]
    constant_nunique: int
    leaked_corr_with_target: float
    high_missing_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


def inject_synthetic_corruptions(
    frame: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> pd.DataFrame:
    """Return a copy with the three fixed Arm C defect columns.

    The leakage column uses random noise orthogonal to the centered binary
    target, then scales that noise to achieve the requested sample correlation.
    """
    if target_column not in frame:
        raise KeyError(f"Target column {target_column!r} is not present")
    if any(column in frame for column in SYNTHETIC_COLUMNS):
        raise ValueError("Input already contains a reserved Arm C synthetic column")

    target = frame[target_column]
    if target.isna().any():
        raise ValueError("Arm C synthetic leakage injection requires a complete target")
    numeric_target = pd.to_numeric(target, errors="coerce")
    if numeric_target.isna().any():
        numeric_target = pd.Series(pd.factorize(target, sort=True)[0], index=target.index, dtype=float)

    rng = np.random.default_rng(SYNTHETIC_CORRUPTION_SEED)
    centered_target = numeric_target.to_numpy(dtype=float) - float(numeric_target.mean())
    target_std = float(np.sqrt(np.mean(centered_target**2)))
    if target_std == 0.0:
        raise ValueError("Arm C binary leakage injection requires target variance")

    noise = rng.standard_normal(len(frame))
    noise -= noise.mean()
    noise -= centered_target * (float(np.dot(noise, centered_target)) / float(np.dot(centered_target, centered_target)))
    noise_std = float(np.sqrt(np.mean(noise**2)))
    if noise_std == 0.0:
        raise ValueError("Synthetic leakage noise unexpectedly has zero variance")
    noise *= target_std * np.sqrt((1.0 / TARGET_LEAK_CORRELATION**2) - 1.0) / noise_std

    corrupted = frame.copy(deep=True)
    corrupted[SYNTH_CONSTANT] = 1
    corrupted[SYNTH_LEAKED] = numeric_target.to_numpy(dtype=float) + noise
    high_missing = rng.normal(size=len(frame))
    missing_count = round(len(frame) * HIGH_MISSING_FRACTION)
    missing_rows = rng.choice(len(frame), size=missing_count, replace=False)
    high_missing[missing_rows] = np.nan
    corrupted[SYNTH_HIGH_MISSING] = high_missing
    return corrupted


def synthetic_corruption_report(frame: pd.DataFrame, target_column: str) -> SyntheticCorruptionReport:
    """Measure the planted defects from the actual injected frame."""
    target = frame[target_column]
    numeric_target = pd.to_numeric(target, errors="coerce")
    if numeric_target.isna().any():
        numeric_target = pd.Series(pd.factorize(target, sort=True)[0], index=target.index, dtype=float)
    return SyntheticCorruptionReport(
        seed=SYNTHETIC_CORRUPTION_SEED,
        columns={
            "constant": SYNTH_CONSTANT,
            "leakage": SYNTH_LEAKED,
            "high_missing": SYNTH_HIGH_MISSING,
        },
        constant_nunique=int(frame[SYNTH_CONSTANT].nunique(dropna=True)),
        leaked_corr_with_target=float(frame[SYNTH_LEAKED].corr(numeric_target)),
        high_missing_pct=float(frame[SYNTH_HIGH_MISSING].isna().mean() * 100.0),
    )
