"""Milestone 9 arm execution gates and cross-run benchmark aggregation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Literal

import pandas as pd
import numpy as np

from benchmark_datasets import BENCHMARKS, DEFAULT_BENCHMARK_DIR, load_benchmark
from pipeline_supervisor import PipelineState, run_supervised_pipeline
from preprocessing_agent import preprocessing_batch_count
from synthetic_corruptions import inject_synthetic_corruptions, synthetic_corruption_report


Arm = Literal["ab", "c"]
LAYER1_METRIC_KEYS = (
    "total_citations",
    "total_claims",
    "citation_fabrication_rate",
    "citation_misstatement_rate",
    "claim_fabrication_rate",
    "claim_misstatement_rate",
    "severity_violation_rate",
)
STAGES = ("eda", "preprocess", "model_selection", "evaluation")
WINE_COLUMN_ALIASES = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide",
}
FLEXIBLE_EXPECTATIONS = {
    "standard_or_robust_by_skew": {"standard", "robust"},
    "onehot_or_ordinal_by_cardinality": {"onehot", "ordinal"},
}
AMES_MODEL_ALIASES = {"logreg": "linreg"}
TITANIC_LEAKAGE_ADDENDUM_COLUMNS = ("boat", "body")


@dataclass(frozen=True)
class ExperimentRun:
    dataset: str
    arm: Arm
    temperature: float
    repeat: int

    @property
    def run_id(self) -> str:
        temperature_label = "t00" if self.temperature == 0.0 else "t07"
        return f"m9_{self.dataset}_{self.arm}_{temperature_label}_r{self.repeat}"


def planned_runs() -> list[ExperimentRun]:
    """Return the pre-registered Arm A/B and single-run Arm C matrix."""
    runs: list[ExperimentRun] = []
    for dataset in BENCHMARKS:
        runs.extend(ExperimentRun(dataset, "ab", 0.0, repeat) for repeat in (1, 2, 3))
        runs.append(ExperimentRun(dataset, "ab", 0.7, 1))
        runs.append(ExperimentRun(dataset, "c", 0.0, 1))
    return runs


def write_preregistration_template(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing pre-registration: {destination}")
    template = {
        name: {
            "notes": None,
            "column_plans": None,
            "model_ranking": None,
            "model_rejected": None,
            "primary_metric": None,
        }
        for name in BENCHMARKS
    }
    destination.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def validate_preregistration(path: str | Path) -> tuple[Path, str]:
    preregistration = Path(path)
    if not preregistration.exists():
        raise FileNotFoundError(
            f"Pre-registration is required before any arm: {preregistration} does not exist"
        )
    payload = json.loads(preregistration.read_text(encoding="utf-8"))
    missing = []
    for dataset in BENCHMARKS:
        entry = payload.get(dataset)
        if not isinstance(entry, dict):
            missing.append(dataset)
            continue
        has_column_plan = bool(entry.get("column_plans") or entry.get("column_plans_by_type"))
        required_values = [entry.get("model_ranking"), entry.get("model_rejected"), entry.get("primary_metric")]
        if not has_column_plan or not all(required_values):
            missing.append(dataset)
    if missing:
        raise ValueError(
            "Pre-registration must contain column_plans or column_plans_by_type, plus "
            "model_ranking, model_rejected, and primary_metric "
            f"for every benchmark; incomplete: {missing}"
        )
    digest = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    return preregistration, digest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _expected_call_range(n_features: int) -> tuple[int, int]:
    # Batched preprocessing generation, three other generating-agent calls,
    # 5-12 EDA critics, one critic per
    # preprocessing column, four model critics, and one evaluation critic.
    preprocess_batches = preprocessing_batch_count(n_features)
    return n_features + 13 + preprocess_batches, n_features + 20 + preprocess_batches


def _validate_planted_frame(original: pd.DataFrame, planted: pd.DataFrame, target: str) -> list[str]:
    if len(original) != len(planted) or not original.index.equals(planted.index):
        raise ValueError("plant_defects must preserve row count and index")
    if target not in planted or not original[target].equals(planted[target]):
        raise ValueError("plant_defects must preserve the target column exactly")
    removed = [column for column in original if column not in planted]
    added = [column for column in planted if column not in original]
    if removed or len(added) != 3:
        raise ValueError(
            "plant_defects must add exactly three columns and remove none; "
            f"added={added}, removed={removed}"
        )
    return added


def _identify_planted_defects(raw_profile: dict, added_columns: list[str]) -> dict[str, str]:
    rules = {
        "constant": "is_constant",
        "leakage": "leakage_suspect",
        "high_missing": "has_high_missing",
    }
    matches: dict[str, list[str]] = {}
    for defect, flag in rules.items():
        matches[defect] = [
            column for column in added_columns if raw_profile.get(f"col:{column}:{flag}") is True
        ]
    ambiguous = {defect: columns for defect, columns in matches.items() if len(columns) != 1}
    if ambiguous:
        raise ValueError(f"Planted defects do not verify as exactly one of each type: {ambiguous}")
    identified = {defect: columns[0] for defect, columns in matches.items()}
    if len(set(identified.values())) != 3:
        raise ValueError(f"Each planted defect must map to a distinct column: {identified}")
    return identified


def _arm_c_recall(state: PipelineState, planted: dict[str, str]) -> dict:
    flag_by_defect = {
        "constant": "is_constant",
        "leakage": "leakage_suspect",
        "high_missing": "has_high_missing",
    }
    rows = []
    for defect, column in planted.items():
        metric_key = f"col:{column}:{flag_by_defect[defect]}"
        matches = [
            finding.claim_id
            for finding in state.eda_findings
            if finding.severity == "critical"
            and any(reference.metric_key == metric_key for reference in finding.evidence)
        ]
        rows.append(
            {
                "defect": defect,
                "column": column,
                "expected_metric_key": metric_key,
                "recalled": bool(matches),
                "matching_claim_ids": matches,
            }
        )
    return {
        "recall": sum(row["recalled"] for row in rows),
        "total": 3,
        "defects": rows,
    }


def apply_manual_dataset_corrections(
    dataset: str,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply only explicitly pre-registered, dataset-specific source corrections."""
    corrected = frame.copy(deep=True)
    if dataset != "home_credit":
        return corrected, []
    if "DAYS_EMPLOYED" not in corrected:
        raise KeyError("Home Credit source is missing DAYS_EMPLOYED")
    if "DAYS_EMPLOYED_ANOM" in corrected:
        raise ValueError("Home Credit source already contains DAYS_EMPLOYED_ANOM")

    sentinel = 365243
    anomaly_mask = corrected["DAYS_EMPLOYED"].eq(sentinel)
    raw_mean = float(corrected["DAYS_EMPLOYED"].mean())
    corrected["DAYS_EMPLOYED_ANOM"] = anomaly_mask.astype(bool)
    corrected.loc[anomaly_mask, "DAYS_EMPLOYED"] = np.nan
    cleaned_mean = float(corrected["DAYS_EMPLOYED"].mean())
    return corrected, [
        {
            "column": "DAYS_EMPLOYED",
            "operation": "replace sentinel with NaN and add boolean anomaly indicator",
            "sentinel_value": sentinel,
            "rows_corrected": int(anomaly_mask.sum()),
            "raw_mean": raw_mean,
            "cleaned_mean": cleaned_mean,
            "indicator_column": "DAYS_EMPLOYED_ANOM",
        }
    ]


def _named_preregistered_plans(dataset: str, entry: dict) -> dict[str, dict]:
    if "column_plans" in entry:
        aliases = WINE_COLUMN_ALIASES if dataset == "wine_quality" else {}
        return {
            aliases.get(column, column): plan
            for column, plan in entry["column_plans"].items()
        }

    plans: dict[str, dict] = {}
    for group_name, group in entry["column_plans_by_type"].items():
        if group_name == "everything_else":
            continue
        for column in group.get("columns", []):
            plans[column] = {
                key: value
                for key, value in group.items()
                if key not in {"columns", "reason", "preprocessing_before_pipeline", "imputation_value"}
            }
    return plans


def validate_preregistered_columns(
    dataset: str,
    entry: dict,
    dataset_columns,
) -> None:
    """Reject unresolved reference names before any paid pipeline call."""
    missing = sorted(set(_named_preregistered_plans(dataset, entry)) - set(dataset_columns))
    if missing:
        raise ValueError(
            f"Pre-registration names do not match {dataset} columns after explicit aliases: {missing}"
        )


def _expected_value_matches(expected, actual) -> bool:
    if expected in FLEXIBLE_EXPECTATIONS:
        return actual in FLEXIBLE_EXPECTATIONS[expected]
    return expected == actual


def _titanic_leakage_addendum(state: PipelineState) -> dict:
    results = {}
    for column in TITANIC_LEAKAGE_ADDENDUM_COLUMNS:
        metric_key = f"col:{column}:leakage_suspect"
        finding_ids = [
            finding.claim_id
            for finding in state.eda_findings
            if finding.severity == "critical"
            and any(reference.metric_key == metric_key for reference in finding.evidence)
        ]
        results[column] = {
            "status": "pre-run addendum; not included in column-plan agreement score",
            "leakage_suspect": state.raw_profile.get(metric_key) if state.raw_profile else None,
            "critical_eda_finding_claim_ids": finding_ids,
        }
    return results


def compare_run_to_preregistration(
    dataset: str,
    state: PipelineState,
    preregistered_entry: dict,
    manual_corrections: list[dict],
) -> dict:
    """Compare a completed run to the static human baseline after execution."""
    expected_plans = _named_preregistered_plans(dataset, preregistered_entry)
    actual_plans = {plan.column: plan for plan in state.column_plans}
    comparison_fields = ("action", "imputation", "encoding", "scaling")
    column_rows = []
    for column, expected in expected_plans.items():
        actual = actual_plans.get(column)
        field_matches = {
            field: bool(actual is not None and _expected_value_matches(expected[field], getattr(actual, field)))
            for field in comparison_fields
            if field in expected
        }
        column_rows.append(
            {
                "column": column,
                "expected": {field: expected[field] for field in comparison_fields if field in expected},
                "actual": (
                    {field: getattr(actual, field) for field in comparison_fields}
                    if actual is not None
                    else None
                ),
                "field_matches": field_matches,
                "matches": bool(actual is not None and all(field_matches.values())),
            }
        )

    pipeline_dropped = sorted(plan.column for plan in state.column_plans if plan.action == "drop")
    preregistered_dropped = sorted(
        column for column, expected in expected_plans.items() if expected.get("action") == "drop"
    )
    expected_keep = {
        column for column, expected in expected_plans.items() if expected.get("action") != "drop"
    }
    actual_ranking = [
        choice.model_id
        for choice in sorted(
            (choice for choice in state.model_choices if not choice.is_rejection),
            key=lambda choice: choice.rank or 99,
        )
    ]
    model_aliases = AMES_MODEL_ALIASES if dataset == "ames_housing" else {}
    expected_ranking = [
        model_aliases.get(model_id, model_id)
        for model_id in preregistered_entry["model_ranking"]
    ]
    actual_rejected = next(
        (choice.model_id for choice in state.model_choices if choice.is_rejection),
        None,
    )
    expected_rejected = model_aliases.get(
        preregistered_entry["model_rejected"],
        preregistered_entry["model_rejected"],
    )
    actual_metric = (
        state.metric_choice.primary_metric
        if state.metric_choice is not None
        else (
            getattr(state, "evaluation_results", [])[0].primary_metric
            if getattr(state, "evaluation_results", []) else None
        )
    )
    named_columns = set(expected_plans)
    unscored_columns = sorted(set(actual_plans) - named_columns)
    matched_count = sum(row["matches"] for row in column_rows)

    comparison = {
        "column_plan_comparisons": column_rows,
        "column_plan_matches": matched_count,
        "column_plans_scored": len(column_rows),
        "column_plan_agreement_rate": matched_count / len(column_rows) if column_rows else None,
        "preregistered_drop_columns": preregistered_dropped,
        "pipeline_dropped_columns": pipeline_dropped,
        "expected_drop_but_kept": sorted(set(preregistered_dropped) - set(pipeline_dropped)),
        "expected_keep_but_dropped": sorted(expected_keep & set(pipeline_dropped)),
        "unscored_columns": unscored_columns,
        "model_comparison": {
            "preregistered_ranking": expected_ranking,
            "pipeline_ranking": actual_ranking,
            "top_model_match": bool(expected_ranking and actual_ranking and expected_ranking[0] == actual_ranking[0]),
            "ranking_exact_match": expected_ranking == actual_ranking,
            "preregistered_rejected": expected_rejected,
            "pipeline_rejected": actual_rejected,
            "rejected_model_match": expected_rejected == actual_rejected,
        },
        "metric_comparison": {
            "preregistered": preregistered_entry["primary_metric"],
            "pipeline": actual_metric,
            "matches": preregistered_entry["primary_metric"] == actual_metric,
        },
    }
    if dataset == "titanic":
        comparison["titanic_leakage_positive_control_addendum"] = _titanic_leakage_addendum(state)
    if dataset == "ames_housing":
        structural_columns = set(
            preregistered_entry["column_plans_by_type"]["structural_na_means_none"]["columns"]
        )
        dropped_structural = sorted(structural_columns & set(pipeline_dropped))
        comparison["ames_structural_na_divergence"] = {
            "occurred": bool(dropped_structural),
            "structural_na_columns_dropped": dropped_structural,
            "structural_na_columns_total": len(structural_columns),
        }
    if dataset == "home_credit":
        correction = next(
            (row for row in manual_corrections if row["column"] == "DAYS_EMPLOYED"),
            None,
        )
        comparison["home_credit_sentinel_divergence"] = {
            "manual_correction_applied": correction is not None,
            "uncorrected_sentinel_reached_profiler": correction is None,
            "correction_audit": correction,
            "profiled_missing_pct": (
                state.raw_profile.get("col:DAYS_EMPLOYED:missing_pct")
                if state.raw_profile else None
            ),
            "anomaly_indicator_profiled": bool(
                state.raw_profile and "col:DAYS_EMPLOYED_ANOM:dtype" in state.raw_profile
            ),
        }
    return comparison


def run_experiment(
    experiment: ExperimentRun,
    *,
    preregistration_path: str | Path,
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    runs_dir: str | Path = "runs",
    plant_defects: Callable[[pd.DataFrame, str, str], pd.DataFrame] | None = None,
    model: str | None = None,
    critic_model: str | None = None,
) -> PipelineState:
    """Run one explicitly selected arm after enforcing pre-registration."""
    if experiment.temperature not in {0.0, 0.7}:
        raise ValueError("Milestone 9 temperatures are fixed at 0.0 and 0.7")
    if experiment.arm == "c" and (experiment.temperature != 0.0 or experiment.repeat != 1):
        raise ValueError("Arm C has exactly one temperature=0.0 run per dataset")
    if experiment.arm == "ab" and experiment.temperature == 0.0 and experiment.repeat not in {1, 2, 3}:
        raise ValueError("temperature=0.0 repeats must be 1, 2, or 3")
    if experiment.arm == "ab" and experiment.temperature == 0.7 and experiment.repeat != 1:
        raise ValueError("temperature=0.7 has exactly one variance-check run")
    preregistration, preregistration_sha = validate_preregistration(preregistration_path)
    spec, source_frame, source_dataset_path = load_benchmark(
        experiment.dataset,
        benchmark_dir=benchmark_dir,
    )
    preregistered_payload = json.loads(preregistration.read_text(encoding="utf-8"))
    validate_preregistered_columns(
        experiment.dataset,
        preregistered_payload[experiment.dataset],
        source_frame.columns,
    )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY must be set before running an experimental arm")
    frame, manual_corrections = apply_manual_dataset_corrections(
        experiment.dataset,
        source_frame,
    )
    run_dir = Path(runs_dir) / experiment.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = source_dataset_path
    added_columns: list[str] = []
    if experiment.arm == "c":
        injector = plant_defects or inject_synthetic_corruptions
        planted_frame = injector(frame.copy(deep=True), spec.target_column, spec.task_type)
        added_columns = _validate_planted_frame(frame, planted_frame, spec.target_column)
        frame = planted_frame
        _write_json(run_dir / "synthetic_corruption.json", synthetic_corruption_report(frame, spec.target_column).to_dict())
    if manual_corrections or experiment.arm == "c":
        dataset_path = run_dir / "input_dataset.csv"
        frame.to_csv(dataset_path, index=False)

    state = run_supervised_pipeline(
        str(dataset_path),
        spec.target_column,
        spec.task_type,
        experiment.run_id,
        runs_dir=runs_dir,
        model=model,
        critic_model=critic_model,
        agent_temperature=experiment.temperature,
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metadata = {
        **asdict(experiment),
        "run_id": experiment.run_id,
        "reported_arms": ["A", "B"] if experiment.arm == "ab" else ["C"],
        "source_dataset_path": str(source_dataset_path),
        "dataset_path": str(dataset_path),
        "target_column": spec.target_column,
        "task_type": spec.task_type,
        "preregistration_path": str(preregistration),
        "preregistration_sha256": preregistration_sha,
        "planted_columns": added_columns,
        "manual_dataset_corrections": manual_corrections,
        "call_cap_used": state.call_cap_used,
    }
    _write_json(run_dir / "experiment_metadata.json", metadata)
    if experiment.arm == "ab":
        _write_json(run_dir / "arm_a_metrics.json", metrics)
        _write_json(run_dir / "arm_b_metrics.json", {key: metrics.get(key) for key in LAYER1_METRIC_KEYS})
        comparison = compare_run_to_preregistration(
            experiment.dataset,
            state,
            preregistered_payload[experiment.dataset],
            manual_corrections,
        )
        _write_json(run_dir / "preregistration_comparison.json", comparison)
    else:
        planted = _identify_planted_defects(state.raw_profile or {}, added_columns)
        recall = _arm_c_recall(state, planted)
        _write_json(run_dir / "arm_c_recall.json", recall)
        corruption = synthetic_corruption_report(frame, spec.target_column).to_dict()
        corruption["profile"] = {
            "leaked_corr_with_target": (state.raw_profile or {}).get("col:SYNTH_LEAKED:corr_with_target"),
            "constant_is_constant": (state.raw_profile or {}).get("col:SYNTH_CONSTANT:is_constant"),
            "leaked_leakage_suspect": (state.raw_profile or {}).get("col:SYNTH_LEAKED:leakage_suspect"),
            "high_missing_has_high_missing": (state.raw_profile or {}).get("col:SYNTH_HIGH_MISSING:has_high_missing"),
        }
        _write_json(run_dir / "synthetic_corruption.json", corruption)
    return state


def _note_has_select_despite_risk(note: str) -> bool:
    normalized = note.lower()
    risk = re.search(r"risk|overfit|overfitting|high[ -]variance", normalized)
    contradiction = re.search(r"contradict|inconsisten|despite|yet select|still select|reconcil", normalized)
    return bool(risk and contradiction)


def _note_has_invented_narrative(note: str) -> bool:
    normalized = note.lower()
    unsupported = re.search(r"unsupported|not supported|not evidenced|invent", normalized)
    narrative = re.search(r"business|domain|outcome|narrative|real-world|context", normalized)
    return bool(unsupported and narrative)


def _pattern_counts(run_dir: Path) -> tuple[int, int]:
    claims = {row["claim_id"]: row for row in _read_jsonl(run_dir / "claims.jsonl")}
    verdicts = _read_jsonl(run_dir / "layer2_verdicts.jsonl")
    d12 = 0
    d13 = 0
    for verdict in verdicts:
        claim = claims.get(verdict["claim_id"], {})
        note = str(verdict.get("discrepancy_note", ""))
        if claim.get("stage") == "model_selection" and _note_has_select_despite_risk(note):
            d12 += 1
        if claim.get("stage") == "evaluation" and _note_has_invented_narrative(note):
            d13 += 1
    return d12, d13


def _average_metric(run_records: list[dict], key: str) -> float | None:
    values = [record["metrics"].get(key) for record in run_records]
    usable = [float(value) for value in values if value is not None]
    return mean(usable) if usable else None


def _stage_summary(run_records: list[dict]) -> tuple[dict[str, float | None], dict[str, float | None]]:
    means: dict[str, float | None] = {}
    stds: dict[str, float | None] = {}
    for stage in STAGES:
        scores = [
            record["metrics"].get("mean_layer2_score_by_stage", {}).get(stage)
            for record in run_records
        ]
        usable = [float(score) for score in scores if score is not None]
        means[stage] = mean(usable) if usable else None
        stds[stage] = pstdev(usable) if usable else None
    return means, stds


def _decision_set(run_dir: Path) -> set[tuple[str, str]]:
    return {
        (row.get("stage", ""), row.get("decision", ""))
        for row in _read_jsonl(run_dir / "claims.jsonl")
    }


def _temperature_variance_note(temp0: list[dict], temp07: list[dict]) -> str:
    if not temp0 or not temp07:
        return "Variance comparison unavailable until both temperature conditions are complete."
    baseline_means, _ = _stage_summary(temp0)
    warm_means, _ = _stage_summary(temp07)
    score_deltas = {
        stage: abs(warm_means[stage] - baseline_means[stage])
        for stage in STAGES
        if warm_means[stage] is not None and baseline_means[stage] is not None
    }
    baseline_decisions = set.intersection(*(_decision_set(record["run_dir"]) for record in temp0))
    warm_decisions = _decision_set(temp07[0]["run_dir"])
    changed_decisions = len(baseline_decisions.symmetric_difference(warm_decisions))
    notable_stages = [stage for stage, delta in score_deltas.items() if delta >= 0.25]
    if notable_stages or changed_decisions:
        return (
            f"Notable variance: score delta >=0.25 in {notable_stages}; "
            f"{changed_decisions} consensus decision entries differed."
        )
    return "No notable variance in stage means or consensus decisions at temperature 0.7."


def _load_experiment_records(runs_dir: Path) -> list[dict]:
    records = []
    for metadata_path in sorted(runs_dir.glob("*/experiment_metadata.json")):
        run_dir = metadata_path.parent
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        records.append(
            {
                "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                "run_dir": run_dir,
            }
        )
    return records


def _aggregate_preregistration_comparisons(run_records: list[dict]) -> dict:
    comparisons = []
    for record in run_records:
        path = record["run_dir"] / "preregistration_comparison.json"
        if path.exists():
            comparisons.append(
                (
                    record["metadata"]["run_id"],
                    json.loads(path.read_text(encoding="utf-8")),
                )
            )
    if not comparisons:
        return {
            "temp0_runs_compared": 0,
            "column_plan_agreement_rate": None,
            "top_model_match_rate": None,
            "ranking_exact_match_rate": None,
            "rejected_model_match_rate": None,
            "metric_match_rate": None,
            "preregistered_drop_columns": [],
            "pipeline_dropped_columns_by_run": {},
        }

    def boolean_rate(path: tuple[str, str]) -> float:
        section, key = path
        return mean(bool(comparison[section][key]) for _, comparison in comparisons)

    agreement_rates = [
        comparison["column_plan_agreement_rate"]
        for _, comparison in comparisons
        if comparison["column_plan_agreement_rate"] is not None
    ]
    result = {
        "temp0_runs_compared": len(comparisons),
        "column_plan_agreement_rate": mean(agreement_rates) if agreement_rates else None,
        "top_model_match_rate": boolean_rate(("model_comparison", "top_model_match")),
        "ranking_exact_match_rate": boolean_rate(("model_comparison", "ranking_exact_match")),
        "rejected_model_match_rate": boolean_rate(("model_comparison", "rejected_model_match")),
        "metric_match_rate": boolean_rate(("metric_comparison", "matches")),
        "preregistered_drop_columns": comparisons[0][1]["preregistered_drop_columns"],
        "pipeline_dropped_columns_by_run": {
            run_id: comparison["pipeline_dropped_columns"]
            for run_id, comparison in comparisons
        },
        "expected_keep_but_dropped_by_run": {
            run_id: comparison["expected_keep_but_dropped"]
            for run_id, comparison in comparisons
        },
        "expected_drop_but_kept_by_run": {
            run_id: comparison["expected_drop_but_kept"]
            for run_id, comparison in comparisons
        },
        "model_comparison_by_run": {
            run_id: comparison["model_comparison"]
            for run_id, comparison in comparisons
        },
        "metric_comparison_by_run": {
            run_id: comparison["metric_comparison"]
            for run_id, comparison in comparisons
        },
    }
    for key in (
        "titanic_leakage_positive_control_addendum",
        "ames_structural_na_divergence",
        "home_credit_sentinel_divergence",
    ):
        relevant = {
            run_id: comparison[key]
            for run_id, comparison in comparisons
            if key in comparison
        }
        if relevant:
            result[f"{key}_by_run"] = relevant
    return result


def aggregate_benchmark_runs(
    *,
    runs_dir: str | Path = "runs",
    output_path: str | Path = "benchmark_summary.json",
) -> dict:
    """Aggregate the three temperature-zero repeats and one variance run."""
    records = _load_experiment_records(Path(runs_dir))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["metadata"]["dataset"]].append(record)

    per_dataset: dict[str, dict] = {}
    for dataset in BENCHMARKS:
        dataset_records = grouped.get(dataset, [])
        ab_temp0 = [
            row for row in dataset_records
            if row["metadata"]["arm"] == "ab" and row["metadata"]["temperature"] == 0.0
        ]
        ab_temp07 = [
            row for row in dataset_records
            if row["metadata"]["arm"] == "ab" and row["metadata"]["temperature"] == 0.7
        ]
        c_temp0 = [
            row for row in dataset_records
            if row["metadata"]["arm"] == "c" and row["metadata"]["temperature"] == 0.0
        ]
        stage_means, stage_stds = _stage_summary(ab_temp0)
        d12 = d13 = 0
        for record in ab_temp0:
            run_d12, run_d13 = _pattern_counts(record["run_dir"])
            d12 += run_d12
            d13 += run_d13

        recalled_sets = []
        for record in c_temp0:
            recall_path = record["run_dir"] / "arm_c_recall.json"
            if recall_path.exists():
                recall = json.loads(recall_path.read_text(encoding="utf-8"))
                recalled_sets.append(
                    {row["defect"] for row in recall["defects"] if row["recalled"]}
                )
        consistently_recalled = set.intersection(*recalled_sets) if recalled_sets else set()
        per_dataset[dataset] = {
            "mean_layer2_score_by_stage": stage_means,
            "std_layer2_score_by_stage": stage_stds,
            "citation_fabrication_rate": _average_metric(ab_temp0, "citation_fabrication_rate"),
            "severity_violation_rate": _average_metric(ab_temp0, "severity_violation_rate"),
            "arm_c_recall": len(consistently_recalled),
            "select_despite_risk_count": d12,
            "invented_narrative_count": d13,
            "temp07_variance_note": _temperature_variance_note(ab_temp0, ab_temp07),
            "preregistration_comparison": _aggregate_preregistration_comparisons(ab_temp0),
            "temp0_ab_runs_aggregated": len(ab_temp0),
            "temp0_c_runs_aggregated": len(c_temp0),
        }

    all_stage_means: dict[str, float | None] = {}
    for stage in STAGES:
        values = [
            row["mean_layer2_score_by_stage"][stage]
            for row in per_dataset.values()
            if row["mean_layer2_score_by_stage"][stage] is not None
        ]
        all_stage_means[stage] = mean(values) if values else None
    model_selection_lowest = 0
    for row in per_dataset.values():
        scores = row["mean_layer2_score_by_stage"]
        model_score = scores.get("model_selection")
        other_scores = [score for stage, score in scores.items() if stage != "model_selection" and score is not None]
        if model_score is not None and other_scores and model_score < min(other_scores):
            model_selection_lowest += 1

    summary = {
        "per_dataset": per_dataset,
        "aggregate": {
            "overall_arm_c_recall_rate": sum(row["arm_c_recall"] for row in per_dataset.values()) / 15,
            "mean_layer2_score_by_stage_all_datasets": all_stage_means,
            "model_selection_lowest_in_n_of_5_datasets": model_selection_lowest,
            "select_despite_risk_total": sum(row["select_despite_risk_count"] for row in per_dataset.values()),
            "invented_narrative_total": sum(row["invented_narrative_count"] for row in per_dataset.values()),
        },
    }
    _write_json(Path(output_path), summary)
    return summary


def _load_defect_function(module_path: str | Path, dataset: str) -> Callable:
    path = Path(module_path)
    spec = importlib.util.spec_from_file_location("milestone9_user_defects", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import defect module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    functions = getattr(module, "PLANT_DEFECTS", None)
    if not isinstance(functions, dict) or dataset not in functions or not callable(functions[dataset]):
        raise ValueError("Defect module must expose PLANT_DEFECTS mapping every dataset name to a callable")
    return functions[dataset]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan, run, or aggregate Milestone 9 experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    init = subparsers.add_parser("init-preregistration")
    init.add_argument("--output", default="benchmarks/preregistered_plans.json")
    run = subparsers.add_parser("run")
    run.add_argument("--dataset", choices=list(BENCHMARKS), required=True)
    run.add_argument("--arm", choices=["ab", "c"], required=True)
    run.add_argument("--temperature", choices=[0.0, 0.7], type=float, required=True)
    run.add_argument("--repeat", choices=[1, 2, 3], type=int, required=True)
    run.add_argument("--preregistration", default="benchmarks/preregistered_plans.json")
    run.add_argument("--defect-module")
    run.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR))
    run.add_argument("--runs-dir", default="runs")
    run.add_argument("--model")
    run.add_argument("--critic-model")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--runs-dir", default="runs")
    aggregate.add_argument("--output", default="benchmark_summary.json")
    args = parser.parse_args()

    if args.command == "plan":
        matrix = [asdict(run) | {"run_id": run.run_id} for run in planned_runs()]
        print(json.dumps({"total_executions": len(matrix), "runs": matrix}, indent=2))
        return 0
    if args.command == "init-preregistration":
        print(write_preregistration_template(args.output))
        return 0
    if args.command == "aggregate":
        print(json.dumps(aggregate_benchmark_runs(runs_dir=args.runs_dir, output_path=args.output), indent=2))
        return 0

    experiment = ExperimentRun(args.dataset, args.arm, args.temperature, args.repeat)
    defect_function = None
    if args.arm == "c":
        defect_function = _load_defect_function(args.defect_module, args.dataset) if args.defect_module else None
    state = run_experiment(
        experiment,
        preregistration_path=args.preregistration,
        benchmark_dir=args.benchmark_dir,
        runs_dir=args.runs_dir,
        plant_defects=defect_function,
        model=args.model,
        critic_model=args.critic_model,
    )
    print(json.dumps({"run_id": state.run_id, "completed": state.completed, "aborted": state.aborted}, indent=2))
    return 1 if state.aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
