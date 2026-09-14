"""No-network tests for Milestone 9 experiment gates and aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from benchmark_experiments import (
    ExperimentRun,
    _arm_c_recall,
    _expected_value_matches,
    _identify_planted_defects,
    _validate_planted_frame,
    apply_manual_dataset_corrections,
    aggregate_benchmark_runs,
    compare_run_to_preregistration,
    planned_runs,
    validate_preregistered_columns,
    validate_preregistration,
    write_preregistration_template,
)
from evidence_validator import EvidenceRef
from pipeline_supervisor import compute_call_cap
from preprocessing_agent import derive_column_plan_defaults
from profile_dataset import profile_dataset
from synthetic_corruptions import (
    HIGH_MISSING_FRACTION,
    SYNTH_CONSTANT,
    SYNTH_HIGH_MISSING,
    SYNTH_LEAKED,
    TARGET_LEAK_CORRELATION,
    inject_synthetic_corruptions,
    synthetic_corruption_report,
)


def test_fixed_matrix_has_25_unique_executions() -> None:
    runs = planned_runs()
    assert len(runs) == 25
    assert len({run.run_id for run in runs}) == 25
    assert sum(run.arm == "ab" for run in runs) == 20
    assert sum(run.arm == "c" for run in runs) == 5
    assert all(run.temperature == 0.0 and run.repeat == 1 for run in runs if run.arm == "c")


def test_titanic_arm_c_injection_is_reproducible_and_measured() -> None:
    frame = pd.read_csv("benchmarks/titanic.csv")
    first = inject_synthetic_corruptions(frame, "survived", "binary")
    second = inject_synthetic_corruptions(frame, "survived", "binary")
    report = synthetic_corruption_report(first, "survived")
    profile = profile_dataset(first, "survived", "binary")
    defaults = {
        default.column: default
        for default in derive_column_plan_defaults(profile, "survived")
    }

    assert first[[SYNTH_CONSTANT, SYNTH_LEAKED, SYNTH_HIGH_MISSING]].equals(
        second[[SYNTH_CONSTANT, SYNTH_LEAKED, SYNTH_HIGH_MISSING]]
    )
    assert report.constant_nunique == 1
    assert report.leaked_corr_with_target == pytest.approx(TARGET_LEAK_CORRELATION, abs=1e-12)
    assert report.high_missing_pct == pytest.approx(
        round(round(len(frame) * HIGH_MISSING_FRACTION) / len(frame) * 100.0, 12)
    )
    assert profile[f"col:{SYNTH_CONSTANT}:is_constant"] is True
    assert profile[f"col:{SYNTH_LEAKED}:leakage_suspect"] is True
    assert profile[f"col:{SYNTH_HIGH_MISSING}:has_high_missing"] is True
    assert defaults[SYNTH_CONSTANT].drop_reason == "is_constant"
    assert defaults[SYNTH_LEAKED].drop_reason == "leakage_suspect"
    assert defaults[SYNTH_HIGH_MISSING].drop_reason == "has_high_missing"


def test_arm_c_categorical_target_leakage_is_profiled() -> None:
    frame = pd.DataFrame(
        {
            "feature": range(1_000),
            "income": [">50K" if index % 4 == 0 else "<=50K" for index in range(1_000)],
        }
    )

    corrupted = inject_synthetic_corruptions(frame, "income", "binary")
    profile = profile_dataset(corrupted, "income", "binary")

    assert profile[f"col:{SYNTH_LEAKED}:corr_with_target"] > 0.95
    assert profile[f"col:{SYNTH_LEAKED}:leakage_suspect"] is True


def test_call_caps_cover_preflight_upper_estimates_with_target_headroom() -> None:
    estimates = {
        "home_credit": (122, 150, 9),
        "ames_housing": (82, 107, 6),
        "adult_income": (15, 35, 1),
        "wine_quality": (12, 32, 1),
        "titanic": (14, 34, 1),
    }
    for _, (n_columns, upper_estimate, preprocess_batches) in estimates.items():
        cap = compute_call_cap(n_columns)
        assert cap > upper_estimate
        assert cap >= upper_estimate + preprocess_batches


def test_preregistration_gate_rejects_template_until_user_fills_it(tmp_path: Path) -> None:
    path = write_preregistration_template(tmp_path / "plans.json")
    with pytest.raises(ValueError, match="incomplete"):
        validate_preregistration(path)
    payload = json.loads(path.read_text())
    for entry in payload.values():
        entry["column_plans"] = {"feature": {"action": "scale"}}
        entry["model_ranking"] = ["gbt", "rf", "logreg"]
        entry["model_rejected"] = "knn"
        entry["primary_metric"] = "roc_auc"
    path.write_text(json.dumps(payload))
    _, digest = validate_preregistration(path)
    assert len(digest) == 64


def test_arm_c_validates_and_scores_exact_evidence_keys() -> None:
    original = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})
    planted = original.assign(constant_defect=1, leakage_defect=original["target"], missing_defect=[None] * 6 + list(range(4)))
    added = _validate_planted_frame(original, planted, "target")
    profile = {
        **{f"col:{column}:is_constant": column == "constant_defect" for column in added},
        **{f"col:{column}:leakage_suspect": column == "leakage_defect" for column in added},
        **{f"col:{column}:has_high_missing": column == "missing_defect" for column in added},
    }
    identified = _identify_planted_defects(profile, added)
    findings = [
        SimpleNamespace(
            claim_id=f"eda_{index:03d}",
            severity="critical",
            evidence=[EvidenceRef(f"col:{column}:{flag}", True)],
        )
        for index, (column, flag) in enumerate(
            [
                ("constant_defect", "is_constant"),
                ("leakage_defect", "leakage_suspect"),
                ("missing_defect", "has_high_missing"),
            ],
            1,
        )
    ]
    recall = _arm_c_recall(SimpleNamespace(eda_findings=findings), identified)
    assert recall["recall"] == 3


def test_home_credit_manual_correction_is_run_local_and_audited() -> None:
    source = pd.DataFrame({"DAYS_EMPLOYED": [365243, -100, -200], "TARGET": [0, 1, 0]})
    corrected, audit = apply_manual_dataset_corrections("home_credit", source)
    assert source["DAYS_EMPLOYED"].tolist() == [365243, -100, -200]
    assert corrected["DAYS_EMPLOYED"].isna().tolist() == [True, False, False]
    assert corrected["DAYS_EMPLOYED_ANOM"].tolist() == [True, False, False]
    assert audit[0]["rows_corrected"] == 1
    assert audit[0]["raw_mean"] != audit[0]["cleaned_mean"]


def test_comparison_uses_wine_aliases_and_titanic_addendum() -> None:
    wine_state = SimpleNamespace(
        column_plans=[SimpleNamespace(column="fixed acidity", action="scale", imputation="none", encoding="none", scaling="standard")],
        model_choices=[
            SimpleNamespace(model_id="gbt", rank=1, is_rejection=False),
            SimpleNamespace(model_id="rf", rank=2, is_rejection=False),
            SimpleNamespace(model_id="logreg", rank=3, is_rejection=False),
            SimpleNamespace(model_id="knn", rank=None, is_rejection=True),
        ],
        metric_choice=SimpleNamespace(primary_metric="macro_f1"),
        raw_profile={},
        eda_findings=[],
    )
    wine_entry = {
        "column_plans": {"fixed_acidity": {"action": "scale", "scaling": "standard"}},
        "model_ranking": ["gbt", "rf", "logreg"],
        "model_rejected": "knn",
        "primary_metric": "macro_f1",
    }
    comparison = compare_run_to_preregistration("wine_quality", wine_state, wine_entry, [])
    assert comparison["column_plan_agreement_rate"] == 1.0
    assert comparison["column_plan_comparisons"][0]["column"] == "fixed acidity"

    titanic_state = SimpleNamespace(
        column_plans=[SimpleNamespace(column="boat", action="drop", imputation="none", encoding="none", scaling="none")],
        model_choices=[],
        metric_choice=None,
        raw_profile={"col:boat:leakage_suspect": True, "col:body:leakage_suspect": False},
        eda_findings=[SimpleNamespace(claim_id="eda_001", severity="critical", evidence=[EvidenceRef("col:boat:leakage_suspect", True)])],
    )
    titanic_entry = {
        "column_plans": {},
        "model_ranking": [],
        "model_rejected": "knn",
        "primary_metric": "roc_auc",
    }
    comparison = compare_run_to_preregistration("titanic", titanic_state, titanic_entry, [])
    addendum = comparison["titanic_leakage_positive_control_addendum"]
    assert addendum["boat"]["leakage_suspect"] is True
    assert addendum["boat"]["critical_eda_finding_claim_ids"] == ["eda_001"]


def test_named_column_validation_is_exact_after_wine_only_aliases() -> None:
    wine_entry = {
        "column_plans": {"fixed_acidity": {"action": "scale"}},
    }
    validate_preregistered_columns("wine_quality", wine_entry, ["fixed acidity"])
    with pytest.raises(ValueError, match="FireplaceQu"):
        validate_preregistered_columns(
            "ames_housing",
            {"column_plans_by_type": {"structural_na_means_none": {"columns": ["FireplaceQu"], "action": "impute_and_encode"}}},
            ["Fireplace.Qu"],
        )
    assert _expected_value_matches("standard_or_robust_by_skew", "standard")
    assert _expected_value_matches("standard_or_robust_by_skew", "robust")
    assert _expected_value_matches("onehot_or_ordinal_by_cardinality", "ordinal")
    assert not _expected_value_matches("onehot_or_ordinal_by_cardinality", "target")


def _write_run(root: Path, dataset: str, repeat: int, *, temp: float = 0.0) -> None:
    run = ExperimentRun(dataset, "ab", temp, repeat)
    run_dir = root / run.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "experiment_metadata.json").write_text(json.dumps({**run.__dict__, "run_id": run.run_id}))
    (run_dir / "metrics.json").write_text(json.dumps({
        "mean_layer2_score_by_stage": {"eda": 3.0, "preprocess": 2.8, "model_selection": 1.5, "evaluation": 2.5},
        "citation_fabrication_rate": 0.0,
        "severity_violation_rate": 0.1,
    }))
    (run_dir / "claims.jsonl").write_text(
        json.dumps({"claim_id": "model_select_001", "stage": "model_selection", "decision": "Select risky model"}) + "\n"
        + json.dumps({"claim_id": "evaluate_001", "stage": "evaluation", "decision": "Use metric for business outcome"}) + "\n"
    )
    (run_dir / "layer2_verdicts.jsonl").write_text(
        json.dumps({"claim_id": "model_select_001", "layer2_score": 1, "discrepancy_note": "The justification acknowledges overfitting risk yet still selects it without reconciliation."}) + "\n"
        + json.dumps({"claim_id": "evaluate_001", "layer2_score": 2, "discrepancy_note": "The business outcome narrative is unsupported by the cited evidence."}) + "\n"
    )
    (run_dir / "preregistration_comparison.json").write_text(json.dumps({
        "column_plan_agreement_rate": 0.75,
        "preregistered_drop_columns": ["fnlwgt", "education"],
        "pipeline_dropped_columns": ["fnlwgt"],
        "expected_keep_but_dropped": [],
        "expected_drop_but_kept": ["education"],
        "model_comparison": {"top_model_match": True, "ranking_exact_match": False, "rejected_model_match": True},
        "metric_comparison": {"matches": True},
    }))


def test_aggregation_tracks_d12_d13_and_stage_means(tmp_path: Path) -> None:
    for repeat in (1, 2, 3):
        _write_run(tmp_path, "adult_income", repeat)
    _write_run(tmp_path, "adult_income", 1, temp=0.7)
    summary = aggregate_benchmark_runs(runs_dir=tmp_path, output_path=tmp_path / "summary.json")
    adult = summary["per_dataset"]["adult_income"]
    assert adult["mean_layer2_score_by_stage"]["model_selection"] == 1.5
    assert adult["std_layer2_score_by_stage"]["model_selection"] == 0.0
    assert adult["select_despite_risk_count"] == 3
    assert adult["invented_narrative_count"] == 3
    assert adult["preregistration_comparison"]["temp0_runs_compared"] == 3
    assert adult["preregistration_comparison"]["column_plan_agreement_rate"] == 0.75
    assert adult["preregistration_comparison"]["metric_match_rate"] == 1.0
    assert summary["aggregate"]["model_selection_lowest_in_n_of_5_datasets"] == 1
