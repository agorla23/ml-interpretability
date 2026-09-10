"""Executable regression tests for the preprocessing leakage guard."""

from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from evidence_validator import EvidenceRef
from preprocessing_agent import (
    PREPROCESSING_BATCH_SIZE,
    ColumnPlan,
    _apply_llm_payloads,
    _make_plan_from_default,
    derive_column_plan_defaults,
    run_preprocessing_agent,
)
from preprocessing_pipeline import (
    KFoldTargetEncoder,
    _build_transformers,
    build_preprocessing_pipeline,
)
from profile_dataset import profile_dataset


MINI_DATASET = Path("/Users/akhilgorla/Downloads/mini_dataset.csv")
AMES_DATASET = Path("benchmarks/ames_housing.csv")
AMES_PROFILE = Path("benchmarks/profiles/ames_housing.json")
HOME_CREDIT_DATASET = Path("benchmarks/home_credit_50k.csv")
HOME_CREDIT_PROFILE = Path("benchmarks/profiles/home_credit.json")


class PriorRecordingTargetEncoder(KFoldTargetEncoder):
    """Record explicit priors so the test can inspect inner-fold behavior."""

    def _fit_column_mapping(self, values, target, prior):
        if not hasattr(self, "recorded_priors_"):
            self.recorded_priors_ = []
        self.recorded_priors_.append(float(prior))
        return super()._fit_column_mapping(values, target, prior)


def _column_plans_from_profile(raw_profile: dict, target_column: str) -> list[ColumnPlan]:
    plans: list[ColumnPlan] = []
    for index, default in enumerate(
        derive_column_plan_defaults(raw_profile, target_column),
        start=1,
    ):
        evidence = [
            EvidenceRef(metric_key=key, claimed_value=raw_profile[key])
            for key in default.evidence_keys
        ]
        plans.append(
            ColumnPlan(
                claim_id=f"preprocess_test_{index:03d}",
                stage="preprocessing",
                decision=f"Apply deterministic preprocessing to {default.column}",
                justification="Executable pipeline regression test.",
                evidence=evidence,
                confidence=1.0,
                column=default.column,
                action=default.action,
                imputation=default.imputation,
                encoding=default.encoding,
                scaling=default.scaling,
                rule_derived_action=default.action,
                override_reason="",
                is_override=False,
            )
        )
    return plans


def test_kfold_target_encoder_runs_and_uses_inner_priors() -> None:
    rng = np.random.default_rng(7)
    categories = np.array([f"category_{index:02d}" for index in range(20)])
    X = pd.DataFrame({"high_cardinality": np.tile(categories, 10)})
    y = pd.Series(rng.integers(0, 2, size=len(X)))

    encoder = PriorRecordingTargetEncoder(n_splits=5, smoothing=10.0, random_state=11)
    encoded = encoder.fit_transform(X, y)
    transformed = encoder.transform(X.iloc[:13])

    assert encoded.shape == (200, 1)
    assert transformed.shape == (13, 1)
    inner_priors = encoder.recorded_priors_[:-1]
    assert len(inner_priors) == 5
    assert any(not np.isclose(prior, encoder.global_mean_) for prior in inner_priors)

    print(f"fit_transform_shape={encoded.shape}")
    print(f"transform_shape={transformed.shape}")
    print(f"outer_global_mean={encoder.global_mean_:.6f}")
    print(f"inner_priors={[round(prior, 6) for prior in inner_priors]}")
    print("inner_prior_differs_from_outer=True")


def test_kfold_target_encoder_rejects_degenerate_oof_input() -> None:
    encoder = KFoldTargetEncoder()
    try:
        encoder.fit_transform(pd.DataFrame({"category": ["only"]}), pd.Series([1]))
    except ValueError as error:
        assert str(error) == (
            "KFoldTargetEncoder requires at least 2 rows for out-of-fold encoding"
        )
    else:
        raise AssertionError("Single-row OOF encoding did not raise ValueError")
    print("single_row_oof_rejected=True")


def test_build_pipeline_cross_validate_on_mini_dataset() -> None:
    df = pd.read_csv(MINI_DATASET)
    raw_profile = profile_dataset(df, target_column="target", task_type="binary")
    plans = _column_plans_from_profile(raw_profile, target_column="target")
    preprocessing = build_preprocessing_pipeline(plans, raw_profile)
    estimator = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=17)
    results = cross_validate(
        estimator,
        df.drop(columns=["target"]),
        df["target"],
        cv=cv,
        scoring="accuracy",
        error_score="raise",
    )

    kept_columns = [plan.column for plan in plans if plan.action != "drop"]
    transformer_groups = _build_transformers(plans, raw_profile)
    assert len(results["test_score"]) == 4
    assert np.isfinite(results["test_score"]).all()
    assert sum(len(columns) for _, _, columns in transformer_groups) == len(kept_columns)
    assert len(transformer_groups) < len(kept_columns)

    print(f"kept_columns={kept_columns}")
    print(f"transformer_group_count={len(transformer_groups)}")
    print(f"test_scores={results['test_score'].tolist()}")
    print(f"mean_test_accuracy={results['test_score'].mean():.6f}")
    print("cross_validate_completed=True")


def _plans_for_columns(profile_path: Path, target_column: str, columns: list[str]) -> list[ColumnPlan]:
    raw_profile = json.loads(profile_path.read_text())
    defaults = {
        default.column: default
        for default in derive_column_plan_defaults(raw_profile, target_column)
    }
    return [
        _make_plan_from_default(defaults[column], raw_profile, index)
        for index, column in enumerate(columns, start=1)
    ]


def test_numeric_categories_with_missing_values_cross_validate_on_ames_and_home_credit() -> None:
    ames_columns = ["Bsmt.Full.Bath", "Bsmt.Half.Bath", "Garage.Cars"]
    home_credit_columns = [
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_QRT",
    ]

    ames = pd.read_csv(AMES_DATASET)
    ames_profile = json.loads(AMES_PROFILE.read_text())
    ames_plans = _plans_for_columns(AMES_PROFILE, "price", ames_columns)
    ames_estimator = Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline(ames_plans, ames_profile)),
            ("model", Ridge()),
        ]
    )
    ames_scores = cross_validate(
        ames_estimator,
        ames[ames_columns],
        ames["price"],
        cv=KFold(n_splits=3, shuffle=True, random_state=17),
        scoring="neg_mean_absolute_error",
        error_score="raise",
    )["test_score"]

    home_credit = pd.read_csv(HOME_CREDIT_DATASET)
    home_credit_profile = json.loads(HOME_CREDIT_PROFILE.read_text())
    home_credit_plans = _plans_for_columns(HOME_CREDIT_PROFILE, "TARGET", home_credit_columns)
    home_credit_estimator = Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline(home_credit_plans, home_credit_profile)),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    home_credit_scores = cross_validate(
        home_credit_estimator,
        home_credit[home_credit_columns],
        home_credit["TARGET"],
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=17),
        scoring="roc_auc",
        error_score="raise",
    )["test_score"]

    assert np.isfinite(ames_scores).all()
    assert np.isfinite(home_credit_scores).all()
    print(f"ames_neg_mae_scores={ames_scores.tolist()}")
    print(f"home_credit_roc_auc_scores={home_credit_scores.tolist()}")


class BatchEchoClient:
    def __init__(self):
        self.messages = self
        self.requests = []

    def create(self, **kwargs):
        request = json.loads(kwargs["messages"][0]["content"])
        self.requests.append(request)
        plans = []
        for default in request["rule_defaults"]:
            evidence_key = default["evidence_keys"][0]
            plans.append(
                {
                    "column": default["column"],
                    "justification": f"Batch justification for {default['column']}.",
                    "evidence": [
                        {
                            "metric_key": evidence_key,
                            "claimed_value": request["profile_subset"][evidence_key],
                        }
                    ],
                    "confidence": 0.9,
                    "override_reason": "",
                    "imputation": default["imputation"],
                    "encoding": default["encoding"],
                    "scaling": default["scaling"],
                }
            )
        block = SimpleNamespace(
            type="tool_use",
            name="record_preprocessing_plans",
            input={"plans": plans},
        )
        return SimpleNamespace(content=[block])


def test_preprocessing_agent_batches_wide_profiles_with_chunk_local_evidence() -> None:
    rng = np.random.default_rng(31)
    feature_columns = [f"feature_{index:02d}" for index in range(32)]
    frame = pd.DataFrame(
        rng.normal(size=(80, len(feature_columns))),
        columns=feature_columns,
    )
    frame["target"] = rng.integers(0, 2, size=len(frame))
    raw_profile = profile_dataset(frame, "target", "binary")
    client = BatchEchoClient()

    plans, _ = run_preprocessing_agent(
        raw_profile,
        frame,
        "target",
        client=client,
    )

    assert PREPROCESSING_BATCH_SIZE == 15
    assert [len(request["rule_defaults"]) for request in client.requests] == [15, 15, 2]
    assert [plan.claim_id for plan in plans] == [
        f"prep_{index:03d}" for index in range(1, 33)
    ]
    assert all(plan.justification and plan.confidence > 0 for plan in plans)
    for request in client.requests:
        batch_columns = {default["column"] for default in request["rule_defaults"]}
        assert all(
            any(key.startswith(f"col:{column}:") for column in batch_columns)
            for key in request["profile_subset"]
        )

    print(f"batch_sizes={[len(request['rule_defaults']) for request in client.requests]}")
    print(f"claim_ids={[plan.claim_id for plan in plans]}")


def test_same_transform_override_reason_keeps_rule_plan_without_consuming_override() -> None:
    raw_profile = {
        "col:feature:dtype": "float64",
        "col:feature:nunique": 20,
    }
    plan = ColumnPlan(
        claim_id="prep_001",
        stage="preprocess",
        decision="scale feature",
        justification="Model justification.",
        evidence=[EvidenceRef("col:feature:dtype", "float64")],
        confidence=0.9,
        column="feature",
        action="scale",
        imputation="none",
        encoding="none",
        scaling="standard",
        rule_derived_action="scale",
        override_reason="",
        is_override=False,
    )
    payload = {
        "column": "feature",
        "justification": "The standard rule is appropriate.",
        "evidence": [{"metric_key": "col:feature:dtype", "claimed_value": "float64"}],
        "confidence": 0.9,
        "override_reason": "Considered robust scaling, but the rule-selected standard scaling remains appropriate.",
        "imputation": "none",
        "encoding": "none",
        "scaling": "standard",
    }

    logs = _apply_llm_payloads([plan], [payload], raw_profile)

    assert plan.is_override is False
    assert plan.action == plan.rule_derived_action == "scale"
    assert plan.scaling == "standard"
    assert plan.justification == payload["override_reason"]
    assert not plan.justification.startswith("Final preprocessing plan for")
    assert logs == []
    print(json.dumps({
        "is_override": plan.is_override,
        "action": plan.action,
        "rule_derived_action": plan.rule_derived_action,
        "scaling": plan.scaling,
        "justification": plan.justification,
    }, sort_keys=True))


if __name__ == "__main__":
    test_kfold_target_encoder_runs_and_uses_inner_priors()
    test_kfold_target_encoder_rejects_degenerate_oof_input()
    test_build_pipeline_cross_validate_on_mini_dataset()
