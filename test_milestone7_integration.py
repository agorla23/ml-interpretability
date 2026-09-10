"""Executable no-network integration test for Milestone 7 selection and CV."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from evaluation_agent import evaluate_selected_models, run_metric_choice_agent
from evaluation_agent import persist_evaluation_results
from layer1_persistence import append_layer1_claims, persist_layer1_run, rollup_claim_verdicts
from layer2_critic import persist_layer2_run, run_layer2_critic
from model_selection_agent import run_model_selection_agent
from preprocessing_agent import derive_column_plan_defaults, run_preprocessing_agent
from profile_dataset import profile_dataset


MINI_DATASET = Path("/Users/akhilgorla/Downloads/mini_dataset.csv")


class FakeClient:
    def __init__(self, tool_name, payload):
        self.messages = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=tool_name, input=payload)]))


def _preprocessing_payload(profile, target):
    return {"plans": [{"column": default.column, "justification": "Use deterministic preprocessing.", "evidence": [{"metric_key": key, "claimed_value": profile[key]} for key in default.evidence_keys], "confidence": 1.0, "override_reason": "", "imputation": default.imputation, "encoding": default.encoding, "scaling": default.scaling} for default in derive_column_plan_defaults(profile, target)]}


def test_milestone7_selection_and_evaluation() -> None:
    df = pd.read_csv(MINI_DATASET)
    profile = profile_dataset(df, "target", "binary")
    plans, preprocessing = run_preprocessing_agent(profile, df, "target", client=FakeClient("record_preprocessing_plans", _preprocessing_payload(profile, "target")))
    assert not hasattr(preprocessing.named_steps["preprocess"], "transformers_")

    choices_payload = {"choices": [
        {"model_id": "logreg", "rank": 1, "is_rejection": False, "decision": "Select logistic regression.", "justification": "Small tabular data and numeric/categorical mix favor an interpretable linear baseline; cited correlations and multicollinearity guide this tradeoff.", "evidence": [{"metric_key": "n_rows", "claimed_value": profile["n_rows"]}, {"metric_key": "col:age:dtype", "claimed_value": profile["col:age:dtype"]}], "confidence": 0.8},
        {"model_id": "rf", "rank": 2, "is_rejection": False, "decision": "Select random forest.", "justification": "It can represent nonlinear signal while retaining a feature-importance tradeoff for the small mixed table.", "evidence": [{"metric_key": "n_rows", "claimed_value": profile["n_rows"]}, {"metric_key": "col:income:corr_with_target", "claimed_value": profile["col:income:corr_with_target"]}], "confidence": 0.7},
        {"model_id": "gbt", "rank": 3, "is_rejection": False, "decision": "Select GBT.", "justification": "It provides a nonlinear comparison for the mixed table despite the small-sample interpretability tradeoff.", "evidence": [{"metric_key": "n_rows", "claimed_value": profile["n_rows"]}, {"metric_key": "col:age:is_multicollinear", "claimed_value": profile["col:age:is_multicollinear"]}], "confidence": 0.5},
        {"model_id": "knn", "rank": None, "is_rejection": True, "decision": "Reject KNN.", "justification": "The encoded feature shape makes distance-based performance less compelling than the selected models.", "evidence": [{"metric_key": "n_rows", "claimed_value": profile["n_rows"]}, {"metric_key": "col:city:dtype", "claimed_value": profile["col:city:dtype"]}], "confidence": 0.7},
    ]}
    dropped = [plan.column for plan in plans if plan.action == "drop"]
    retained = [plan.column for plan in plans if plan.action != "drop"]
    choices = run_model_selection_agent(profile, "binary", retained_columns=retained, dropped_columns=dropped, client=FakeClient("record_model_choices", choices_payload))
    metric_payload = {"decision": "Use ROC-AUC.", "justification": "The observed 80/20 target balance supports the deterministic ROC-AUC choice.", "evidence": [{"metric_key": "target_class_balance", "claimed_value": profile["target_class_balance"]}], "confidence": 1.0}
    metric_choice = run_metric_choice_agent(profile, "binary", retained_columns=retained, dropped_columns=dropped, client=FakeClient("record_metric_justification", metric_payload))
    assert [reference.metric_key for reference in metric_choice.evidence] == ["target_class_balance"]
    assert metric_choice.evidence_verdicts[0].layer1_code == "EVIDENCE_ACCURATE"
    results, degraded = evaluate_selected_models(choices, preprocessing, df, "target", "binary", metric_choice.primary_metric, metric_choice.secondary_metrics)

    assert [result.model_id for result in results] == ["logreg", "rf", "gbt"]
    assert all(result.usable_folds == 4 for result in results)
    assert degraded is True
    assert all(not hasattr(preprocessing.named_steps["preprocess"], "transformers_") for _ in results)
    with tempfile.TemporaryDirectory(prefix="milestone7_") as temp_dir:
        _, run_dir, _ = persist_layer1_run(choices, severity_violation_rate=0.0, run_id="integration", runs_dir=temp_dir)
        metric_verdicts, _ = append_layer1_claims(run_dir, [metric_choice])
        metrics = persist_evaluation_results(run_dir, results, degraded)
        critic_payload = {"claim_id": "ignored", "layer2_score": 3, "discrepancy_note": "", "critic_confidence": 1.0}
        selection_layer2 = run_layer2_critic(
            rollup_claim_verdicts(choices), choices, profile,
            client=FakeClient("record_layer2_verdict", critic_payload),
        )
        persist_layer2_run(run_dir, selection_layer2)
        metric_layer2 = run_layer2_critic(metric_verdicts, [metric_choice], profile, client=FakeClient("record_layer2_verdict", critic_payload))
        metrics = persist_layer2_run(run_dir, metric_layer2, append=True)
        assert metrics["models_evaluated"] == 3
        assert metrics["stratification_degraded"] is True
        assert metrics["mean_layer2_score_by_stage"] == {"evaluation": 3.0, "model_selection": 3.0}
        assert len((run_dir / "evaluation_results.jsonl").read_text().splitlines()) == 3
    print("selected_models=" + str([result.model_id for result in results]))
    print("primary_metric=" + metric_choice.primary_metric)
    print("evaluation_evidence_keys=" + str([reference.metric_key for reference in metric_choice.evidence]))
    print("evaluation_evidence_verdicts=" + str([verdict.layer1_code for verdict in metric_choice.evidence_verdicts]))
    print("usable_folds=" + str([result.usable_folds for result in results]))
    print("stratification_degraded=" + str(degraded))


if __name__ == "__main__":
    test_milestone7_selection_and_evaluation()
