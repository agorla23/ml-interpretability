"""Executable no-network integration test for the complete Milestone 6 flow."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from eda_agent import Finding
from evidence_integration import attach_evidence_verdicts_to_claims
from evidence_validator import EvidenceRef
from layer1_persistence import persist_layer1_run, rollup_claim_verdicts
from layer2_critic import persist_layer2_run, run_layer2_critic
from preprocessing_agent import derive_column_plan_defaults, run_preprocessing_agent
from preprocessing_persistence import persist_preprocessing_run
from profile_dataset import profile_dataset


MINI_DATASET = Path("/Users/akhilgorla/Downloads/mini_dataset.csv")


class FakeMessages:
    def __init__(self, responses):
        self.responses = iter(responses)

    def create(self, **kwargs):
        tool_name, payload = next(self.responses)
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name=tool_name, input=payload)]
        )


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def _preprocessing_client(raw_profile: dict, target_column: str) -> FakeClient:
    defaults = derive_column_plan_defaults(raw_profile, target_column)
    payloads = []
    for default in defaults:
        payloads.append(
            {
                "column": default.column,
                "justification": (
                    f"The deterministic {default.action} plan follows the cited profile evidence."
                ),
                "evidence": [
                    {"metric_key": key, "claimed_value": raw_profile[key]}
                    for key in default.evidence_keys
                ],
                "confidence": 1.0,
                "override_reason": "",
                "imputation": default.imputation,
                "encoding": default.encoding,
                "scaling": default.scaling,
            }
        )
    return FakeClient([("record_preprocessing_plans", {"plans": payloads})])


def _critic_client(call_count: int) -> FakeClient:
    return FakeClient(
        [
            (
                "record_layer2_verdict",
                {
                    "claim_id": "ignored_by_adapter",
                    "layer2_score": 3,
                    "discrepancy_note": "",
                    "critic_confidence": 1.0,
                },
            )
            for _ in range(call_count)
        ]
    )


def test_complete_milestone6_flow() -> None:
    df = pd.read_csv(MINI_DATASET)
    raw_profile = profile_dataset(df, target_column="target", task_type="binary")

    eda_finding = Finding(
        claim_id="eda_001",
        stage="eda",
        decision="Inspect dataset dimensions.",
        justification="The profile reports the row count.",
        evidence=[EvidenceRef("n_rows", raw_profile["n_rows"])],
        confidence=1.0,
        severity="info",
        llm_stated_severity="info",
        rule_derived_severity="info",
        severity_rule_violation=False,
    )
    attach_evidence_verdicts_to_claims([eda_finding], raw_profile)

    with tempfile.TemporaryDirectory(prefix="milestone6_") as temp_dir:
        _, run_dir, _ = persist_layer1_run(
            [eda_finding],
            severity_violation_rate=0.0,
            run_id="integration",
            runs_dir=temp_dir,
        )
        plans, pipeline = run_preprocessing_agent(
            raw_profile,
            df,
            "target",
            client=_preprocessing_client(raw_profile, "target"),
        )
        preprocessing_claim_verdicts, metrics = persist_preprocessing_run(run_dir, plans)

        plans_by_column = {plan.column: plan for plan in plans}
        assert "target" not in plans_by_column
        assert plans_by_column["days_to_followup"].action == "drop"
        assert plans_by_column["account_status"].action == "drop"
        assert plans_by_column["customer_id"].action == "drop"
        assert plans_by_column["age_duplicate"].action == "drop"
        assert plans_by_column["age"].action != "drop"
        # age missingness correlates only 0.14 with the target; the old
        # constant-imputation expectation came from cross-column clustering.
        assert plans_by_column["age"].imputation == "mean"
        assert plans_by_column["region_code"].encoding == "onehot"
        assert plans_by_column["region_code"].scaling == "none"
        assert plans_by_column["satisfaction_score"].encoding == "ordinal"
        assert plans_by_column["satisfaction_score"].scaling == "none"
        assert plans_by_column["income"].scaling == "robust"

        column_transformer = pipeline.named_steps["preprocess"]
        assert not hasattr(column_transformer, "transformers_")
        assert metrics["total_claims"] == 1 + len(plans)
        assert metrics["preprocess_claims"] == len(plans)
        assert metrics["columns_dropped"] == 4
        assert metrics["columns_kept"] == 6

        eda_claim_verdicts = rollup_claim_verdicts([eda_finding])
        eda_layer2 = run_layer2_critic(
            eda_claim_verdicts,
            [eda_finding],
            raw_profile,
            client=_critic_client(len(eda_claim_verdicts)),
        )
        persist_layer2_run(run_dir, eda_layer2)
        preprocessing_layer2 = run_layer2_critic(
            preprocessing_claim_verdicts,
            plans,
            raw_profile,
            client=_critic_client(len(preprocessing_claim_verdicts)),
        )
        metrics = persist_layer2_run(run_dir, preprocessing_layer2, append=True)

        assert metrics["layer2_claims_scored"] == 1 + len(plans)
        assert metrics["mean_layer2_score_by_stage"] == {
            "eda": 3.0,
            "preprocess": 3.0,
        }
        assert len((run_dir / "column_plans.jsonl").read_text().splitlines()) == len(plans)
        assert len((run_dir / "claim_verdicts.jsonl").read_text().splitlines()) == 1 + len(plans)
        assert len((run_dir / "layer2_verdicts.jsonl").read_text().splitlines()) == 1 + len(plans)

        print("column_plans:")
        for plan in plans:
            print(
                f"{plan.column}: action={plan.action} imputation={plan.imputation} "
                f"encoding={plan.encoding} scaling={plan.scaling} "
                f"rule_derived_action={plan.rule_derived_action} "
                f"is_override={plan.is_override} override_reason={plan.override_reason!r}"
            )
        print(f"pipeline={pipeline!r}")
        print("column_transformer_has_transformers_=False")
        print("metrics=" + json.dumps(metrics, sort_keys=True))
        print("preprocessing_layer2_count=" + str(len(preprocessing_layer2)))
        print("milestone6_integration_passed=True")


if __name__ == "__main__":
    test_complete_milestone6_flow()
