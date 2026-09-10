import copy
import json
from pathlib import Path
from types import SimpleNamespace

from rescore_prep_override_fix import rescore_historical_overrides


class _FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["messages"][0]["content"])
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="record_layer2_verdict",
                    input={
                        "claim_id": "ignored_by_critic_parser",
                        "layer2_score": 3,
                        "discrepancy_note": "",
                        "critic_confidence": 0.9,
                    },
                )
            ]
        )


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_rescore_changes_only_justification_and_keeps_critic_deterministic(tmp_path):
    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    plan = {
        "claim_id": "prep_006",
        "stage": "preprocess",
        "decision": "scale feature",
        "justification": "Standard scaling is applied.",
        "evidence": [{"metric_key": "col:feature:is_skewed", "claimed_value": True}],
        "confidence": 0.8,
        "column": "feature",
        "action": "scale",
        "imputation": "none",
        "encoding": "none",
        "scaling": "robust",
        "rule_derived_action": "scale",
        "override_reason": "Skew requires robust scaling.",
        "is_override": True,
        "evidence_verdicts": [
            {
                "claim_id": "prep_006",
                "metric_key": "col:feature:is_skewed",
                "layer1_code": "EVIDENCE_ACCURATE",
            }
        ],
    }
    original_plan = copy.deepcopy(plan)
    _write_jsonl(run_dir / "column_plans.jsonl", [plan])
    _write_jsonl(
        run_dir / "layer2_verdicts.jsonl",
        [{"claim_id": "prep_006", "layer2_score": 1}],
    )
    (run_dir / "state.json").write_text(
        json.dumps({"raw_profile": {"col:feature:is_skewed": True}}),
        encoding="utf-8",
    )
    output = tmp_path / "rescore.jsonl"
    fake_client = _FakeClient()

    results = rescore_historical_overrides(
        run_ids=["run_a"],
        claim_ids=["prep_006"],
        runs_root=tmp_path / "runs",
        output_path=output,
        client=fake_client,
        model="fake-model",
    )

    assert json.loads((run_dir / "column_plans.jsonl").read_text()) == original_plan
    assert results[0]["old_justification"] == "Standard scaling is applied."
    assert results[0]["new_justification"].endswith("scaling=robust.")
    assert results[0]["old_layer2_score"] == 1
    assert results[0]["new_layer2_score"] == 3
    assert json.loads(output.read_text()) == results[0]
    assert fake_client.messages.calls[0]["temperature"] == 0
    isolated_payload = json.loads(fake_client.messages.calls[0]["messages"][0]["content"])
    assert isolated_payload["evidence_profile_subset"] == {"col:feature:is_skewed": True}
