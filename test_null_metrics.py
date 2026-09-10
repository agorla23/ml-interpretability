"""Empty aggregate metrics must serialize as null, never a fabricated zero."""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from layer1_persistence import ClaimVerdict, compute_layer1_metrics
from layer2_critic import compute_layer2_metrics, persist_layer2_run


def test_empty_layer1_rates_are_none():
    metrics = compute_layer1_metrics([], [], None)
    assert metrics["citation_fabrication_rate"] is None
    assert metrics["citation_misstatement_rate"] is None
    assert metrics["claim_fabrication_rate"] is None
    assert metrics["claim_misstatement_rate"] is None


def test_unscored_stage_and_overall_layer2_mean_serialize_as_null():
    fabricated = ClaimVerdict("evaluate_001", "evaluation", ["FABRICATED_EVIDENCE"], "FABRICATED_EVIDENCE")
    with tempfile.TemporaryDirectory(prefix="empty_layer2_") as temp_dir:
        run_dir = Path(temp_dir)
        (run_dir / "claim_verdicts.jsonl").write_text(json.dumps(asdict(fabricated)) + "\n", encoding="utf-8")
        (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
        metrics = persist_layer2_run(run_dir, [])
        rendered = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["mean_layer2_score"] is None
        assert metrics["mean_layer2_score_by_stage"]["evaluation"] is None
        assert rendered["mean_layer2_score"] is None
        assert rendered["mean_layer2_score_by_stage"]["evaluation"] is None
        print(json.dumps(rendered, indent=2, sort_keys=True))
