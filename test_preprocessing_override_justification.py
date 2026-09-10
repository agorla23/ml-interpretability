"""Regression coverage for final-state preprocessing override justifications."""

from __future__ import annotations

from evidence_validator import EvidenceRef
from preprocessing_agent import ColumnPlan, _apply_llm_payloads


def test_approved_override_replaces_default_framed_justification_with_final_plan() -> None:
    raw_profile = {"n_rows": 100}
    plan = ColumnPlan(
        claim_id="prep_006",
        stage="preprocess",
        decision="scale free sulfur dioxide",
        justification="Standard scaling is applied per rule defaults.",
        evidence=[EvidenceRef("n_rows", 100)],
        confidence=1.0,
        column="free sulfur dioxide",
        action="scale",
        imputation="none",
        encoding="none",
        scaling="standard",
        rule_derived_action="scale",
        override_reason="",
        is_override=False,
    )
    payload = {
        "column": "free sulfur dioxide",
        "justification": "Standard scaling is applied per rule defaults.",
        "evidence": [{"metric_key": "n_rows", "claimed_value": 100}],
        "confidence": 0.9,
        "override_reason": "Robust scaling better handles the skewed distribution.",
        "imputation": "none",
        "encoding": "none",
        "scaling": "robust",
    }

    _apply_llm_payloads([plan], [payload], raw_profile)

    assert plan.is_override is True
    assert plan.scaling == "robust"
    assert plan.override_reason == "Robust scaling better handles the skewed distribution."
    assert plan.justification == (
        "Final preprocessing plan for 'free sulfur dioxide': action=scale, "
        "imputation=none, encoding=none, scaling=robust."
    )
    assert "standard scaling" not in plan.justification.lower()
