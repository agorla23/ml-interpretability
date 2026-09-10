"""Regression tests for scalar, None, boolean, and dict evidence validation."""

from evidence_validator import Claim, EvidenceRef, validate_evidence


def _claim(claimed_value, metric_key="target_class_balance"):
    return Claim("test", "evaluation", "choose metric", "profile evidence", [EvidenceRef(metric_key, claimed_value)], 1.0)


def test_dict_evidence_accurate_with_json_string_keys():
    verdict = validate_evidence(_claim({"0": 0.8, "1": 0.2}), {"target_class_balance": {0: 0.8, 1: 0.2}})[0]
    assert verdict.layer1_code == "EVIDENCE_ACCURATE"


def test_dict_evidence_inaccurate_per_key_value():
    verdict = validate_evidence(_claim({"0": 0.8, "1": 0.3}), {"target_class_balance": {0: 0.8, 1: 0.2}})[0]
    assert verdict.layer1_code == "EVIDENCE_MISSTATED"


def test_explicit_dict_subkey_uses_scalar_tolerance():
    verdict = validate_evidence(_claim(0.805, "target_class_balance[0]"), {"target_class_balance": {0: 0.8, 1: 0.2}})[0]
    assert verdict.layer1_code == "EVIDENCE_ACCURATE"


def test_scalar_boolean_and_none_rules_are_unchanged():
    raw = {"numeric": 100.0, "boolean": False, "missing": None}
    claim = Claim("scalar", "test", "test", "test", [EvidenceRef("numeric", 101.0), EvidenceRef("boolean", False), EvidenceRef("missing", 1.0)], 1.0)
    assert [item.layer1_code for item in validate_evidence(claim, raw)] == ["EVIDENCE_ACCURATE", "EVIDENCE_ACCURATE", "FABRICATED_EVIDENCE"]
