from eda_agent import _severity_from_evidence
from evidence_validator import EvidenceRef


def _ref(metric: str, value) -> EvidenceRef:
    prefix = "target:" if metric == "target:is_imbalanced" else "col:feature:"
    metric_key = metric if metric == "target:is_imbalanced" else prefix + metric
    return EvidenceRef(metric_key=metric_key, claimed_value=value)


def test_false_risk_flags_do_not_escalate_severity() -> None:
    assert _severity_from_evidence([_ref("target:is_imbalanced", False)]) == "info"
    assert _severity_from_evidence([_ref("leakage_suspect", False)]) == "info"
    assert _severity_from_evidence([_ref("is_constant", False)]) == "info"
    assert _severity_from_evidence([_ref("has_high_missing", False)]) == "info"
    assert _severity_from_evidence([_ref("is_multicollinear", False)]) == "info"
    assert _severity_from_evidence([_ref("has_moderate_missing", False)]) == "info"
    assert _severity_from_evidence([_ref("is_skewed", False)]) == "info"
    assert _severity_from_evidence([_ref("missingness_informative", False)]) == "info"


def test_true_risk_flags_preserve_tier_precedence() -> None:
    assert _severity_from_evidence([_ref("is_skewed", True)]) == "warn"
    assert _severity_from_evidence([_ref("target:is_imbalanced", True)]) == "critical"
    assert _severity_from_evidence(
        [_ref("is_skewed", True), _ref("leakage_suspect", True)]
    ) == "critical"


def test_false_critical_flag_does_not_mask_true_warning_flag() -> None:
    assert _severity_from_evidence(
        [_ref("has_high_missing", False), _ref("has_moderate_missing", True)]
    ) == "warn"
