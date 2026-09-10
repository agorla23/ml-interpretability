from dataclasses import dataclass
import re
from typing import Any, Literal
 
# ---------------------------------------------------------------------------
# Core shapes
# ---------------------------------------------------------------------------
 
@dataclass
class EvidenceRef:
    metric_key: str
    claimed_value: float | str | bool | dict | None
 
 
@dataclass
class Claim:
    claim_id: str
    stage: str
    decision: str
    justification: str
    evidence: list[EvidenceRef]
    confidence: float
 
 
@dataclass
class EvidenceVerdict:
    claim_id: str
    metric_key: str
    layer1_code: Literal[
        "FABRICATED_EVIDENCE", "EVIDENCE_ACCURATE", "EVIDENCE_MISSTATED"
    ]
 
 
# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
 
RELATIVE_ERROR_TOLERANCE = 0.01  # 1%
 
 
def _is_number(value) -> bool:
    # bool is technically a subclass of int in Python, so explicitly exclude it
    # here -- otherwise True/False would be treated as 1/0 and compared
    # numerically instead of by exact equality, which is the wrong behavior.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scalar_values_match(real_value, claimed_value) -> bool:
    if _is_number(real_value) and _is_number(claimed_value):
        denominator = abs(real_value) if real_value != 0 else 1e-8
        relative_error = abs(claimed_value - real_value) / denominator
        return relative_error <= RELATIVE_ERROR_TOLERANCE
    return claimed_value == real_value


def _matching_dict_key(mapping: dict, key_text: str):
    matches = [key for key in mapping if str(key) == key_text]
    return matches[0] if len(matches) == 1 else None


def _dict_values_match(real_value: dict, claimed_value) -> bool:
    if not isinstance(claimed_value, dict) or len(real_value) != len(claimed_value):
        return False
    for real_key, real_item in real_value.items():
        claimed_key = _matching_dict_key(claimed_value, str(real_key))
        if claimed_key is None or not _scalar_values_match(real_item, claimed_value[claimed_key]):
            return False
    return True


def _resolve_metric_value(metric_key: str, raw_profile: dict) -> tuple[bool, Any]:
    if metric_key in raw_profile:
        return True, raw_profile[metric_key]

    # Explicit dict sub-key convention: target_class_balance[0]. JSON object
    # keys are strings, so matching uses their string representation.
    match = re.fullmatch(r"(.+)\[([^\[\]]+)\]", metric_key)
    if match is None:
        return False, None
    parent_key, child_key_text = match.groups()
    parent_value = raw_profile.get(parent_key)
    if not isinstance(parent_value, dict):
        return False, None
    child_key = _matching_dict_key(parent_value, child_key_text)
    if child_key is None:
        return False, None
    return True, parent_value[child_key]


def resolve_evidence_value(metric_key: str, raw_profile: dict) -> Any:
    """Resolve a direct profile key or documented ``dict_key[subkey]`` key."""
    exists, value = _resolve_metric_value(metric_key, raw_profile)
    if not exists:
        raise KeyError(metric_key)
    return value
 
 
def validate_evidence(claim: Claim, raw_profile: dict) -> list[EvidenceVerdict]:
    """
    Checks every EvidenceRef in `claim` against `raw_profile`.
 
    Rules:
      1. metric_key not present in raw_profile at all -> FABRICATED_EVIDENCE
      2. metric_key present, real value is None, claimed value is not None
         -> FABRICATED_EVIDENCE. The metric genuinely doesn't exist (e.g.
         vif for a column excluded from the VIF matrix) -- there is no true
         number to be "off" from, so a confidently stated value here is
         invented, not merely inaccurate. Treated the same severity as
         rule 1, not as a misstatement.
      2b. metric_key present, real value is None, claimed value is also
         None -> EVIDENCE_ACCURATE (agent correctly reported "no value").
      3. metric_key present, both real and claimed values are numeric
         -> compare relative error; within tolerance -> EVIDENCE_ACCURATE,
         else -> EVIDENCE_MISSTATED
      4. metric_key present, non-numeric comparison (str/bool) -> exact
         equality check -> EVIDENCE_ACCURATE or EVIDENCE_MISSTATED
      5. metric_key value is a dict -> claimed_value must be a dict with the
         same keys and per-key values within the existing 1% numeric tolerance.
         A single dict item may instead be cited with an explicit sub-key such
         as ``target_class_balance[0]``; prose-only dict descriptions are not
         accepted as evidence.
    """
    results: list[EvidenceVerdict] = []
 
    for ref in claim.evidence:
        # Rule 1: key must exist at all
        metric_exists, real_value = _resolve_metric_value(ref.metric_key, raw_profile)
        if not metric_exists:
            results.append(
                EvidenceVerdict(
                    claim_id=claim.claim_id,
                    metric_key=ref.metric_key,
                    layer1_code="FABRICATED_EVIDENCE",
                )
            )
            continue
 
        # Rule 2 / 2b: real value is None
        if real_value is None:
            if ref.claimed_value is None:
                code = "EVIDENCE_ACCURATE"
            else:
                # Agent invented a concrete value for a metric that has no
                # true value at all -- treated as fabrication, not a
                # garden-variety numeric miss.
                code = "FABRICATED_EVIDENCE"
            results.append(
                EvidenceVerdict(
                    claim_id=claim.claim_id,
                    metric_key=ref.metric_key,
                    layer1_code=code,
                )
            )
            continue

        # Rule 5: dictionary evidence is checked key-by-key without changing
        # the scalar exact-match and numeric-tolerance rules below.
        if isinstance(real_value, dict):
            code = "EVIDENCE_ACCURATE" if _dict_values_match(real_value, ref.claimed_value) else "EVIDENCE_MISSTATED"
            results.append(EvidenceVerdict(claim.claim_id, ref.metric_key, code))
            continue
 
        # Rule 3: both numeric -> relative error comparison
        if _is_number(real_value) and _is_number(ref.claimed_value):
            code = (
                "EVIDENCE_ACCURATE"
                if _scalar_values_match(real_value, ref.claimed_value)
                else "EVIDENCE_MISSTATED"
            )
            results.append(
                EvidenceVerdict(
                    claim_id=claim.claim_id,
                    metric_key=ref.metric_key,
                    layer1_code=code,
                )
            )
            continue
 
        # Rule 4: non-numeric (string / bool) -> exact match
        code = (
            "EVIDENCE_ACCURATE"
            if ref.claimed_value == real_value
            else "EVIDENCE_MISSTATED"
        )
        results.append(
            EvidenceVerdict(
                claim_id=claim.claim_id,
                metric_key=ref.metric_key,
                layer1_code=code,
            )
        )
 
    return results
 
 
# ---------------------------------------------------------------------------
# The 3 test cases from the spec discussion
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    raw_profile = {
        "col:age:missing_pct": 15.0,
        "col:age:is_skewed": False,
        "col:target:vif": None,
    }
 
    claim_1 = Claim(
        claim_id="test_001", stage="eda",
        decision="drop column X", justification="mostly empty",
        evidence=[EvidenceRef(metric_key="col:X:missing_pct", claimed_value=90.0)],
        confidence=0.9,
    )
 
    claim_2 = Claim(
        claim_id="test_002", stage="eda",
        decision="impute age", justification="15% missing, use median",
        evidence=[EvidenceRef(metric_key="col:age:missing_pct", claimed_value=15.0)],
        confidence=0.8,
    )
 
    claim_3 = Claim(
        claim_id="test_003", stage="eda",
        decision="drop age", justification="40% missing, too sparse",
        evidence=[EvidenceRef(metric_key="col:age:missing_pct", claimed_value=40.0)],
        confidence=0.7,
    )
 
    # extra edge cases worth checking beyond the original 3
    claim_4_bool_match = Claim(
        claim_id="test_004", stage="eda",
        decision="keep age as continuous", justification="not skewed",
        evidence=[EvidenceRef(metric_key="col:age:is_skewed", claimed_value=False)],
        confidence=0.6,
    )
 
    claim_5_none_but_claimed_number = Claim(
        claim_id="test_005", stage="eda",
        decision="flag target vif", justification="vif is 12.3, multicollinear",
        evidence=[EvidenceRef(metric_key="col:target:vif", claimed_value=12.3)],
        confidence=0.5,
    )
    # expected now: FABRICATED_EVIDENCE (was EVIDENCE_MISSTATED before the fix)
 
    claim_6_none_correctly_reported = Claim(
        claim_id="test_006", stage="eda",
        decision="skip vif check for target", justification="vif not applicable",
        evidence=[EvidenceRef(metric_key="col:target:vif", claimed_value=None)],
        confidence=0.9,
    )
    # expected: EVIDENCE_ACCURATE
 
    for claim in [
        claim_1, claim_2, claim_3, claim_4_bool_match,
        claim_5_none_but_claimed_number, claim_6_none_correctly_reported,
    ]:
        for verdict in validate_evidence(claim, raw_profile):
            print(verdict)
