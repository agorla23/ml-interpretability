"""Stress test for the Layer-2 critic.

Purpose: the real EDA run scored 8/8 claims as a perfect 2.0. Before treating
that as evidence of a lenient/broken critic, this script checks whether the
critic is capable of giving a lower score at all. It hand-builds claims with
real, accurate evidence but decisions that overreach or do not actually follow
from that evidence, bypassing the EDA agent entirely so the critic is tested in
isolation.

Run from the project directory with ANTHROPIC_API_KEY set:

    python3 stress_test_layer2_critic.py
"""

import os
import sys
from pathlib import Path

from evidence_validator import Claim, EvidenceRef
from layer1_persistence import ClaimVerdict
from layer2_critic import run_layer2_critic


# Real values pulled from the mini_dataset.csv profile output. These are
# genuinely accurate, so any low score below is a Layer-2 reasoning judgment,
# not a Layer-1 fabrication/misstatement issue.
RAW_PROFILE_SUBSET = {
    "col:income:is_skewed": True,
    "col:income:skew": 2.95,
    "col:account_status:is_constant": True,
    "col:account_status:nunique": 1,
    "col:customer_id:corr_with_target": 0.087,
    "col:customer_id:is_id_like": True,
    "col:age:has_moderate_missing": True,
    "col:age:missing_pct": 15.0,
    "col:age:missingness_informative": True,
}


STRESS_CLAIMS = [
    Claim(
        claim_id="stress_001",
        stage="eda",
        decision="Drop the 'income' column entirely",
        justification=(
            "Income is highly skewed (skew=2.95), so it should be removed "
            "from the dataset before modeling."
        ),
        evidence=[
            EvidenceRef(metric_key="col:income:is_skewed", claimed_value=True),
            EvidenceRef(metric_key="col:income:skew", claimed_value=2.95),
        ],
        confidence=0.9,
    ),
    Claim(
        claim_id="stress_002",
        stage="eda",
        decision="Impute missing values in 'account_status' using KNN",
        justification=(
            "The column 'account_status' is constant, with only 1 unique "
            "value across all records, so KNN imputation should be used to "
            "fill in any gaps."
        ),
        evidence=[
            EvidenceRef(metric_key="col:account_status:is_constant", claimed_value=True),
            EvidenceRef(metric_key="col:account_status:nunique", claimed_value=1),
        ],
        confidence=0.85,
    ),
    Claim(
        claim_id="stress_003",
        stage="eda",
        decision="Prioritize 'customer_id' as the most important predictive feature",
        justification=(
            "customer_id shows a correlation of 0.087 with the target, "
            "indicating it is a strong and reliable predictor that should "
            "be prioritized in feature selection."
        ),
        evidence=[
            EvidenceRef(metric_key="col:customer_id:corr_with_target", claimed_value=0.087),
            EvidenceRef(metric_key="col:customer_id:is_id_like", claimed_value=True),
        ],
        confidence=0.8,
    ),
    Claim(
        claim_id="stress_004",
        stage="eda",
        decision="Drop the 'age' column entirely due to missing data",
        justification=(
            "Age has 15% missing values, so it should be removed from the "
            "dataset."
        ),
        evidence=[
            EvidenceRef(metric_key="col:age:has_moderate_missing", claimed_value=True),
            EvidenceRef(metric_key="col:age:missing_pct", claimed_value=15.0),
        ],
        confidence=0.75,
    ),
]


# All claims are treated as Layer-1 accurate: this tests Layer 2 in isolation.
FAKE_CLAIM_VERDICTS = [
    ClaimVerdict(
        claim_id=claim.claim_id,
        stage=claim.stage,
        citation_layer1_codes=["EVIDENCE_ACCURATE"] * len(claim.evidence),
        claim_layer1_code="EVIDENCE_ACCURATE",
    )
    for claim in STRESS_CLAIMS
]


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set in this shell. Run:\n"
            "  export ANTHROPIC_API_KEY='your_real_key_here'",
            file=sys.stderr,
        )
        return 2

    verdicts = run_layer2_critic(
        FAKE_CLAIM_VERDICTS,
        STRESS_CLAIMS,
        RAW_PROFILE_SUBSET,
    )

    lines = ["Stress test results (real, accurate evidence + deliberately bad decisions):", ""]
    for claim, verdict in zip(STRESS_CLAIMS, verdicts):
        lines.extend(
            [
                f"{claim.claim_id}",
                f"  decision:     {claim.decision}",
                f"  score:        {verdict.layer2_score}",
                f"  note:         {verdict.discrepancy_note}",
                f"  confidence:   {verdict.critic_confidence}",
                "",
            ]
        )

    mean = sum(v.layer2_score for v in verdicts) / len(verdicts)
    lines.append(f"Mean stress-test score: {mean:.2f}")
    lines.append(
        "If this comes back near 2.0 (same as the real run), the critic "
        "likely IS too lenient. If it comes back meaningfully lower and "
        "varies across the 4 cases, the critic is discriminating correctly "
        "and the earlier 2.0/2.0 result was legitimate given how clean the "
        "mini dataset's real findings were."
    )
    output = "\n".join(lines)
    print(output)
    Path("stress_test_layer2_output.txt").write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
