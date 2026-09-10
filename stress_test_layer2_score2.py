"""Stress test for the Layer-2 critic's score-2 bucket on the 0-3 scale.

Context: scores 0, 1, and 3 have all been observed in real output. Score 2,
"right direction, but incomplete OR thinly evidenced," needs a focused test
before ``mean_layer2_score`` is treated as a fine-grained metric.

Run from the project directory with ANTHROPIC_API_KEY set:

    python3 stress_test_layer2_score2.py
"""

import os
import sys
from pathlib import Path

from evidence_validator import Claim, EvidenceRef
from layer1_persistence import ClaimVerdict
from layer2_critic import run_layer2_critic


# Real values from the actual mini_dataset.csv profile.
RAW_PROFILE_SUBSET = {
    "col:age:missing_pct": 15.0,
    "col:age:has_moderate_missing": True,
    "col:age:missingness_informative": True,
    "col:age:missingness_cluster_corr_max": 1.0,
    "col:income:is_skewed": True,
    "col:income:skew": 2.95,
    "col:income:min": 39000.0,
    "col:income:max": 300000.0,
    "col:days_to_followup:leakage_suspect": True,
    "col:days_to_followup:corr_with_target": 0.997,
    "col:account_status:is_constant": True,
    "col:account_status:nunique": 1,
}


STRESS_CLAIMS = [
    Claim(
        claim_id="stress2_001",
        stage="eda",
        decision="Impute missing values in 'age' using the median",
        justification=(
            "The 'age' column has 15% missing values, which is moderate. The "
            "missingness is also flagged as informative. Median imputation "
            "will fill the gaps so the column can be retained for modeling."
        ),
        evidence=[
            EvidenceRef(metric_key="col:age:missing_pct", claimed_value=15.0),
            EvidenceRef(metric_key="col:age:has_moderate_missing", claimed_value=True),
            EvidenceRef(metric_key="col:age:missingness_informative", claimed_value=True),
        ],
        confidence=0.85,
    ),
    Claim(
        claim_id="stress2_002",
        stage="eda",
        decision="Flag 'days_to_followup' for manual review before modeling",
        justification=(
            "The 'days_to_followup' column is flagged as a leakage suspect "
            "with a 0.997 correlation to the target. This should be noted "
            "and reviewed by the modeling team before the dataset is used."
        ),
        evidence=[
            EvidenceRef(metric_key="col:days_to_followup:leakage_suspect", claimed_value=True),
            EvidenceRef(metric_key="col:days_to_followup:corr_with_target", claimed_value=0.997),
        ],
        confidence=0.8,
    ),
    Claim(
        claim_id="stress2_003",
        stage="eda",
        decision="Apply a log transformation to the 'income' column",
        justification=(
            "The 'income' column ranges from 39,000 to 300,000, a very wide "
            "spread. A log transformation will compress this range and "
            "improve model performance."
        ),
        evidence=[
            EvidenceRef(metric_key="col:income:min", claimed_value=39000.0),
            EvidenceRef(metric_key="col:income:max", claimed_value=300000.0),
        ],
        confidence=0.75,
    ),
    Claim(
        claim_id="stress2_004",
        stage="eda",
        decision="Drop the constant column 'account_status'",
        justification=(
            "The 'account_status' column is constant, with only 1 unique "
            "value across all rows. A column with no variation carries no "
            "information for any model and should be removed."
        ),
        evidence=[
            EvidenceRef(metric_key="col:account_status:is_constant", claimed_value=True),
            EvidenceRef(metric_key="col:account_status:nunique", claimed_value=1),
        ],
        confidence=0.95,
    ),
]


FAKE_CLAIM_VERDICTS = [
    ClaimVerdict(
        claim_id=claim.claim_id,
        stage=claim.stage,
        citation_layer1_codes=["EVIDENCE_ACCURATE"] * len(claim.evidence),
        claim_layer1_code="EVIDENCE_ACCURATE",
    )
    for claim in STRESS_CLAIMS
]


EXPECTED = {
    "stress2_001": (2, "2a incomplete: cites informative missingness, no indicator column"),
    "stress2_002": (2, "2a incomplete: 'flag for review' falls short of removal"),
    "stress2_003": (2, "2b thin: min/max range does not establish skew"),
    "stress2_004": (3, "control: right, complete, directly evidenced"),
}


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

    lines = ["Score-2 bucket stress test (0-3 scale)", ""]
    hits = 0
    for claim, verdict in zip(STRESS_CLAIMS, verdicts):
        expected_score, rationale = EXPECTED[claim.claim_id]
        match = "MATCH" if verdict.layer2_score == expected_score else "MISS"
        if verdict.layer2_score == expected_score:
            hits += 1
        lines.extend(
            [
                f"{claim.claim_id}  [{match}]",
                f"  decision:  {claim.decision}",
                f"  expected:  {expected_score}  ({rationale})",
                f"  actual:    {verdict.layer2_score}",
                f"  note:      {verdict.discrepancy_note}",
                f"  conf:      {verdict.critic_confidence}",
                "",
            ]
        )

    lines.extend(
        [
            f"Matched expectation on {hits}/{len(STRESS_CLAIMS)} cases.",
            "",
            "How to read this:",
            "- If the three 2-cases all score 2 and the control scores 3, the",
            "  score-2 bucket is reachable and the full 0-3 scale is verified.",
            "- If the 2-cases score 3, the critic is too lenient at the 2/3",
            "  boundary -- it treats 'right direction' as sufficient for top",
            "  marks and ignores completeness/evidence-quality.",
            "- If the 2-cases score 1, the critic is collapsing 'incomplete'",
            "  into 'wrong direction', same failure the 0/1 split was meant to",
            "  fix, one level up.",
            "- If the CONTROL scores below 3, the rubric change made the critic",
            "  over-strict and none of the other results are trustworthy.",
        ]
    )
    output = "\n".join(lines)
    print(output)
    Path("stress_test_layer2_score2_output.txt").write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
