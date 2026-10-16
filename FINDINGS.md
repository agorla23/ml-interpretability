# Confirmed Findings and Fixes

This file is the review-oriented record for the benchmark matrix. Generated run artifacts are intentionally excluded from Git; for any individual run, the supporting records live in `runs/<run-id>/` as `claims.jsonl`, `layer1_verdicts.jsonl`, `layer2_verdicts.jsonl`, `prompts.jsonl`, and `metrics.json`.

## Pipeline Correctness

| Area | Confirmed finding | Resolution |
| --- | --- | --- |
| EDA severity mapping | Metric names in cited evidence could be treated as sufficient for escalation, even when their claimed values were false. | Severity matching was made value-sensitive: an imbalance or multicollinearity escalation now requires the corresponding true flag. |
| Numeric categorical preprocessing | A numeric-but-categorical column with missing values could reach `OneHotEncoder` as a mix of floats and the string missing sentinel. | These columns are stringified before constant-value imputation, preserving the categorical route while giving the encoder homogeneous values. |
| Crash diagnosability | A pipeline crash could leave partial artifacts without a clear record of the active stage or exception. | An unhandled-stage exception now writes an attempt-specific crash dump with partial prompts, claims, stage, and traceback before re-raising. |
| Wide preprocess requests | Ames' 81-column preprocess completion hit the output cap, yielding an empty payload and deterministic fallback claims with blank justification/confidence. | Preprocess planning is batched by column with per-chunk evidence, preserving claim identities and allowing normal LLM plans for wide datasets. |
| Fallback visibility | A successful-looking metrics file did not expose that a stage had fallen back to rule-derived behavior. | `metrics.json` now records `stages_fell_back_to_rule_default`, empty when no fallback occurred. |
| Override accounting | A non-empty `override_reason` could consume the global override cap and use override wording even when the proposed plan matched the rule default. | An override now means an actual transform difference; matching plans retain the model's explanation as ordinary justification. |

## Benchmark and Evaluation Observations

| Scope | Confirmed observation | Interpretation |
| --- | --- | --- |
| Layer 2 coverage | Layer 2 accepts claims that pass Layer 1's deterministic citation check. A Home Credit claim with `EVIDENCE_MISSTATED` was therefore absent from the Layer 2 verdict file. | This is an explicit evidence-quality gate for criticism coverage, not an LLM-critic decision that blocks downstream processing. |
| Synthetic Arm C | Arm C adds reproducible constant, target-leak, and high-missingness columns without modifying Arm A/B or historical artifacts. | It provides known ground truth for measuring detection and preprocessing behavior. |
| EDA coverage | Synthetic high-missingness columns were not always mentioned by EDA on the widest datasets, even when the deterministic preprocessing rule dropped them. | This is an EDA coverage miss, distinct from a preprocessing failure. |

## Corrections

The writeup attributed a repeated KNN-rejection pattern with Layer-2 scores
`1, 1, 3, 3` to Adult Income. Reproducing the finding from the raw run artifacts
through the lakehouse showed that the pattern belongs to **Ames Housing** across
its four AB runs. Under the Layer-2 rubric, score `1` means the evidence was
described honestly but the decision moved in the wrong direction; no KNN claim
has a recorded score of `0`. The corresponding Adult Income KNN rejection scored
`2, 3, 3, 3` across its four AB runs.

## How to Read a Run

- `claims.jsonl`: stage claims and their cited evidence.
- `layer1_verdicts.jsonl`: deterministic citation validation.
- `layer2_verdicts.jsonl`: LLM critic judgments for Layer 1-eligible claims.
- `prompts.jsonl`: issued prompts and recorded responses.
- `metrics.json`: run-level counts, scores, fallback state, and model/evaluation summaries.

The findings above are deliberately limited to items checked against those persisted artifacts and the corresponding code paths.
