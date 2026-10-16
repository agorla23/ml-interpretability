USE CATALOG workspace;
USE SCHEMA mlinterp;

-- "The planted-defect benchmark caught 13 of 15 defects; the two misses were
-- the high-missingness columns in ames_housing and home_credit."
SELECT
  dataset,
  defect,
  planted_count,
  detected_count,
  missed_columns,
  SUM(detected_count) OVER () AS all_detected,
  SUM(planted_count) OVER () AS all_planted
FROM gold_arm_c_detection
ORDER BY dataset, defect;

-- "Adult Income rejected KNN with Layer-2 scores 1, 1, 3, 3 across the
-- repeated temperature-0.0 benchmark runs."
SELECT
  dataset,
  decision_key,
  run_count,
  score_by_run,
  layer2_scores,
  score_spread
FROM gold_decision_consistency
WHERE dataset = 'adult_income'
  AND decision_key = 'model_selection|knn|reject';

-- "The Layer-2 coverage gate excluded exactly two EVIDENCE_MISSTATED claims
-- and scored all 1,625 Layer-1-eligible claims."
SELECT
  SUM(layer1_eligible_count) AS layer1_eligible_claims,
  SUM(layer2_claim_count) AS layer2_verdicts,
  SUM(eligible_missing_layer2_count) AS eligible_without_layer2,
  SUM(ineligible_with_layer2_count) AS ineligible_with_layer2,
  SUM(evidence_misstated_excluded_count) AS evidence_misstated_excluded
FROM gold_layer2_scores;

-- "Home Credit achieved PR-AUC about 0.22 against a positive-class base rate
-- of about 0.08."
SELECT
  run_id,
  selected_model,
  primary_metric,
  primary_metric_value,
  positive_class_base_rate,
  primary_metric_value - positive_class_base_rate AS absolute_lift
FROM gold_model_metrics
WHERE dataset = 'home_credit'
ORDER BY run_id;

-- "Mean Layer-2 score varies by stage and temperature."
SELECT
  stage,
  temperature,
  SUM(layer2_claim_count) AS scored_claims,
  SUM(mean_layer2_score * layer2_claim_count) / SUM(layer2_claim_count)
    AS mean_layer2_score,
  SUM(score_0_count) AS score_0_count,
  SUM(score_1_count) AS score_1_count,
  SUM(score_2_count) AS score_2_count,
  SUM(score_3_count) AS score_3_count
FROM gold_layer2_scores
GROUP BY stage, temperature
ORDER BY stage, temperature;
