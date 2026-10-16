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

-- "A KNN rejection on Ames Housing, verbatim identical in two runs, scored 1, 1, 3, 3 across the four AB runs."
WITH ames_knn AS (
  SELECT
    r.dataset,
    c.decision_key,
    COUNT(*) AS run_count,
    SORT_ARRAY(
      COLLECT_LIST(
        NAMED_STRUCT(
          'run_id', r.run_id,
          'layer2_score', CAST(c.layer2_score AS INT)
        )
      )
    ) AS score_by_run,
    ARRAY_SORT(COLLECT_LIST(CAST(c.layer2_score AS INT))) AS layer2_scores
  FROM silver_claims_enriched AS c
  INNER JOIN silver_runs AS r ON c.run_id = r.run_id
  WHERE r.dataset = 'ames_housing'
    AND r.arm = 'ab'
    AND c.decision_key = 'model_selection|knn|reject'
  GROUP BY r.dataset, c.decision_key
)
SELECT
  *,
  ARRAY(1, 1, 3, 3) AS expected_layer2_scores,
  layer2_scores = ARRAY(1, 1, 3, 3) AS matches_expected
FROM ames_knn;

-- "For contrast, Adult Income's KNN rejection scored 2, 3, 3, 3 across the
-- four AB runs."
WITH adult_income_knn AS (
  SELECT
    r.dataset,
    c.decision_key,
    COUNT(*) AS run_count,
    SORT_ARRAY(
      COLLECT_LIST(
        NAMED_STRUCT(
          'run_id', r.run_id,
          'layer2_score', CAST(c.layer2_score AS INT)
        )
      )
    ) AS score_by_run,
    ARRAY_SORT(COLLECT_LIST(CAST(c.layer2_score AS INT))) AS layer2_scores
  FROM silver_claims_enriched AS c
  INNER JOIN silver_runs AS r ON c.run_id = r.run_id
  WHERE r.dataset = 'adult_income'
    AND r.arm = 'ab'
    AND c.decision_key = 'model_selection|knn|reject'
  GROUP BY r.dataset, c.decision_key
)
SELECT
  *,
  ARRAY(2, 3, 3, 3) AS expected_layer2_scores,
  layer2_scores = ARRAY(2, 3, 3, 3) AS matches_expected
FROM adult_income_knn;

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
