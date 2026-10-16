# Databricks notebook source
# ruff: noqa: F821
"""Build reviewer-facing gold tables from the normalized silver layer."""

# COMMAND ----------

import re

from pyspark.sql import functions as F


dbutils.widgets.text("catalog", "workspace", "Unity Catalog catalog")
dbutils.widgets.text("schema", "mlinterp", "Unity Catalog schema")
dbutils.widgets.text("volume", "raw", "Unity Catalog volume")


def checked_identifier(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


catalog = checked_identifier(dbutils.widgets.get("catalog"), "catalog")
schema = checked_identifier(dbutils.widgets.get("schema"), "schema")
volume = checked_identifier(dbutils.widgets.get("volume"), "volume")
spark.sql(f"USE CATALOG `{catalog}`")
spark.sql(f"USE SCHEMA `{schema}`")

# Reading this widget keeps the same configuration contract across notebooks.
export_root = f"/Volumes/{catalog}/{schema}/{volume}/lakehouse_export"


def table(name: str):
    return spark.table(f"{catalog}.{schema}.{name}")


def write_gold(df, name: str) -> None:
    df.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{catalog}.{schema}.{name}")


# COMMAND ----------

runs = table("silver_runs").select(
    "run_id", "dataset", "arm", "temperature", "repeat", "target_column"
)
claims = table("silver_claims").select("run_id", "claim_id", "stage")
layer1_evidence = table("silver_layer1_evidence_verdicts")

gold_layer1_rates = (
    layer1_evidence.join(claims, ["run_id", "claim_id"], "inner")
    .join(runs.select("run_id", "dataset"), "run_id", "inner")
    .groupBy("dataset", "stage")
    .agg(
        F.count("*").alias("evidence_count"),
        F.sum(
            F.when(F.col("layer1_code") == "EVIDENCE_ACCURATE", 1).otherwise(0)
        ).alias("accurate_count"),
        F.sum(
            F.when(F.col("layer1_code") == "EVIDENCE_MISSTATED", 1).otherwise(0)
        ).alias("misstated_count"),
        F.sum(
            F.when(F.col("layer1_code") == "EVIDENCE_FABRICATED", 1).otherwise(0)
        ).alias("fabricated_count"),
    )
    .withColumn("accuracy_rate", F.col("accurate_count") / F.col("evidence_count"))
    .withColumn(
        "misstatement_rate", F.col("misstated_count") / F.col("evidence_count")
    )
)
write_gold(gold_layer1_rates, "gold_layer1_rates")

# COMMAND ----------

claims_enriched = table("silver_claims_enriched")
gold_layer2_scores = (
    claims_enriched.join(
        runs.select("run_id", "dataset", "temperature"), "run_id", "inner"
    )
    .groupBy("dataset", "stage", "temperature")
    .agg(
        F.count("*").alias("claim_count"),
        F.sum(
            F.when(F.col("claim_layer1_code") == "EVIDENCE_ACCURATE", 1).otherwise(0)
        ).alias("layer1_eligible_count"),
        F.count("layer2_score").alias("layer2_claim_count"),
        F.avg("layer2_score").alias("mean_layer2_score"),
        *[
            F.sum(F.when(F.col("layer2_score") == score, 1).otherwise(0)).alias(
                f"score_{score}_count"
            )
            for score in range(4)
        ],
        F.sum(
            F.when(
                (F.col("claim_layer1_code") == "EVIDENCE_ACCURATE")
                & F.col("layer2_score").isNull(),
                1,
            ).otherwise(0)
        ).alias("eligible_missing_layer2_count"),
        F.sum(
            F.when(
                (F.col("claim_layer1_code") != "EVIDENCE_ACCURATE")
                & F.col("layer2_score").isNotNull(),
                1,
            ).otherwise(0)
        ).alias("ineligible_with_layer2_count"),
        F.sum(
            F.when(
                (F.col("claim_layer1_code") == "EVIDENCE_MISSTATED")
                & F.col("layer2_score").isNull(),
                1,
            ).otherwise(0)
        ).alias("evidence_misstated_excluded_count"),
    )
)
write_gold(gold_layer2_scores, "gold_layer2_scores")

# COMMAND ----------

gold_arm_c_detection = (
    table("silver_arm_c_defects")
    .join(runs.select("run_id", "dataset"), "run_id", "inner")
    .groupBy("dataset", "defect")
    .agg(
        F.count("*").alias("planted_count"),
        F.sum(F.col("recalled").cast("long")).alias("detected_count"),
        F.sort_array(
            F.collect_set(F.when(~F.col("recalled"), F.col("column_name")))
        ).alias("missed_columns"),
    )
    .withColumn("detection_rate", F.col("detected_count") / F.col("planted_count"))
)
write_gold(gold_arm_c_detection, "gold_arm_c_detection")

# COMMAND ----------

# A decision is expected at most once per run. Collapsing first makes the
# cross-run grain explicit even if a malformed export contains duplicates.
per_run_decision = (
    claims_enriched.join(
        runs.select("run_id", "dataset", "temperature"), "run_id", "inner"
    )
    .filter(
        (F.col("temperature") == 0.0)
        & F.col("decision_key").isNotNull()
        & F.col("layer2_score").isNotNull()
    )
    .groupBy("dataset", "decision_key", "run_id")
    .agg(F.max("layer2_score").alias("layer2_score"))
)

gold_decision_consistency = (
    per_run_decision.groupBy("dataset", "decision_key")
    .agg(
        F.count("*").alias("run_count"),
        F.sort_array(
            F.collect_list(F.struct("run_id", "layer2_score"))
        ).alias("score_by_run"),
        F.sort_array(F.collect_list("layer2_score")).alias("layer2_scores"),
        F.min("layer2_score").alias("min_layer2_score"),
        F.max("layer2_score").alias("max_layer2_score"),
    )
    .filter(F.col("run_count") >= 2)
    .withColumn(
        "score_spread", F.col("max_layer2_score") - F.col("min_layer2_score")
    )
    .orderBy(F.desc("score_spread"), "dataset", "decision_key")
)
write_gold(gold_decision_consistency, "gold_decision_consistency")

# COMMAND ----------

selected_metrics = table("silver_run_metrics").filter(
    (F.col("reported_arm") != "B") & F.col("best_model_id").isNotNull()
).alias("m")
model_evaluations = table("silver_model_evaluations").select(
    "run_id",
    "model_id",
    F.col("primary_metric").alias("evaluation_primary_metric"),
    F.col("primary_metric_mean").alias("evaluation_primary_metric_value"),
).alias("e")

home_credit_base_rate = (
    table("silver_evidence_refs")
    .join(runs.select("run_id", "dataset", "target_column"), "run_id", "inner")
    .filter(
        (F.col("dataset") == "home_credit")
        & (
            F.col("metric_key")
            == F.concat(
                F.lit("col:"), F.col("target_column"), F.lit(":mean")
            )
        )
    )
    .groupBy("run_id")
    .agg(F.max(F.col("claimed_value_json").cast("double")).alias("base_rate"))
).alias("b")

model_runs = runs.alias("r")

gold_model_metrics = (
    selected_metrics.join(
        model_runs,
        F.col("m.run_id") == F.col("r.run_id"),
        "inner",
    )
    .join(
        model_evaluations,
        (F.col("m.run_id") == F.col("e.run_id"))
        & (F.col("m.best_model_id") == F.col("e.model_id")),
        "left",
    )
    .join(
        home_credit_base_rate,
        F.col("m.run_id") == F.col("b.run_id"),
        "left",
    )
    .select(
        F.col("m.run_id").alias("run_id"),
        F.col("r.dataset").alias("dataset"),
        F.col("r.arm").alias("arm"),
        F.col("r.temperature").alias("temperature"),
        F.col("r.repeat").alias("repeat"),
        F.col("m.best_model_id").alias("selected_model"),
        F.col("e.evaluation_primary_metric").alias("primary_metric"),
        F.coalesce(
            F.col("e.evaluation_primary_metric_value"),
            F.col("m.best_primary_metric_value"),
        ).alias("primary_metric_value"),
        F.when(F.col("r.dataset") == "home_credit", F.col("b.base_rate")).alias(
            "positive_class_base_rate"
        ),
    )
)
write_gold(gold_model_metrics, "gold_model_metrics")

# COMMAND ----------

display(
    spark.createDataFrame(
        [
            ("gold_layer1_rates", gold_layer1_rates.count()),
            ("gold_layer2_scores", gold_layer2_scores.count()),
            ("gold_arm_c_detection", gold_arm_c_detection.count()),
            ("gold_decision_consistency", gold_decision_consistency.count()),
            ("gold_model_metrics", gold_model_metrics.count()),
        ],
        "table_name string, row_count long",
    )
)
