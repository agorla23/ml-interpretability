# Databricks notebook source
# ruff: noqa: F821
"""Cast, deduplicate, join, and validate bronze benchmark tables."""

# COMMAND ----------

import re
from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql import types as T


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

# Reading the widget documents the shared notebook contract even though bronze
# tables, rather than the volume path, are this notebook's direct input.
export_root = f"/Volumes/{catalog}/{schema}/{volume}/lakehouse_export"

# COMMAND ----------

SCHEMA_DDLS = {
    "runs": """
        run_id STRING, source_run_dir STRING, dataset STRING, arm STRING,
        temperature_code STRING, temperature DOUBLE, repeat BIGINT,
        task_type STRING, target_column STRING, source_dataset_path STRING,
        dataset_path STRING, preregistration_path STRING,
        preregistration_sha256 STRING, call_cap_used BIGINT,
        reported_arms ARRAY<STRING>, planted_columns ARRAY<STRING>,
        manual_dataset_corrections_json STRING
    """,
    "claims": """
        run_id STRING, claim_id STRING, stage STRING, subject_type STRING,
        subject STRING, decision STRING, decision_key STRING, decision_text STRING,
        confidence DOUBLE, justification STRING, severity STRING,
        llm_stated_severity STRING, rule_derived_severity STRING,
        severity_rule_violation BOOLEAN, column STRING, action STRING,
        encoding STRING, imputation STRING, scaling STRING, is_override BOOLEAN,
        override_reason STRING, rule_derived_action STRING, model_id STRING,
        rank BIGINT, is_rejection BOOLEAN, primary_metric STRING,
        secondary_metrics ARRAY<STRING>
    """,
    "evidence_refs": """
        run_id STRING, claim_id STRING, evidence_index BIGINT, metric_key STRING,
        claimed_value_type STRING, claimed_value_json STRING
    """,
    "layer1_claim_verdicts": """
        run_id STRING, claim_id STRING, stage STRING,
        citation_layer1_codes ARRAY<STRING>, claim_layer1_code STRING
    """,
    "layer1_evidence_verdicts": """
        run_id STRING, claim_id STRING, metric_key STRING, layer1_code STRING
    """,
    "layer2_verdicts": """
        run_id STRING, claim_id STRING, layer2_score BIGINT,
        discrepancy_note STRING, critic_confidence DOUBLE
    """,
    "run_metrics": """
        run_id STRING, reported_arm STRING, source_artifact STRING,
        run_completed BOOLEAN, run_aborted BOOLEAN, call_cap_used BIGINT,
        llm_call_count BIGINT, total_claims BIGINT, total_citations BIGINT,
        citation_fabrication_rate DOUBLE, citation_misstatement_rate DOUBLE,
        claim_fabrication_rate DOUBLE, claim_misstatement_rate DOUBLE,
        severity_violation_rate DOUBLE, override_rate DOUBLE, columns_kept BIGINT,
        columns_dropped BIGINT, preprocess_claims BIGINT, models_evaluated BIGINT,
        best_model_id STRING, best_primary_metric_value DOUBLE,
        stratification_degraded BOOLEAN, wrong_direction_rate DOUBLE,
        mischaracterization_rate DOUBLE, layer2_claims_scored BIGINT,
        layer2_scale STRING, mean_layer2_score DOUBLE, mean_layer2_score_eda DOUBLE,
        mean_layer2_score_preprocess DOUBLE,
        mean_layer2_score_model_selection DOUBLE,
        mean_layer2_score_evaluation DOUBLE, stage_fallbacks_json STRING
    """,
    "model_evaluations": """
        run_id STRING, model_id STRING, rank BIGINT, primary_metric STRING,
        primary_metric_mean DOUBLE, primary_metric_std DOUBLE,
        train_score_mean DOUBLE, val_score_mean DOUBLE, train_val_gap DOUBLE,
        usable_folds BIGINT, overfitting_flag BOOLEAN
    """,
    "model_evaluation_metrics": """
        run_id STRING, model_id STRING, metric_role STRING, metric_name STRING,
        mean DOUBLE, std DOUBLE
    """,
    "prompt_calls": """
        run_id STRING, call_index BIGINT, stage STRING, latency_seconds DOUBLE,
        input_tokens BIGINT, output_tokens BIGINT, prompt_model STRING,
        temperature DOUBLE, max_tokens BIGINT, response_id STRING,
        response_model STRING, stop_reason STRING, error_json STRING,
        prompt_json STRING, raw_response_json STRING
    """,
    "arm_c_defects": """
        run_id STRING, defect STRING, column_name STRING,
        expected_metric_key STRING, recalled BOOLEAN,
        matching_claim_ids ARRAY<STRING>, seed BIGINT, observed_value DOUBLE,
        profile_value DOUBLE, profile_detected BOOLEAN
    """,
    "data_quality": """
        run_id STRING, rule STRING, severity STRING, artifact STRING,
        record_key STRING, message STRING, details_json STRING
    """,
}

GRAINS = {
    "runs": ["run_id"],
    "claims": ["run_id", "claim_id"],
    "evidence_refs": ["run_id", "claim_id", "evidence_index"],
    "layer1_claim_verdicts": ["run_id", "claim_id"],
    "layer1_evidence_verdicts": ["run_id", "claim_id", "metric_key"],
    "layer2_verdicts": ["run_id", "claim_id"],
    "run_metrics": ["run_id", "reported_arm"],
    "model_evaluations": ["run_id", "model_id"],
    "model_evaluation_metrics": [
        "run_id",
        "model_id",
        "metric_role",
        "metric_name",
    ],
    "prompt_calls": ["run_id", "call_index"],
    "arm_c_defects": ["run_id", "defect"],
    "data_quality": ["run_id", "rule", "artifact", "record_key", "message"],
}


def explicit_schema(ddl: str) -> T.StructType:
    return spark.createDataFrame([], ddl).schema


def cast_columns(df, ddl: str):
    schema_definition = explicit_schema(ddl)
    selected = [
        F.col(field.name).cast(field.dataType).alias(field.name)
        for field in schema_definition.fields
    ]
    return df.select(
        *selected,
        F.col("_ingested_at"),
        F.col("_export_manifest_sha256"),
    )


def write_delta(df, table_name: str) -> None:
    df.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{catalog}.{schema}.{table_name}")


checked_at = datetime.now(timezone.utc)
quality_rows = []

for table_name, ddl in SCHEMA_DDLS.items():
    bronze = spark.table(f"{catalog}.{schema}.bronze_{table_name}")
    casted = cast_columns(bronze, ddl)
    keys = GRAINS[table_name]

    bronze_count = bronze.count()
    null_key_count = casted.filter(
        F.expr(" OR ".join(f"`{key}` IS NULL" for key in keys))
    ).count()
    duplicate_key_count = (
        casted.groupBy(*keys).count().filter(F.col("count") > 1).count()
    )
    expected_silver_count = casted.select(*keys).distinct().count()
    silver = casted.dropDuplicates(keys)
    silver_count = silver.count()

    write_delta(silver, f"silver_{table_name}")
    quality_rows.extend(
        [
            (
                checked_at,
                table_name,
                "row_count",
                "INFO",
                silver_count,
                bronze_count,
                f"Silver rows after grain deduplication; source={export_root}",
            ),
            (
                checked_at,
                table_name,
                "null_keys",
                "PASS" if null_key_count == 0 else "FAIL",
                null_key_count,
                0,
                ", ".join(keys),
            ),
            (
                checked_at,
                table_name,
                "duplicate_keys",
                "PASS" if duplicate_key_count == 0 else "FAIL",
                duplicate_key_count,
                0,
                ", ".join(keys),
            ),
            (
                checked_at,
                table_name,
                "deduplicated_row_count",
                "PASS" if silver_count == expected_silver_count else "FAIL",
                silver_count,
                expected_silver_count,
                "One row per declared table grain",
            ),
        ]
    )

# COMMAND ----------

claims = spark.table(f"{catalog}.{schema}.silver_claims").alias("c")
layer1 = spark.table(f"{catalog}.{schema}.silver_layer1_claim_verdicts").alias("l1")
layer2 = spark.table(f"{catalog}.{schema}.silver_layer2_verdicts").alias("l2")

claims_enriched = (
    claims.join(layer1, ["run_id", "claim_id"], "left")
    .join(layer2, ["run_id", "claim_id"], "left")
    .select(
        "c.*",
        F.col("l1.citation_layer1_codes").alias("citation_layer1_codes"),
        F.col("l1.claim_layer1_code").alias("claim_layer1_code"),
        F.col("l2.layer2_score").alias("layer2_score"),
        F.col("l2.discrepancy_note").alias("layer2_discrepancy_note"),
        F.col("l2.critic_confidence").alias("layer2_critic_confidence"),
    )
)
write_delta(claims_enriched, "silver_claims_enriched")

claim_count = claims_enriched.count()
layer1_joined = claims_enriched.filter(F.col("claim_layer1_code").isNotNull()).count()
layer2_joined = claims_enriched.filter(F.col("layer2_score").isNotNull()).count()
eligible_missing_layer2 = claims_enriched.filter(
    (F.col("claim_layer1_code") == "EVIDENCE_ACCURATE")
    & F.col("layer2_score").isNull()
).count()
ineligible_with_layer2 = claims_enriched.filter(
    (F.col("claim_layer1_code") != "EVIDENCE_ACCURATE")
    & F.col("layer2_score").isNotNull()
).count()

quality_rows.extend(
    [
        (
            checked_at,
            "claims_enriched",
            "layer1_join_coverage",
            "PASS" if layer1_joined == claim_count else "FAIL",
            layer1_joined,
            claim_count,
            "Claims with a claim-level Layer 1 verdict",
        ),
        (
            checked_at,
            "claims_enriched",
            "layer2_join_coverage",
            "INFO",
            layer2_joined,
            claim_count,
            "Left join intentionally retains Layer-1-ineligible claims",
        ),
        (
            checked_at,
            "claims_enriched",
            "eligible_missing_layer2",
            "PASS" if eligible_missing_layer2 == 0 else "FAIL",
            eligible_missing_layer2,
            0,
            "Layer-1-accurate claims without Layer 2",
        ),
        (
            checked_at,
            "claims_enriched",
            "ineligible_with_layer2",
            "PASS" if ineligible_with_layer2 == 0 else "FAIL",
            ineligible_with_layer2,
            0,
            "Layer 2 verdicts on claims that did not pass Layer 1",
        ),
    ]
)

quality_schema = T.StructType(
    [
        T.StructField("checked_at", T.TimestampType(), False),
        T.StructField("table_name", T.StringType(), False),
        T.StructField("check_type", T.StringType(), False),
        T.StructField("status", T.StringType(), False),
        T.StructField("observed_value", T.LongType(), False),
        T.StructField("expected_value", T.LongType(), False),
        T.StructField("details", T.StringType(), False),
    ]
)
quality_checks = spark.createDataFrame(quality_rows, quality_schema)
write_delta(quality_checks, "silver_quality_checks")

display(quality_checks.orderBy("status", "table_name", "check_type"))
