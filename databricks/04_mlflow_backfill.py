# Databricks notebook source
# ruff: noqa: F821
"""Backfill one idempotent MLflow run per normalized benchmark run."""

# COMMAND ----------

import json
import re
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as F


dbutils.widgets.text("catalog", "workspace", "Unity Catalog catalog")
dbutils.widgets.text("schema", "mlinterp", "Unity Catalog schema")
dbutils.widgets.text("volume", "raw", "Unity Catalog volume")
dbutils.widgets.text(
    "experiment_path",
    "/Shared/mlinterp-benchmarks",
    "Preferred MLflow experiment path",
)


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


def select_experiment(preferred_path: str):
    try:
        return mlflow.set_experiment(preferred_path)
    except Exception as shared_error:
        current_user = spark.sql("SELECT current_user()").first()[0]
        fallback_path = f"/Users/{current_user}/mlinterp-benchmarks"
        print(
            f"Could not use {preferred_path!r} ({shared_error}); "
            f"falling back to {fallback_path!r}."
        )
        return mlflow.set_experiment(fallback_path)


experiment = select_experiment(dbutils.widgets.get("experiment_path"))
client = MlflowClient()

# COMMAND ----------

runs = table("silver_runs")
selected_metrics = table("silver_run_metrics").filter(
    (F.col("reported_arm") != "B") & F.col("best_model_id").isNotNull()
)

prompt_models = table("silver_prompt_calls").groupBy("run_id").agg(
    F.array_join(
        F.sort_array(
            F.collect_set(
                F.when(
                    ~F.col("stage").startswith("critic_"), F.col("prompt_model")
                )
            )
        ),
        ",",
    ).alias("agent_model"),
    F.array_join(
        F.sort_array(
            F.collect_set(
                F.when(F.col("stage").startswith("critic_"), F.col("prompt_model"))
            )
        ),
        ",",
    ).alias("critic_model"),
)

layer1_rates = table("silver_layer1_evidence_verdicts").groupBy("run_id").agg(
    F.avg(
        F.when(F.col("layer1_code") == "EVIDENCE_ACCURATE", 1.0).otherwise(0.0)
    ).alias("layer1_accuracy_rate")
)
layer2_scores = table("silver_layer2_verdicts").groupBy("run_id").agg(
    F.avg("layer2_score").alias("mean_layer2_score")
)
arm_c_recall = table("silver_arm_c_defects").groupBy("run_id").agg(
    F.avg(F.col("recalled").cast("double")).alias("arm_c_recall")
)

backfill_rows = (
    runs.join(selected_metrics, "run_id", "left")
    .join(prompt_models, "run_id", "left")
    .join(layer1_rates, "run_id", "left")
    .join(layer2_scores, "run_id", "left")
    .join(arm_c_recall, "run_id", "left")
    .select(
        "run_id",
        "dataset",
        "arm",
        "temperature",
        "repeat",
        "agent_model",
        "critic_model",
        F.col("_export_manifest_sha256").alias("export_manifest_sha256"),
        F.col("best_primary_metric_value").alias("primary_model_metric"),
        "layer1_accuracy_rate",
        "mean_layer2_score",
        "override_rate",
        "arm_c_recall",
    )
    .orderBy("run_id")
    .collect()
)

# COMMAND ----------

ARTIFACT_TABLES = {
    "claims": ("silver_claims", ["run_id", "claim_id"]),
    "layer1_claim_verdicts": (
        "silver_layer1_claim_verdicts",
        ["run_id", "claim_id"],
    ),
    "layer1_evidence_verdicts": (
        "silver_layer1_evidence_verdicts",
        ["run_id", "claim_id", "metric_key"],
    ),
    "layer2_verdicts": ("silver_layer2_verdicts", ["run_id", "claim_id"]),
}


def rows_for_run(table_name: str, run_id: str, order_columns: list[str]):
    return [
        row.asDict(recursive=True)
        for row in table(table_name)
        .filter(F.col("run_id") == run_id)
        .orderBy(*order_columns)
        .collect()
    ]


def write_run_artifacts(directory: Path, run_id: str) -> None:
    for artifact_name, (table_name, order_columns) in ARTIFACT_TABLES.items():
        records = rows_for_run(table_name, run_id, order_columns)
        (directory / f"{artifact_name}.json").write_text(
            json.dumps(records, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )


def non_null_values(row, names: list[str]) -> dict[str, float]:
    return {
        name: float(row[name])
        for name in names
        if row[name] is not None
    }


results = []
for row in backfill_rows:
    run_id = row["run_id"]
    existing = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.run_id = '{run_id}'",
        max_results=1,
    )
    if existing:
        results.append((run_id, "skipped", existing[0].info.run_id))
        continue

    tags = {
        "run_id": run_id,
        "export_manifest_sha256": row["export_manifest_sha256"],
        "env_provenance": (
            "reconstructed: benchmark ran on unpinned global Python; "
            "see requirements.txt"
        ),
    }
    params = {
        "dataset": row["dataset"],
        "arm": row["arm"],
        "temperature": row["temperature"],
        "repeat": row["repeat"],
        "agent_model": row["agent_model"] or "unknown",
        "critic_model": row["critic_model"] or "unknown",
    }
    metrics = non_null_values(
        row,
        [
            "primary_model_metric",
            "layer1_accuracy_rate",
            "mean_layer2_score",
            "override_rate",
            "arm_c_recall",
        ],
    )

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=run_id,
        tags=tags,
    ) as active_run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_directory = Path(temporary_directory)
            write_run_artifacts(artifact_directory, run_id)
            mlflow.log_artifacts(
                str(artifact_directory), artifact_path="normalized_records"
            )
        results.append((run_id, "created", active_run.info.run_id))

display(spark.createDataFrame(results, "benchmark_run_id string, status string, mlflow_run_id string"))
