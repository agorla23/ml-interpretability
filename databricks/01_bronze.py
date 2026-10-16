# Databricks notebook source
# ruff: noqa: F821
"""Load a lakehouse export from a Unity Catalog volume into bronze Delta tables."""

# COMMAND ----------

import hashlib
import json
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

export_root = f"/Volumes/{catalog}/{schema}/{volume}/lakehouse_export"
manifest_path = f"{export_root}/manifest.json"
manifest_row = (
    spark.read.format("binaryFile")
    .load(manifest_path)
    .select("content")
    .first()
)
if manifest_row is None:
    raise FileNotFoundError(f"Export manifest not found: {manifest_path}")

manifest_bytes = bytes(manifest_row["content"])
manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
manifest = json.loads(manifest_bytes.decode("utf-8"))

# COMMAND ----------

load_results = []
for table_name, table_details in sorted(manifest["tables"].items()):
    checked_identifier(table_name, "export table")
    expected_count = int(table_details["row_count"])
    source_path = f"{export_root}/{table_details['path']}"
    target_table = f"{catalog}.{schema}.bronze_{table_name}"

    bronze_df = (
        spark.read.parquet(source_path)
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_export_manifest_sha256", F.lit(manifest_sha256))
    )
    bronze_df.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(target_table)

    actual_count = spark.table(target_table).count()
    if actual_count != expected_count:
        raise RuntimeError(
            f"Bronze row-count mismatch for {table_name}: "
            f"manifest={expected_count}, Delta={actual_count}"
        )
    load_results.append((table_name, expected_count, actual_count, target_table))

display(
    spark.createDataFrame(
        load_results,
        "table_name STRING, manifest_row_count LONG, bronze_row_count LONG, target_table STRING",
    )
)
