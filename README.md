# ML Interpretability Pipeline

[![CI](https://github.com/agorla23/ml-interpretability/actions/workflows/ci.yml/badge.svg)](https://github.com/agorla23/ml-interpretability/actions/workflows/ci.yml)

An evidence-backed, LLM-assisted pipeline for analyzing tabular machine-learning datasets. It profiles a dataset, asks an LLM for structured decisions, verifies the factual citations in those decisions, and persists the artifacts needed to inspect a run afterward.

## Architecture

Each run moves through four stages:

1. **EDA** identifies data-quality, leakage, distribution, and relationship findings from a deterministic profile.
2. **Preprocessing** produces a per-column transformation plan, including imputation, encoding, scaling, and drops.
3. **Model selection** chooses candidate models appropriate to the task and observed data.
4. **Evaluation** cross-validates the selected models and records their metrics and diagnostics.

The pipeline has two independent verification layers. **Layer 1** deterministically checks each cited metric key and value against the profiler output, recording whether the evidence is accurate, misstated, or unresolved. **Layer 2** is an LLM critic that evaluates the reasoning and confidence of claims that pass the Layer 1 evidence check. Both layers are persisted separately from the model-facing claims so their judgments can be audited.

## Setup

```bash
git clone https://github.com/agorla23/ml-interpretability.git
cd ml-interpretability
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export ANTHROPIC_API_KEY='your-key-here'
```

Benchmark datasets are cached locally under `benchmarks/` and are intentionally not committed. Build or refresh the cache with the dataset utility; Home Credit requires a locally downloaded `application_train.csv` source.

```bash
python -m mlinterp.benchmarks.benchmark_datasets --help
python -m mlinterp.benchmarks.benchmark_experiments --help
```

To run a benchmark arm after preparing its local dataset cache and preregistration plan:

```bash
python -m mlinterp.benchmarks.benchmark_experiments run --dataset titanic --arm ab --temperature 0.0 --repeat 1
```

Run the supervised pipeline on a prepared dataset:

```bash
python -m mlinterp.pipeline.run_pipeline --dataset tests/fixtures/mini_dataset.csv --target target --task-type binary --run-id example
```

Run outputs are written under `runs/<run-id>/`. They include `claims.jsonl`, Layer 1 and Layer 2 verdict files, prompts, errors, and `metrics.json`; those generated artifacts remain local by design.

## Lakehouse and experiment tracking

The Databricks analysis layer is additive and reads persisted benchmark artifacts:

```text
runs/ → Parquet export → Unity Catalog volume → bronze/silver/gold Delta
                                                    ├── MLflow
                                                    └── SQL findings checks
```

First, build the normalized local export. The command accepts only the Milestone 9 run-directory naming convention and writes source hashes and row counts to `manifest.json`.

```bash
python -m mlinterp.lakehouse.export --runs-dir runs --out lakehouse_export/
python -m scripts.verify_lakehouse_gold --export-dir lakehouse_export/
```

The Databricks notebooks default to catalog `workspace`, schema `mlinterp`, and volume `raw`; all three can be changed with notebook widgets. To upload and run the layer in the default Free Edition workspace:

1. Configure the current Databricks CLI for the workspace, then copy the complete export directory into the existing Unity Catalog volume:

   ```bash
   databricks fs cp lakehouse_export \
     dbfs:/Volumes/workspace/mlinterp/raw/lakehouse_export \
     --recursive --overwrite
   ```

2. Import the source-format notebooks into your own workspace folder (replace the email address):

   ```bash
   databricks workspace import-dir databricks \
     /Users/you@example.com/mlinterp-databricks --overwrite
   ```

   The same files can instead be imported from the Workspace UI. The first-line `Databricks notebook source` marker and `COMMAND` separators preserve notebook cells.

3. Attach serverless compute and run `01_bronze`, `02_silver`, `03_gold`, then `04_mlflow_backfill`, in that order. Bronze checks every loaded row count against the uploaded manifest; silver records schema, key, deduplication, and join checks; gold contains only aggregate analysis tables.
4. Open `databricks/queries.sql` in the SQL editor and run each statement against `workspace.mlinterp`.
5. Review the experiment at `/Shared/mlinterp-benchmarks`. If `/Shared` is not writable, the backfill notebook reports and uses `/Users/<current-user>/mlinterp-benchmarks`. Re-running it is safe because an existing MLflow run with the same `run_id` tag is skipped.

No API key is needed for export, notebook execution, SQL queries, or MLflow backfill. Prompt payloads remain in `silver_prompt_calls`; no gold table includes them.

## Findings

See [FINDINGS.md](FINDINGS.md) for the concise record of confirmed benchmark findings and fixes referenced by the accompanying writeup. It distinguishes pipeline correctness issues from dataset-specific outcomes and points to the persisted artifacts used for review.

## Tests

```bash
python3 -m pytest -q
```

## Development

Create and activate a virtual environment, then install the runtime and development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Run linting and tests before opening a pull request:

```bash
ruff check .
python -m pytest -q
```

Tests that require the gitignored benchmark CSVs skip when those files are not present. Use `python -m mlinterp.benchmarks.benchmark_datasets` to build the local benchmark-data cache when you want to run the full suite.
