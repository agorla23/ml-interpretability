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
pip install -r requirements.txt
export ANTHROPIC_API_KEY='your-key-here'
```

Benchmark datasets are cached locally under `benchmarks/` and are intentionally not committed. Build or refresh the cache with the dataset utility; Home Credit requires a locally downloaded `application_train.csv` source.

```bash
python3 benchmark_datasets.py --help
python3 benchmark_experiments.py --help
```

To run a benchmark arm after preparing its local dataset cache and preregistration plan:

```bash
python3 benchmark_experiments.py run --dataset titanic --arm ab --temperature 0.0 --repeat 1
```

Run outputs are written under `runs/<run-id>/`. They include `claims.jsonl`, Layer 1 and Layer 2 verdict files, prompts, errors, and `metrics.json`; those generated artifacts remain local by design.

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

Tests that require the gitignored benchmark CSVs skip when those files are not present. Use `benchmark_datasets.py` to build the local benchmark-data cache when you want to run the full suite.
