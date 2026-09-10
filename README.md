# ML Interpretability Pipeline

A LangGraph-supervised pipeline that profiles tabular datasets, generates evidence-backed EDA and preprocessing claims, selects models, evaluates them, and scores claims with a Layer-2 critic.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY='your-key-here'
```

Downloaded benchmark data and generated run artifacts are intentionally excluded from Git. Rebuild the local benchmark cache with:

```bash
python3 benchmark_datasets.py
```

## Run

```bash
python3 run_pipeline.py \
  --dataset benchmarks/ames_housing.csv \
  --target price \
  --task-type regression \
  --run-id ames_example
```

Artifacts for a run are written under `runs/<run-id>/`.

## Test

```bash
python3 -m pytest -q
```
