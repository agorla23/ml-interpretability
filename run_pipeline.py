"""Canonical CLI entrypoint for the LangGraph-supervised pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pipeline_supervisor import run_supervised_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LangGraph-supervised ML interpretability pipeline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--critic-model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set in this shell.", file=sys.stderr)
        return 2

    state = run_supervised_pipeline(
        args.dataset,
        args.target,
        args.task_type,
        args.run_id,
        model=args.model,
        critic_model=args.critic_model,
        agent_temperature=args.temperature,
    )
    metrics_path = Path("runs") / args.run_id / "metrics.json"
    print(f"run_id: {state.run_id}")
    print(f"completed: {state.completed}")
    print(f"stage_status: {json.dumps(state.stage_status, sort_keys=True)}")
    layer2_by_claim = {
        verdict.claim_id: verdict for verdict in state.all_layer2_verdicts
    }
    print("model_selection:")
    for choice in state.model_choices:
        verdict = layer2_by_claim.get(choice.claim_id)
        evidence_keys = [reference.metric_key for reference in choice.evidence]
        print(f"  {choice.claim_id} model={choice.model_id} rank={choice.rank} rejection={choice.is_rejection}")
        print(f"    justification: {choice.justification}")
        print(f"    evidence_keys: {json.dumps(evidence_keys)}")
        if verdict is not None:
            print(f"    layer2_score: {verdict.layer2_score}")
            print(f"    discrepancy_note: {verdict.discrepancy_note}")
    print("metrics:")
    print(metrics_path.read_text(encoding="utf-8"), end="")
    return 1 if state.aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
