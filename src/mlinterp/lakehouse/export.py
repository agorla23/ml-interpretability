"""Export normalized Milestone 9 run artifacts as Parquet tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from mlinterp.lakehouse.run_parser import (
    TABLE_NAMES,
    discover_run_directories,
    parse_run_directory,
)


SECRET_PATTERNS = (
    ("anthropic_api_key", re.compile(rb"sk-ant-[A-Za-z0-9_-]{8,}", re.IGNORECASE)),
    ("openai_api_key", re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}", re.IGNORECASE)),
    ("aws_access_key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("github_token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}", re.IGNORECASE)),
    ("google_api_key", re.compile(rb"AIza[0-9A-Za-z_-]{30,}")),
    ("slack_token", re.compile(rb"xox[baprs]-[0-9A-Za-z-]{10,}", re.IGNORECASE)),
    (
        "private_key",
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.IGNORECASE),
    ),
    ("bearer_token", re.compile(rb"Bearer\s+[A-Za-z0-9._~+/-]{20,}", re.IGNORECASE)),
    (
        "jwt",
        re.compile(rb"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ),
    (
        "secret_assignment",
        re.compile(
            rb"(?:api[_-]?key|secret|access[_-]?token|authorization).{0,12}"
            rb"[=:].{0,4}[A-Za-z0-9_./+=-]{20,}",
            re.IGNORECASE,
        ),
    ),
)


def _table_schemas() -> dict[str, pa.Schema]:
    string = pa.string()
    integer = pa.int64()
    floating = pa.float64()
    boolean = pa.bool_()
    strings = pa.list_(string)
    return {
        "runs": pa.schema(
            [
                ("run_id", string),
                ("source_run_dir", string),
                ("dataset", string),
                ("arm", string),
                ("temperature_code", string),
                ("temperature", floating),
                ("repeat", integer),
                ("task_type", string),
                ("target_column", string),
                ("source_dataset_path", string),
                ("dataset_path", string),
                ("preregistration_path", string),
                ("preregistration_sha256", string),
                ("call_cap_used", integer),
                ("reported_arms", strings),
                ("planted_columns", strings),
                ("manual_dataset_corrections_json", string),
            ]
        ),
        "claims": pa.schema(
            [
                ("run_id", string),
                ("claim_id", string),
                ("stage", string),
                ("subject_type", string),
                ("subject", string),
                ("decision", string),
                ("decision_key", string),
                ("decision_text", string),
                ("confidence", floating),
                ("justification", string),
                ("severity", string),
                ("llm_stated_severity", string),
                ("rule_derived_severity", string),
                ("severity_rule_violation", boolean),
                ("column", string),
                ("action", string),
                ("encoding", string),
                ("imputation", string),
                ("scaling", string),
                ("is_override", boolean),
                ("override_reason", string),
                ("rule_derived_action", string),
                ("model_id", string),
                ("rank", integer),
                ("is_rejection", boolean),
                ("primary_metric", string),
                ("secondary_metrics", strings),
            ]
        ),
        "evidence_refs": pa.schema(
            [
                ("run_id", string),
                ("claim_id", string),
                ("evidence_index", integer),
                ("metric_key", string),
                ("claimed_value_type", string),
                ("claimed_value_json", string),
            ]
        ),
        "layer1_claim_verdicts": pa.schema(
            [
                ("run_id", string),
                ("claim_id", string),
                ("stage", string),
                ("citation_layer1_codes", strings),
                ("claim_layer1_code", string),
            ]
        ),
        "layer1_evidence_verdicts": pa.schema(
            [
                ("run_id", string),
                ("claim_id", string),
                ("metric_key", string),
                ("layer1_code", string),
            ]
        ),
        "layer2_verdicts": pa.schema(
            [
                ("run_id", string),
                ("claim_id", string),
                ("layer2_score", integer),
                ("discrepancy_note", string),
                ("critic_confidence", floating),
            ]
        ),
        "run_metrics": pa.schema(
            [
                ("run_id", string),
                ("reported_arm", string),
                ("source_artifact", string),
                ("run_completed", boolean),
                ("run_aborted", boolean),
                ("call_cap_used", integer),
                ("llm_call_count", integer),
                ("total_claims", integer),
                ("total_citations", integer),
                ("citation_fabrication_rate", floating),
                ("citation_misstatement_rate", floating),
                ("claim_fabrication_rate", floating),
                ("claim_misstatement_rate", floating),
                ("severity_violation_rate", floating),
                ("override_rate", floating),
                ("columns_kept", integer),
                ("columns_dropped", integer),
                ("preprocess_claims", integer),
                ("models_evaluated", integer),
                ("best_model_id", string),
                ("best_primary_metric_value", floating),
                ("stratification_degraded", boolean),
                ("wrong_direction_rate", floating),
                ("mischaracterization_rate", floating),
                ("layer2_claims_scored", integer),
                ("layer2_scale", string),
                ("mean_layer2_score", floating),
                ("mean_layer2_score_eda", floating),
                ("mean_layer2_score_preprocess", floating),
                ("mean_layer2_score_model_selection", floating),
                ("mean_layer2_score_evaluation", floating),
                ("stage_fallbacks_json", string),
            ]
        ),
        "model_evaluations": pa.schema(
            [
                ("run_id", string),
                ("model_id", string),
                ("rank", integer),
                ("primary_metric", string),
                ("primary_metric_mean", floating),
                ("primary_metric_std", floating),
                ("train_score_mean", floating),
                ("val_score_mean", floating),
                ("train_val_gap", floating),
                ("usable_folds", integer),
                ("overfitting_flag", boolean),
            ]
        ),
        "model_evaluation_metrics": pa.schema(
            [
                ("run_id", string),
                ("model_id", string),
                ("metric_role", string),
                ("metric_name", string),
                ("mean", floating),
                ("std", floating),
            ]
        ),
        "prompt_calls": pa.schema(
            [
                ("run_id", string),
                ("call_index", integer),
                ("stage", string),
                ("latency_seconds", floating),
                ("input_tokens", integer),
                ("output_tokens", integer),
                ("prompt_model", string),
                ("temperature", floating),
                ("max_tokens", integer),
                ("response_id", string),
                ("response_model", string),
                ("stop_reason", string),
                ("error_json", string),
                ("prompt_json", string),
                ("raw_response_json", string),
            ]
        ),
        "arm_c_defects": pa.schema(
            [
                ("run_id", string),
                ("defect", string),
                ("column_name", string),
                ("expected_metric_key", string),
                ("recalled", boolean),
                ("matching_claim_ids", strings),
                ("seed", integer),
                ("observed_value", floating),
                ("profile_value", floating),
                ("profile_detected", boolean),
            ]
        ),
        "data_quality": pa.schema(
            [
                ("run_id", string),
                ("rule", string),
                ("severity", string),
                ("artifact", string),
                ("record_key", string),
                ("message", string),
                ("details_json", string),
            ]
        ),
    }


def find_secret_matches(run_directories: list[Path]) -> list[dict[str, Any]]:
    """Return possible secret locations without exposing matching values."""
    matches: list[dict[str, Any]] = []
    for run_dir in run_directories:
        for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
            with path.open("rb") as handle:
                for line_number, line in enumerate(handle, start=1):
                    for pattern_name, pattern in SECRET_PATTERNS:
                        if pattern.search(line):
                            matches.append(
                                {
                                    "path": path.as_posix(),
                                    "line": line_number,
                                    "pattern": pattern_name,
                                }
                            )
    return matches


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_runs(runs_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Normalize matching runs and write Parquet tables plus a manifest."""
    runs_root = Path(runs_dir)
    output_root = Path(out_dir)
    run_directories = discover_run_directories(runs_root)
    secret_matches = find_secret_matches(run_directories)
    if secret_matches:
        locations = ", ".join(
            f"{match['path']}:{match['line']} ({match['pattern']})"
            for match in secret_matches
        )
        raise RuntimeError(f"Possible secrets found; export stopped: {locations}")

    records = {name: [] for name in TABLE_NAMES}
    source_files: list[Path] = []
    for run_directory in run_directories:
        parsed = parse_run_directory(run_directory)
        source_files.extend(parsed.source_files)
        for table_name in TABLE_NAMES:
            records[table_name].extend(parsed.tables[table_name])

    output_root.mkdir(parents=True, exist_ok=True)
    schemas = _table_schemas()
    table_manifest: dict[str, dict[str, Any]] = {}
    for table_name in TABLE_NAMES:
        output_path = output_root / f"{table_name}.parquet"
        table = pa.Table.from_pylist(records[table_name], schema=schemas[table_name])
        pq.write_table(table, output_path)
        table_manifest[table_name] = {
            "path": output_path.name,
            "row_count": table.num_rows,
        }

    source_manifest = []
    for path in sorted(set(source_files)):
        source_manifest.append(
            {
                "path": path.relative_to(runs_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_dir": runs_root.as_posix(),
        "included_run_directories": [path.name for path in run_directories],
        "tables": table_manifest,
        "source_files": source_manifest,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Milestone 9 benchmark runs as normalized Parquet tables."
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="lakehouse_export/")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = export_runs(args.runs_dir, args.out)
    counts = {
        name: details["row_count"] for name, details in manifest["tables"].items()
    }
    print(json.dumps({"runs": len(manifest["included_run_directories"]), "rows": counts}, indent=2))


if __name__ == "__main__":
    main()
