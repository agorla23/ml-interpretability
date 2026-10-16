"""Parse Milestone 9 benchmark artifacts into normalized, tabular records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RUN_DIRECTORY_PATTERN = re.compile(
    r"^m9_(?P<dataset>[a-z0-9_]+)_(?P<arm>ab|c)_t(?P<temperature>00|07)_r(?P<repeat>[0-9]+)$"
)

COMMON_REQUIRED_FILES = (
    "claims.jsonl",
    "claim_verdicts.jsonl",
    "evidence_verdicts.jsonl",
    "layer2_verdicts.jsonl",
    "evaluation_results.jsonl",
    "experiment_metadata.json",
    "metrics.json",
    "prompts.jsonl",
)
ARM_REQUIRED_FILES = {
    "ab": (
        "arm_a_metrics.json",
        "arm_b_metrics.json",
        "preregistration_comparison.json",
    ),
    "c": (
        "arm_c_recall.json",
        "synthetic_corruption.json",
        "input_dataset.csv",
    ),
}

TABLE_NAMES = (
    "runs",
    "claims",
    "evidence_refs",
    "layer1_claim_verdicts",
    "layer1_evidence_verdicts",
    "layer2_verdicts",
    "run_metrics",
    "model_evaluations",
    "model_evaluation_metrics",
    "prompt_calls",
    "arm_c_defects",
    "data_quality",
)

EDA_FLAG_DECISIONS = {
    "has_high_missing": "flag_high_missing",
    "has_moderate_missing": "flag_missingness",
    "is_constant": "flag_constant",
    "is_duplicate_of_other_column": "flag_duplicate",
    "is_high_cardinality": "flag_high_cardinality",
    "is_id_like": "flag_identifier",
    "is_imbalanced": "flag_class_imbalance",
    "is_low_variance": "flag_low_variance",
    "is_multicollinear": "flag_multicollinearity",
    "is_numeric_but_categorical": "flag_numeric_categorical",
    "is_sequential": "flag_sequential",
    "is_skewed": "flag_skew",
    "is_zero_inflated": "flag_zero_inflation",
    "leakage_suspect": "flag_leakage",
    "missingness_informative": "flag_informative_missingness",
    "missingness_leakage_suspect": "flag_leakage",
}


@dataclass(frozen=True)
class RunIdentity:
    """Identity fields encoded in a Milestone 9 run-directory name."""

    run_id: str
    dataset: str
    arm: str
    temperature_code: str
    temperature: float
    repeat: int


@dataclass
class ParsedRun:
    """Normalized records and source files for one run directory."""

    identity: RunIdentity
    source_files: list[Path]
    tables: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {name: [] for name in TABLE_NAMES}
    )
    _source_files_by_name: dict[str, Path] = field(default_factory=dict, repr=False)


def parse_run_name(name: str) -> RunIdentity:
    """Parse an exact Milestone 9 run-directory name."""
    match = RUN_DIRECTORY_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Not a Milestone 9 benchmark run directory: {name!r}")
    temperature_code = f"t{match.group('temperature')}"
    return RunIdentity(
        run_id=name,
        dataset=match.group("dataset"),
        arm=match.group("arm"),
        temperature_code=temperature_code,
        temperature=0.0 if temperature_code == "t00" else 0.7,
        repeat=int(match.group("repeat")),
    )


def discover_run_directories(runs_dir: str | Path) -> list[Path]:
    """Return only directories that match the approved Milestone 9 grammar."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and RUN_DIRECTORY_PATTERN.fullmatch(path.name)
    )


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _quality_record(
    identity: RunIdentity,
    rule: str,
    artifact: str,
    message: str,
    *,
    record_key: str | None = None,
    severity: str = "error",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": identity.run_id,
        "rule": rule,
        "severity": severity,
        "artifact": artifact,
        "record_key": record_key,
        "message": message,
        "details_json": _json_text(details or {}),
    }


def _missing_file(
    result: ParsedRun,
    artifact: str,
) -> None:
    result.tables["data_quality"].append(
        _quality_record(
            result.identity,
            "missing_required_file",
            artifact,
            f"Required artifact {artifact!r} is missing",
        )
    )


def _load_jsonl(result: ParsedRun, artifact: str) -> list[dict[str, Any]]:
    path = result._source_files_by_name.get(artifact)
    if path is None:
        _missing_file(result, artifact)
        return []

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            result.tables["data_quality"].append(
                _quality_record(
                    result.identity,
                    "malformed_json",
                    artifact,
                    f"Could not parse JSON on line {line_number}: {exc.msg}",
                    record_key=f"line:{line_number}",
                    details={"column": exc.colno},
                )
            )
            continue
        if not isinstance(value, dict):
            result.tables["data_quality"].append(
                _quality_record(
                    result.identity,
                    "malformed_json",
                    artifact,
                    f"JSON line {line_number} is not an object",
                    record_key=f"line:{line_number}",
                )
            )
            continue
        rows.append(value)
    return rows


def _load_json(result: ParsedRun, artifact: str) -> dict[str, Any]:
    path = result._source_files_by_name.get(artifact)
    if path is None:
        _missing_file(result, artifact)
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result.tables["data_quality"].append(
            _quality_record(
                result.identity,
                "malformed_json",
                artifact,
                f"Could not parse JSON: {exc.msg}",
                details={"line": exc.lineno, "column": exc.colno},
            )
        )
        return {}
    if not isinstance(value, dict):
        result.tables["data_quality"].append(
            _quality_record(
                result.identity,
                "malformed_json",
                artifact,
                "JSON document is not an object",
            )
        )
        return {}
    return value


def _preprocessing_decision(claim: dict[str, Any]) -> str | None:
    action = claim.get("action")
    if action == "drop":
        return "drop"
    if action == "scale" and claim.get("scaling") not in {None, "none"}:
        return f"scale_{claim['scaling']}"
    if action == "encode" and claim.get("encoding") not in {None, "none"}:
        return f"encode_{claim['encoding']}"
    if action == "impute_and_scale":
        imputation = claim.get("imputation")
        scaling = claim.get("scaling")
        if imputation not in {None, "none"} and scaling not in {None, "none"}:
            return f"impute_{imputation}+scale_{scaling}"
    if action == "impute_and_encode":
        imputation = claim.get("imputation")
        encoding = claim.get("encoding")
        if imputation not in {None, "none"} and encoding not in {None, "none"}:
            return f"impute_{imputation}+encode_{encoding}"
    return None


def _eda_decision(
    claim: dict[str, Any],
    dataset: str,
) -> tuple[str | None, str | None, str | None]:
    candidates: set[tuple[str, str, str]] = set()
    structured_subjects: set[tuple[str, str]] = set()
    for evidence in claim.get("evidence") or []:
        if not isinstance(evidence, dict):
            continue
        metric_key = evidence.get("metric_key")
        if not isinstance(metric_key, str):
            continue
        parts = metric_key.split(":")
        subject: tuple[str, str] | None = None
        if len(parts) >= 3 and parts[0] == "col":
            subject = ("column", parts[1])
            structured_subjects.add(subject)
        elif len(parts) >= 2 and parts[0] == "target":
            subject = ("dataset", dataset)
            structured_subjects.add(subject)
        decision = EDA_FLAG_DECISIONS.get(parts[-1])
        if subject is not None and decision is not None and evidence.get("claimed_value") is True:
            candidates.add((*subject, decision))

    if len(candidates) == 1:
        return next(iter(candidates))
    if len(structured_subjects) == 1:
        subject_type, subject = next(iter(structured_subjects))
        return subject_type, subject, None
    return None, None, None


def derive_decision_key(
    claim: dict[str, Any],
    dataset: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Derive a stable decision key without inspecting claim prose."""
    stage = claim.get("stage")
    subject_type: str | None = None
    subject: str | None = None
    decision: str | None = None

    if stage == "preprocess" and isinstance(claim.get("column"), str):
        subject_type = "column"
        subject = claim["column"]
        decision = _preprocessing_decision(claim)
    elif stage == "model_selection" and isinstance(claim.get("model_id"), str):
        subject_type = "model"
        subject = claim["model_id"]
        if isinstance(claim.get("is_rejection"), bool):
            decision = "reject" if claim["is_rejection"] else "accept"
    elif stage == "evaluation" and isinstance(claim.get("primary_metric"), str):
        subject_type = "metric"
        subject = claim["primary_metric"]
        decision = "select"
    elif stage == "eda":
        subject_type, subject, decision = _eda_decision(claim, dataset)

    decision_key = None
    if isinstance(stage, str) and subject is not None and decision is not None:
        decision_key = f"{stage}|{subject}|{decision}"
    return subject_type, subject, decision, decision_key


def _normalize_claims(
    result: ParsedRun,
    rows: list[dict[str, Any]],
) -> tuple[set[str], list[dict[str, Any]]]:
    claim_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        claim_id = row.get("claim_id")
        if isinstance(claim_id, str):
            claim_ids.add(claim_id)
        subject_type, subject, decision, decision_key = derive_decision_key(
            row, result.identity.dataset
        )
        if decision_key is None:
            result.tables["data_quality"].append(
                _quality_record(
                    result.identity,
                    "unkeyable_claim",
                    "claims.jsonl",
                    "Claim cannot be keyed from structured fields without parsing prose",
                    record_key=claim_id if isinstance(claim_id, str) else None,
                    severity="warning",
                    details={
                        "stage": row.get("stage"),
                        "subject_type": subject_type,
                        "subject": subject,
                    },
                )
            )
        normalized.append(
            {
                "run_id": result.identity.run_id,
                "claim_id": claim_id,
                "stage": row.get("stage"),
                "subject_type": subject_type,
                "subject": subject,
                "decision": decision,
                "decision_key": decision_key,
                "decision_text": row.get("decision"),
                "confidence": row.get("confidence"),
                "justification": row.get("justification"),
                "severity": row.get("severity"),
                "llm_stated_severity": row.get("llm_stated_severity"),
                "rule_derived_severity": row.get("rule_derived_severity"),
                "severity_rule_violation": row.get("severity_rule_violation"),
                "column": row.get("column"),
                "action": row.get("action"),
                "encoding": row.get("encoding"),
                "imputation": row.get("imputation"),
                "scaling": row.get("scaling"),
                "is_override": row.get("is_override"),
                "override_reason": row.get("override_reason"),
                "rule_derived_action": row.get("rule_derived_action"),
                "model_id": row.get("model_id"),
                "rank": row.get("rank"),
                "is_rejection": row.get("is_rejection"),
                "primary_metric": row.get("primary_metric"),
                "secondary_metrics": row.get("secondary_metrics"),
            }
        )
        for evidence_index, evidence in enumerate(row.get("evidence") or []):
            if not isinstance(evidence, dict):
                continue
            claimed_value = evidence.get("claimed_value")
            result.tables["evidence_refs"].append(
                {
                    "run_id": result.identity.run_id,
                    "claim_id": claim_id,
                    "evidence_index": evidence_index,
                    "metric_key": evidence.get("metric_key"),
                    "claimed_value_type": _value_type(claimed_value),
                    "claimed_value_json": _json_text(claimed_value),
                }
            )
    return claim_ids, normalized


def _normalize_layer1_claims(
    result: ParsedRun,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": result.identity.run_id,
            "claim_id": row.get("claim_id"),
            "stage": row.get("stage"),
            "citation_layer1_codes": row.get("citation_layer1_codes"),
            "claim_layer1_code": row.get("claim_layer1_code"),
        }
        for row in rows
    ]


def _normalize_layer1_evidence(
    result: ParsedRun,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": result.identity.run_id,
            "claim_id": row.get("claim_id"),
            "metric_key": row.get("metric_key"),
            "layer1_code": row.get("layer1_code"),
        }
        for row in rows
    ]


def _normalize_layer2(
    result: ParsedRun,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": result.identity.run_id,
            "claim_id": row.get("claim_id"),
            "layer2_score": row.get("layer2_score"),
            "discrepancy_note": row.get("discrepancy_note"),
            "critic_confidence": row.get("critic_confidence"),
        }
        for row in rows
    ]


def _normalize_metrics(
    identity: RunIdentity,
    metrics: dict[str, Any],
    reported_arm: str,
    source_artifact: str,
) -> dict[str, Any]:
    stage_scores = metrics.get("mean_layer2_score_by_stage") or {}
    return {
        "run_id": identity.run_id,
        "reported_arm": reported_arm,
        "source_artifact": source_artifact,
        "run_completed": metrics.get("run_completed"),
        "run_aborted": metrics.get("run_aborted"),
        "call_cap_used": metrics.get("call_cap_used"),
        "llm_call_count": metrics.get("llm_call_count"),
        "total_claims": metrics.get("total_claims"),
        "total_citations": metrics.get("total_citations"),
        "citation_fabrication_rate": metrics.get("citation_fabrication_rate"),
        "citation_misstatement_rate": metrics.get("citation_misstatement_rate"),
        "claim_fabrication_rate": metrics.get("claim_fabrication_rate"),
        "claim_misstatement_rate": metrics.get("claim_misstatement_rate"),
        "severity_violation_rate": metrics.get("severity_violation_rate"),
        "override_rate": metrics.get("override_rate"),
        "columns_kept": metrics.get("columns_kept"),
        "columns_dropped": metrics.get("columns_dropped"),
        "preprocess_claims": metrics.get("preprocess_claims"),
        "models_evaluated": metrics.get("models_evaluated"),
        "best_model_id": metrics.get("best_model_id"),
        "best_primary_metric_value": metrics.get("best_primary_metric_value"),
        "stratification_degraded": metrics.get("stratification_degraded"),
        "wrong_direction_rate": metrics.get("wrong_direction_rate"),
        "mischaracterization_rate": metrics.get("mischaracterization_rate"),
        "layer2_claims_scored": metrics.get("layer2_claims_scored"),
        "layer2_scale": metrics.get("layer2_scale"),
        "mean_layer2_score": metrics.get("mean_layer2_score"),
        "mean_layer2_score_eda": stage_scores.get("eda"),
        "mean_layer2_score_preprocess": stage_scores.get("preprocess"),
        "mean_layer2_score_model_selection": stage_scores.get("model_selection"),
        "mean_layer2_score_evaluation": stage_scores.get("evaluation"),
        "stage_fallbacks_json": _json_text(metrics.get("stage_fallbacks") or {}),
    }


def _normalize_evaluations(
    result: ParsedRun,
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        model_id = row.get("model_id")
        primary_metric = row.get("primary_metric")
        result.tables["model_evaluations"].append(
            {
                "run_id": result.identity.run_id,
                "model_id": model_id,
                "rank": row.get("rank"),
                "primary_metric": primary_metric,
                "primary_metric_mean": row.get("primary_metric_mean"),
                "primary_metric_std": row.get("primary_metric_std"),
                "train_score_mean": row.get("train_score_mean"),
                "val_score_mean": row.get("val_score_mean"),
                "train_val_gap": row.get("train_val_gap"),
                "usable_folds": row.get("usable_folds"),
                "overfitting_flag": row.get("overfitting_flag"),
            }
        )
        if isinstance(primary_metric, str):
            result.tables["model_evaluation_metrics"].append(
                {
                    "run_id": result.identity.run_id,
                    "model_id": model_id,
                    "metric_role": "primary",
                    "metric_name": primary_metric,
                    "mean": row.get("primary_metric_mean"),
                    "std": row.get("primary_metric_std"),
                }
            )
        secondary = row.get("secondary_metrics") or {}
        if isinstance(secondary, dict):
            for metric_name, values in sorted(secondary.items()):
                values = values if isinstance(values, dict) else {}
                result.tables["model_evaluation_metrics"].append(
                    {
                        "run_id": result.identity.run_id,
                        "model_id": model_id,
                        "metric_role": "secondary",
                        "metric_name": metric_name,
                        "mean": values.get("mean"),
                        "std": values.get("std"),
                    }
                )


def _normalize_prompts(
    result: ParsedRun,
    rows: list[dict[str, Any]],
) -> None:
    for call_index, row in enumerate(rows):
        prompt = row.get("prompt") if isinstance(row.get("prompt"), dict) else {}
        response = row.get("raw_response") if isinstance(row.get("raw_response"), dict) else {}
        tokens = row.get("tokens") if isinstance(row.get("tokens"), dict) else {}
        result.tables["prompt_calls"].append(
            {
                "run_id": result.identity.run_id,
                "call_index": call_index,
                "stage": row.get("stage"),
                "latency_seconds": row.get("latency_seconds"),
                "input_tokens": tokens.get("input"),
                "output_tokens": tokens.get("output"),
                "prompt_model": prompt.get("model"),
                "temperature": prompt.get("temperature"),
                "max_tokens": prompt.get("max_tokens"),
                "response_id": response.get("id"),
                "response_model": response.get("model"),
                "stop_reason": response.get("stop_reason"),
                "error_json": _json_text(row.get("error")),
                "prompt_json": _json_text(row.get("prompt")),
                "raw_response_json": _json_text(row.get("raw_response")),
            }
        )


def _normalize_arm_c(
    result: ParsedRun,
    recall: dict[str, Any],
    corruption: dict[str, Any],
) -> None:
    observed_fields = {
        "constant": "constant_nunique",
        "leakage": "leaked_corr_with_target",
        "high_missing": "high_missing_pct",
    }
    profile = corruption.get("profile") or {}
    profile_fields = {
        "constant": (None, "constant_is_constant"),
        "leakage": ("leaked_corr_with_target", "leaked_leakage_suspect"),
        "high_missing": (None, "high_missing_has_high_missing"),
    }
    for defect in recall.get("defects") or []:
        if not isinstance(defect, dict):
            continue
        defect_name = defect.get("defect")
        profile_value_field, profile_detected_field = profile_fields.get(
            defect_name, (None, None)
        )
        result.tables["arm_c_defects"].append(
            {
                "run_id": result.identity.run_id,
                "defect": defect_name,
                "column_name": defect.get("column"),
                "expected_metric_key": defect.get("expected_metric_key"),
                "recalled": defect.get("recalled"),
                "matching_claim_ids": defect.get("matching_claim_ids"),
                "seed": corruption.get("seed"),
                "observed_value": corruption.get(observed_fields.get(defect_name, "")),
                "profile_value": profile.get(profile_value_field)
                if profile_value_field is not None
                else None,
                "profile_detected": profile.get(profile_detected_field)
                if profile_detected_field is not None
                else None,
            }
        )


def _validate_claim_references(
    result: ParsedRun,
    claim_ids: set[str],
    artifact: str,
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        claim_id = row.get("claim_id")
        if claim_id not in claim_ids:
            result.tables["data_quality"].append(
                _quality_record(
                    result.identity,
                    "orphan_claim_id",
                    artifact,
                    "Verdict references a claim_id not present in claims.jsonl",
                    record_key=claim_id if isinstance(claim_id, str) else None,
                )
            )


def _validate_layer2_gate(
    result: ParsedRun,
    claim_ids: set[str],
    layer1_rows: list[dict[str, Any]],
    layer2_rows: list[dict[str, Any]],
) -> None:
    eligible = {
        row.get("claim_id")
        for row in layer1_rows
        if row.get("claim_id") in claim_ids
        and row.get("claim_layer1_code") == "EVIDENCE_ACCURATE"
    }
    layer2_ids = {row.get("claim_id") for row in layer2_rows}
    for claim_id in sorted(eligible - layer2_ids):
        result.tables["data_quality"].append(
            _quality_record(
                result.identity,
                "layer2_missing_eligible",
                "layer2_verdicts.jsonl",
                "Layer-1-eligible claim has no Layer 2 verdict",
                record_key=claim_id,
            )
        )
    for claim_id in sorted(layer2_ids - eligible, key=lambda value: str(value)):
        result.tables["data_quality"].append(
            _quality_record(
                result.identity,
                "layer2_ineligible_claim",
                "layer2_verdicts.jsonl",
                "Layer 2 verdict exists for a claim that is not Layer-1 eligible",
                record_key=claim_id if isinstance(claim_id, str) else None,
            )
        )


def parse_run_directory(run_dir: str | Path) -> ParsedRun:
    """Parse one run without mutating it, returning violations as table rows."""
    path = Path(run_dir)
    identity = parse_run_name(path.name)
    source_files = sorted(item for item in path.rglob("*") if item.is_file())
    result = ParsedRun(identity=identity, source_files=source_files)
    result._source_files_by_name = {
        item.relative_to(path).as_posix(): item for item in source_files
    }

    required = COMMON_REQUIRED_FILES + ARM_REQUIRED_FILES[identity.arm]
    for artifact in required:
        if artifact not in result._source_files_by_name:
            _missing_file(result, artifact)

    claims_raw = _load_jsonl(result, "claims.jsonl") if (path / "claims.jsonl").exists() else []
    layer1_claims_raw = (
        _load_jsonl(result, "claim_verdicts.jsonl")
        if (path / "claim_verdicts.jsonl").exists()
        else []
    )
    layer1_evidence_raw = (
        _load_jsonl(result, "evidence_verdicts.jsonl")
        if (path / "evidence_verdicts.jsonl").exists()
        else []
    )
    layer2_raw = (
        _load_jsonl(result, "layer2_verdicts.jsonl")
        if (path / "layer2_verdicts.jsonl").exists()
        else []
    )
    evaluations_raw = (
        _load_jsonl(result, "evaluation_results.jsonl")
        if (path / "evaluation_results.jsonl").exists()
        else []
    )
    prompts_raw = (
        _load_jsonl(result, "prompts.jsonl") if (path / "prompts.jsonl").exists() else []
    )
    metadata = (
        _load_json(result, "experiment_metadata.json")
        if (path / "experiment_metadata.json").exists()
        else {}
    )
    metrics = _load_json(result, "metrics.json") if (path / "metrics.json").exists() else {}

    arm_a_metrics: dict[str, Any] = {}
    arm_b_metrics: dict[str, Any] = {}
    arm_c_recall: dict[str, Any] = {}
    synthetic_corruption: dict[str, Any] = {}
    if identity.arm == "ab":
        if (path / "arm_a_metrics.json").exists():
            arm_a_metrics = _load_json(result, "arm_a_metrics.json")
        if (path / "arm_b_metrics.json").exists():
            arm_b_metrics = _load_json(result, "arm_b_metrics.json")
        if (path / "preregistration_comparison.json").exists():
            _load_json(result, "preregistration_comparison.json")
    else:
        if (path / "arm_c_recall.json").exists():
            arm_c_recall = _load_json(result, "arm_c_recall.json")
        if (path / "synthetic_corruption.json").exists():
            synthetic_corruption = _load_json(result, "synthetic_corruption.json")

    result.tables["runs"].append(
        {
            "run_id": identity.run_id,
            "source_run_dir": path.as_posix(),
            "dataset": identity.dataset,
            "arm": identity.arm,
            "temperature_code": identity.temperature_code,
            "temperature": identity.temperature,
            "repeat": identity.repeat,
            "task_type": metadata.get("task_type"),
            "target_column": metadata.get("target_column"),
            "source_dataset_path": metadata.get("source_dataset_path"),
            "dataset_path": metadata.get("dataset_path"),
            "preregistration_path": metadata.get("preregistration_path"),
            "preregistration_sha256": metadata.get("preregistration_sha256"),
            "call_cap_used": metadata.get("call_cap_used"),
            "reported_arms": metadata.get("reported_arms"),
            "planted_columns": metadata.get("planted_columns"),
            "manual_dataset_corrections_json": _json_text(
                metadata.get("manual_dataset_corrections") or []
            ),
        }
    )

    claim_ids, normalized_claims = _normalize_claims(result, claims_raw)
    result.tables["claims"].extend(normalized_claims)
    result.tables["layer1_claim_verdicts"].extend(
        _normalize_layer1_claims(result, layer1_claims_raw)
    )
    result.tables["layer1_evidence_verdicts"].extend(
        _normalize_layer1_evidence(result, layer1_evidence_raw)
    )
    result.tables["layer2_verdicts"].extend(_normalize_layer2(result, layer2_raw))

    _validate_claim_references(result, claim_ids, "claim_verdicts.jsonl", layer1_claims_raw)
    _validate_claim_references(
        result, claim_ids, "evidence_verdicts.jsonl", layer1_evidence_raw
    )
    _validate_claim_references(result, claim_ids, "layer2_verdicts.jsonl", layer2_raw)
    _validate_layer2_gate(result, claim_ids, layer1_claims_raw, layer2_raw)

    primary_arm = "A" if identity.arm == "ab" else "C"
    primary_metrics = arm_a_metrics or metrics
    primary_source = "arm_a_metrics.json" if arm_a_metrics else "metrics.json"
    if primary_metrics:
        result.tables["run_metrics"].append(
            _normalize_metrics(identity, primary_metrics, primary_arm, primary_source)
        )
    if arm_b_metrics:
        result.tables["run_metrics"].append(
            _normalize_metrics(identity, arm_b_metrics, "B", "arm_b_metrics.json")
        )

    _normalize_evaluations(result, evaluations_raw)
    _normalize_prompts(result, prompts_raw)
    if arm_c_recall and synthetic_corruption:
        _normalize_arm_c(result, arm_c_recall, synthetic_corruption)
    return result
