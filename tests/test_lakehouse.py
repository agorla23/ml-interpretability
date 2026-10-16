import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mlinterp.lakehouse.export import export_runs
from mlinterp.lakehouse.run_parser import (
    discover_run_directories,
    parse_run_directory,
    parse_run_name,
)


FIXTURE_RUNS = Path(__file__).parent / "fixtures" / "runs"
HAPPY_RUN = FIXTURE_RUNS / "m9_tiny_c_t00_r1"
BROKEN_RUN = FIXTURE_RUNS / "m9_broken_ab_t00_r1"


def test_parse_run_name() -> None:
    parsed = parse_run_name("m9_adult_income_ab_t07_r3")

    assert parsed.dataset == "adult_income"
    assert parsed.arm == "ab"
    assert parsed.temperature_code == "t07"
    assert parsed.temperature == 0.7
    assert parsed.repeat == 3

    with pytest.raises(ValueError, match="Not a Milestone 9"):
        parse_run_name("m9_adult_income_ab_t10_r1_verify")


def test_discovery_includes_only_exact_run_directories(tmp_path: Path) -> None:
    for name in (
        "m9_tiny_ab_t00_r1",
        "m9_tiny_c_t07_r2",
        "m9_tiny_ab_t00_r1_verify",
        "layer2_fake",
    ):
        (tmp_path / name).mkdir()
    (tmp_path / "loose.jsonl").write_text("{}\n", encoding="utf-8")

    assert [path.name for path in discover_run_directories(tmp_path)] == [
        "m9_tiny_ab_t00_r1",
        "m9_tiny_c_t07_r2",
    ]


def test_happy_run_normalizes_records_and_structured_decision_keys() -> None:
    parsed = parse_run_directory(HAPPY_RUN)

    assert parsed.tables["data_quality"] == []
    assert len(parsed.tables["claims"]) == 4
    assert len(parsed.tables["evidence_refs"]) == 4
    assert len(parsed.tables["layer1_claim_verdicts"]) == 4
    assert len(parsed.tables["layer1_evidence_verdicts"]) == 4
    assert len(parsed.tables["layer2_verdicts"]) == 4

    claims = {row["claim_id"]: row for row in parsed.tables["claims"]}
    assert claims["eda_001"]["subject_type"] == "column"
    assert claims["eda_001"]["decision_key"] == (
        "eda|SYNTH_LEAKED|flag_leakage"
    )
    assert claims["prep_001"]["decision"] == "scale_standard"
    assert claims["prep_001"]["decision_key"] == (
        "preprocess|age|scale_standard"
    )
    assert claims["model_001"]["decision_key"] == (
        "model_selection|gbt|accept"
    )
    assert claims["evaluation_001"]["decision_key"] == (
        "evaluation|roc_auc|select"
    )
    assert claims["prep_001"]["decision_text"] == "scale age"

    evidence = parsed.tables["evidence_refs"][0]
    assert evidence["claimed_value_type"] == "bool"
    assert evidence["claimed_value_json"] == "true"
    assert parsed.tables["arm_c_defects"][0]["recalled"] is True


def test_broken_run_exercises_every_data_quality_rule() -> None:
    parsed = parse_run_directory(BROKEN_RUN)
    quality = parsed.tables["data_quality"]

    assert {row["rule"] for row in quality} == {
        "missing_required_file",
        "malformed_json",
        "orphan_claim_id",
        "layer2_missing_eligible",
        "layer2_ineligible_claim",
        "unkeyable_claim",
    }
    assert len(quality) == 6

    by_rule = {row["rule"]: row for row in quality}
    assert by_rule["missing_required_file"]["artifact"] == (
        "evidence_verdicts.jsonl"
    )
    assert by_rule["malformed_json"]["record_key"] == "line:4"
    assert by_rule["orphan_claim_id"]["record_key"] == "orphan_claim"
    assert by_rule["layer2_missing_eligible"]["record_key"] == (
        "prep_eligible"
    )
    assert by_rule["layer2_ineligible_claim"]["record_key"] == (
        "model_ineligible"
    )
    assert by_rule["unkeyable_claim"]["record_key"] == "eda_unkeyable"

    claims = {row["claim_id"]: row for row in parsed.tables["claims"]}
    assert claims["eda_unkeyable"]["subject"] == "age"
    assert claims["eda_unkeyable"]["decision"] is None
    assert claims["eda_unkeyable"]["decision_key"] is None
    assert claims["eda_unkeyable"]["decision_text"] == "Narrative-only finding"


def test_export_writes_parquet_tables_and_hashed_manifest(tmp_path: Path) -> None:
    output = tmp_path / "lakehouse_export"
    manifest = export_runs(FIXTURE_RUNS, output)

    assert manifest["included_run_directories"] == [
        "m9_broken_ab_t00_r1",
        "m9_tiny_c_t00_r1",
    ]
    assert manifest["tables"]["runs"]["row_count"] == 2
    assert manifest["tables"]["claims"]["row_count"] == 7
    assert manifest["tables"]["data_quality"]["row_count"] == 6

    manifest_on_disk = json.loads((output / "manifest.json").read_text())
    assert manifest_on_disk == manifest
    for table_name, details in manifest["tables"].items():
        parquet_path = output / details["path"]
        assert parquet_path.name == f"{table_name}.parquet"
        assert pq.read_table(parquet_path).num_rows == details["row_count"]

    source = manifest["source_files"][0]
    source_path = FIXTURE_RUNS / source["path"]
    assert source["size_bytes"] == source_path.stat().st_size
    assert source["sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()


def test_export_stops_before_writing_when_secret_pattern_is_found(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "m9_tiny_ab_t00_r1"
    run_dir.mkdir(parents=True)
    (run_dir / "prompts.jsonl").write_text(
        '{"credential":"sk-ant-fixturecredential123456789"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="Possible secrets found"):
        export_runs(runs_dir, output)

    assert not output.exists()
