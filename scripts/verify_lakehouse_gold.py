"""Reproduce the Databricks gold-table logic locally from a lakehouse export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_export(export_dir: Path) -> dict[str, pd.DataFrame]:
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        name: pd.read_parquet(export_dir / details["path"])
        for name, details in manifest["tables"].items()
    }


def build_claims_enriched(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    claims = tables["claims"].copy()
    layer1 = tables["layer1_claim_verdicts"][
        ["run_id", "claim_id", "claim_layer1_code"]
    ]
    layer2 = tables["layer2_verdicts"][
        ["run_id", "claim_id", "layer2_score"]
    ]
    return claims.merge(layer1, on=["run_id", "claim_id"], how="left").merge(
        layer2, on=["run_id", "claim_id"], how="left"
    )


def gold_layer1_rates(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = (
        tables["layer1_evidence_verdicts"]
        .merge(
            tables["claims"][["run_id", "claim_id", "stage"]],
            on=["run_id", "claim_id"],
        )
        .merge(tables["runs"][["run_id", "dataset"]], on="run_id")
    )
    rows = rows.assign(
        accurate=rows["layer1_code"].eq("EVIDENCE_ACCURATE").astype(int),
        misstated=rows["layer1_code"].eq("EVIDENCE_MISSTATED").astype(int),
        fabricated=rows["layer1_code"].eq("EVIDENCE_FABRICATED").astype(int),
    )
    grouped = (
        rows.groupby(["dataset", "stage"], as_index=False)
        .agg(
            evidence_count=("layer1_code", "size"),
            accurate_count=("accurate", "sum"),
            misstated_count=("misstated", "sum"),
            fabricated_count=("fabricated", "sum"),
        )
        .assign(
            accuracy_rate=lambda frame: frame.accurate_count / frame.evidence_count,
            misstatement_rate=lambda frame: frame.misstated_count
            / frame.evidence_count,
        )
    )
    return grouped


def gold_layer2_scores(
    tables: dict[str, pd.DataFrame], claims_enriched: pd.DataFrame
) -> pd.DataFrame:
    rows = claims_enriched.merge(
        tables["runs"][["run_id", "dataset", "temperature"]], on="run_id"
    )
    rows = rows.assign(
        eligible=rows["claim_layer1_code"].eq("EVIDENCE_ACCURATE"),
        scored=rows["layer2_score"].notna(),
        eligible_missing=lambda frame: frame.eligible & ~frame.scored,
        ineligible_scored=lambda frame: ~frame.eligible & frame.scored,
        misstated_excluded=(
            rows["claim_layer1_code"].eq("EVIDENCE_MISSTATED")
            & rows["layer2_score"].isna()
        ),
    )
    for score in range(4):
        rows[f"score_{score}"] = rows["layer2_score"].eq(score).astype(int)

    return rows.groupby(
        ["dataset", "stage", "temperature"], as_index=False
    ).agg(
        claim_count=("claim_id", "size"),
        layer1_eligible_count=("eligible", "sum"),
        layer2_claim_count=("scored", "sum"),
        mean_layer2_score=("layer2_score", "mean"),
        score_0_count=("score_0", "sum"),
        score_1_count=("score_1", "sum"),
        score_2_count=("score_2", "sum"),
        score_3_count=("score_3", "sum"),
        eligible_missing_layer2_count=("eligible_missing", "sum"),
        ineligible_with_layer2_count=("ineligible_scored", "sum"),
        evidence_misstated_excluded_count=("misstated_excluded", "sum"),
    )


def gold_arm_c_detection(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = tables["arm_c_defects"].merge(
        tables["runs"][["run_id", "dataset"]], on="run_id"
    )
    grouped = rows.groupby(["dataset", "defect"], as_index=False).agg(
        planted_count=("recalled", "size"),
        detected_count=("recalled", "sum"),
    )
    misses = (
        rows.loc[~rows["recalled"].astype(bool)]
        .groupby(["dataset", "defect"])["column_name"]
        .agg(lambda values: sorted(set(values)))
    )
    grouped["missed_columns"] = [
        misses.get((row.dataset, row.defect), []) for row in grouped.itertuples()
    ]
    grouped["detection_rate"] = grouped.detected_count / grouped.planted_count
    return grouped


def gold_decision_consistency(
    tables: dict[str, pd.DataFrame], claims_enriched: pd.DataFrame
) -> pd.DataFrame:
    rows = claims_enriched.merge(
        tables["runs"][["run_id", "dataset", "temperature"]], on="run_id"
    )
    rows = rows.loc[
        rows["temperature"].eq(0.0)
        & rows["decision_key"].notna()
        & rows["layer2_score"].notna()
    ]
    per_run = (
        rows.groupby(["dataset", "decision_key", "run_id"], as_index=False)[
            "layer2_score"
        ]
        .max()
        .sort_values("run_id")
    )
    per_run["score_by_run_entry"] = per_run.apply(
        lambda row: {
            "run_id": row["run_id"],
            "layer2_score": int(row["layer2_score"]),
        },
        axis=1,
    )
    grouped = per_run.groupby(["dataset", "decision_key"], as_index=False).agg(
        run_count=("run_id", "size"),
        score_by_run=("score_by_run_entry", list),
        layer2_scores=("layer2_score", lambda scores: sorted(map(int, scores))),
        min_layer2_score=("layer2_score", "min"),
        max_layer2_score=("layer2_score", "max"),
    )
    grouped = grouped.loc[grouped.run_count >= 2].copy()
    grouped["score_spread"] = (
        grouped.max_layer2_score - grouped.min_layer2_score
    )
    return grouped.sort_values(
        ["score_spread", "dataset", "decision_key"],
        ascending=[False, True, True],
    )


def ab_decision_scores(
    tables: dict[str, pd.DataFrame],
    claims_enriched: pd.DataFrame,
    *,
    dataset: str,
    decision_key: str,
    expected_scores: list[int],
) -> pd.DataFrame:
    """Return one decision's scores across exactly the four AB runs."""
    rows = claims_enriched.merge(
        tables["runs"][["run_id", "dataset", "arm"]], on="run_id"
    )
    rows = rows.loc[
        rows["dataset"].eq(dataset)
        & rows["arm"].eq("ab")
        & rows["decision_key"].eq(decision_key)
        & rows["layer2_score"].notna()
    ].sort_values("run_id")
    score_by_run = [
        {"run_id": row.run_id, "layer2_score": int(row.layer2_score)}
        for row in rows.itertuples()
    ]
    scores = sorted(item["layer2_score"] for item in score_by_run)
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "decision_key": decision_key,
                "run_count": len(score_by_run),
                "score_by_run": score_by_run,
                "layer2_scores": scores,
                "expected_layer2_scores": expected_scores,
                "matches_expected": scores == expected_scores,
            }
        ]
    )


def gold_model_metrics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    selected = tables["run_metrics"].loc[
        tables["run_metrics"]["reported_arm"].ne("B")
        & tables["run_metrics"]["best_model_id"].notna()
    ]
    evaluations = tables["model_evaluations"][[
        "run_id",
        "model_id",
        "primary_metric",
        "primary_metric_mean",
    ]]
    rows = (
        selected.merge(tables["runs"], on="run_id")
        .merge(
            evaluations,
            left_on=["run_id", "best_model_id"],
            right_on=["run_id", "model_id"],
            how="left",
        )
        .rename(
            columns={
                "best_model_id": "selected_model",
                "primary_metric_mean": "primary_metric_value",
            }
        )
    )

    evidence = tables["evidence_refs"].merge(
        tables["runs"][["run_id", "dataset", "target_column"]], on="run_id"
    )
    expected_key = "col:" + evidence["target_column"] + ":mean"
    base = evidence.loc[
        evidence["dataset"].eq("home_credit")
        & evidence["metric_key"].eq(expected_key),
        ["run_id", "claimed_value_json"],
    ].copy()
    base["positive_class_base_rate"] = base["claimed_value_json"].map(
        lambda value: float(json.loads(value))
    )
    base = base.groupby("run_id", as_index=False)["positive_class_base_rate"].max()
    rows = rows.merge(base, on="run_id", how="left")
    return rows[[
        "run_id",
        "dataset",
        "arm",
        "temperature",
        "repeat",
        "selected_model",
        "primary_metric",
        "primary_metric_value",
        "positive_class_base_rate",
    ]]


def print_query_results(export_dir: Path) -> None:
    tables = load_export(export_dir)
    claims_enriched = build_claims_enriched(tables)
    layer1 = gold_layer1_rates(tables)
    layer2 = gold_layer2_scores(tables, claims_enriched)
    arm_c = gold_arm_c_detection(tables)
    consistency = gold_decision_consistency(tables, claims_enriched)
    models = gold_model_metrics(tables)

    print("Gold table row counts")
    print(
        pd.DataFrame(
            {
                "table": [
                    "gold_layer1_rates",
                    "gold_layer2_scores",
                    "gold_arm_c_detection",
                    "gold_decision_consistency",
                    "gold_model_metrics",
                ],
                "rows": [len(layer1), len(layer2), len(arm_c), len(consistency), len(models)],
            }
        ).to_string(index=False)
    )

    print("\nQuery 1: planted-defect detection")
    defect_query = arm_c.copy()
    defect_query["all_detected"] = arm_c.detected_count.sum()
    defect_query["all_planted"] = arm_c.planted_count.sum()
    print(defect_query.to_string(index=False))

    print("\nQuery 2: Ames Housing KNN rejection across four AB runs")
    ames_knn = ab_decision_scores(
        tables,
        claims_enriched,
        dataset="ames_housing",
        decision_key="model_selection|knn|reject",
        expected_scores=[1, 1, 3, 3],
    )
    print(ames_knn.to_string(index=False))

    print("\nQuery 3: Adult Income KNN rejection across four AB runs")
    adult_income_knn = ab_decision_scores(
        tables,
        claims_enriched,
        dataset="adult_income",
        decision_key="model_selection|knn|reject",
        expected_scores=[2, 3, 3, 3],
    )
    print(adult_income_knn.to_string(index=False))

    print("\nQuery 4: Layer-2 coverage gate")
    gate_columns = [
        "layer1_eligible_count",
        "layer2_claim_count",
        "eligible_missing_layer2_count",
        "ineligible_with_layer2_count",
        "evidence_misstated_excluded_count",
    ]
    print(layer2[gate_columns].sum().to_frame().T.to_string(index=False))

    print("\nQuery 5: Home Credit PR-AUC and baseline")
    home_credit = models.loc[models.dataset.eq("home_credit")].copy()
    home_credit["absolute_lift"] = (
        home_credit.primary_metric_value - home_credit.positive_class_base_rate
    )
    print(
        home_credit[[
            "run_id",
            "selected_model",
            "primary_metric",
            "primary_metric_value",
            "positive_class_base_rate",
            "absolute_lift",
        ]].sort_values("run_id").to_string(index=False)
    )

    print("\nQuery 6: mean Layer-2 score by stage and temperature")
    stage_temperature = layer2.groupby(["stage", "temperature"], as_index=False).agg(
        scored_claims=("layer2_claim_count", "sum"),
        weighted_score_total=(
            "mean_layer2_score",
            lambda means: sum(
                mean * count
                for mean, count in zip(
                    means,
                    layer2.loc[means.index, "layer2_claim_count"],
                    strict=True,
                )
            ),
        ),
        score_0_count=("score_0_count", "sum"),
        score_1_count=("score_1_count", "sum"),
        score_2_count=("score_2_count", "sum"),
        score_3_count=("score_3_count", "sum"),
    )
    stage_temperature["mean_layer2_score"] = (
        stage_temperature.weighted_score_total / stage_temperature.scored_claims
    )
    print(
        stage_temperature.drop(columns="weighted_score_total").to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Databricks gold queries from a local export."
    )
    parser.add_argument(
        "--export-dir", type=Path, default=Path("lakehouse_export")
    )
    args = parser.parse_args()
    print_query_results(args.export_dir)


if __name__ == "__main__":
    main()
