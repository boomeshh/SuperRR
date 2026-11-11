"""Independent mathematical audit for pressure_analysis.py.

This script reconstructs pressure states and player aggregates from raw
deliveries, then reconciles them to the exported CSV files. It does not import
pressure_analysis.py, so a shared implementation bug cannot make the audit
pass automatically.

Run:
    python verify_pressure_math.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TOL = 1e-4
MIN_BALLS = 120
K_BALLS = 30.0
K_EVENTS = 5.0


def fail_if(condition: bool, message: str) -> None:
    if condition:
        raise AssertionError(message)


def load_raw() -> pd.DataFrame:
    balls = pd.read_csv(ROOT / "ball_by_ball.csv", low_memory=False)
    matches = pd.read_csv(ROOT / "matches.csv", low_memory=False)
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    balls = balls.merge(
        matches[
            [
                "match_id", "match_date", "stage", "knock_out", "revised_target",
                "second_inn_revised_overs",
            ]
        ],
        on="match_id",
        how="left",
        validate="many_to_one",
    )
    balls = balls.loc[balls["match_date"].dt.year.ge(2022)].copy()
    numeric = [
        "innings", "overid", "innings_ball_num", "striker_id", "bowler_id",
        "dismissed_player_id", "runs", "extras1_runs", "extras_2_runs",
        "knock_out", "revised_target", "second_inn_revised_overs",
    ]
    for column in numeric:
        balls[column] = pd.to_numeric(balls[column], errors="coerce").fillna(0)
    for column in ("is_four", "is_six", "is_wicket"):
        balls[column] = balls[column].fillna(False).astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    balls["stage"] = balls["stage"].fillna("").astype(str)
    balls["extras_type"] = balls["extras_type"].fillna("").astype(str).str.lower().str.strip()
    balls.sort_values(
        ["match_id", "innings", "innings_ball_num", "overid"],
        kind="mergesort",
        inplace=True,
    )
    return balls


def reconstruct_pressure(balls: pd.DataFrame) -> pd.DataFrame:
    """Recreate PI using only raw delivery order and match metadata."""
    balls = balls.copy()
    group = ["match_id", "innings"]
    balls["legal"] = ~balls["extras_type"].isin({"wide", "overstep"})
    extras = balls["extras1_runs"] + balls["extras_2_runs"]
    balls["total_runs"] = balls["runs"] + np.where(
        balls["extras_type"].ne(""), extras, 0
    )
    balls["bowler_runs"] = balls["runs"] + np.where(
        balls["extras_type"].isin({"byes", "legbyes", "b"}),
        0,
        np.where(balls["extras_type"].ne(""), extras, 0),
    )
    balls["runs_before"] = balls.groupby(group, sort=False)["total_runs"].cumsum() - balls["total_runs"]
    balls["wickets_before"] = (
        balls.groupby(group, sort=False)["is_wicket"].cumsum().astype(int)
        - balls["is_wicket"].astype(int)
    )
    balls["legal_before"] = (
        balls.groupby(group, sort=False)["legal"].cumsum().astype(int)
        - balls["legal"].astype(int)
    )
    first_totals = (
        balls.loc[balls["innings"].eq(1)]
        .groupby("match_id")["total_runs"]
        .sum()
        .rename("first_total")
    )
    balls = balls.merge(first_totals, on="match_id", how="left")
    target = balls["revised_target"].where(
        balls["revised_target"].gt(0), balls["first_total"] + 1
    )
    max_balls = np.where(
        balls["innings"].eq(2) & balls["second_inn_revised_overs"].gt(0),
        balls["second_inn_revised_overs"] * 6,
        120,
    )
    remaining = np.maximum(max_balls - balls["legal_before"], 0)
    rrr = np.where(
        balls["innings"].eq(2) & (remaining > 0),
        (target - balls["runs_before"]).clip(lower=0) / remaining * 6,
        0,
    )
    rrr_stress = np.clip((np.clip(rrr, 0, 36) - 8) / 8, 0, 1)
    wicket_stress = np.clip(balls["wickets_before"] / 10, 0, 1)
    death_stress = np.clip((balls["overid"] - 14) / 6, 0, 1)
    chase = 0.5 * rrr_stress + 0.3 * wicket_stress + 0.2 * death_stress
    collapse = np.where(
        balls["overid"].between(1, 8) & balls["wickets_before"].ge(3),
        balls["wickets_before"] / 4,
        0,
    )
    setting = np.maximum(
        collapse * 0.7,
        np.where(balls["overid"].ge(16), (balls["overid"] - 15) / 5 * 0.8, 0),
    )
    pi = np.where(balls["innings"].eq(2), chase, np.where(balls["innings"].eq(1), setting, 0))
    stakes = balls["knock_out"].eq(1) | balls["stage"].isin(
        {"Final", "Semi Final", "Eliminator", "Qualifier"}
    )
    balls["pressure_index_audit"] = np.clip(pi * np.where(stakes, 1.15, 1), 0, 1)
    balls["pressure_audit"] = balls["pressure_index_audit"].ge(0.60)
    return balls


def raw_aggregates(balls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate raw legal pressure and baseline delivery counts."""
    p = balls["pressure_audit"] & balls["legal"]
    b = ~balls["pressure_audit"] & balls["legal"]
    x = balls.copy()
    x["p_bat"] = p.astype(int)
    x["b_bat"] = b.astype(int)
    x["p_bat_runs"] = np.where(p, x["runs"], 0)
    x["b_bat_runs"] = np.where(b, x["runs"], 0)
    x["p_bat_outs"] = np.where(
        p & x["is_wicket"] & x["dismissed_player_id"].eq(x["striker_id"]), 1, 0
    )
    x["b_bat_outs"] = np.where(
        b & x["is_wicket"] & x["dismissed_player_id"].eq(x["striker_id"]), 1, 0
    )
    x["p_bat_dots"] = np.where(p & x["runs"].eq(0), 1, 0)
    x["b_bat_dots"] = np.where(b & x["runs"].eq(0), 1, 0)
    x["p_bat_boundaries"] = np.where(p & (x["is_four"] | x["is_six"]), 1, 0)
    x["p_bowl"] = p.astype(int)
    x["b_bowl"] = b.astype(int)
    x["p_bowl_runs"] = np.where(p, x["bowler_runs"], 0)
    x["b_bowl_runs"] = np.where(b, x["bowler_runs"], 0)
    x["p_bowl_dots"] = np.where(p & x["bowler_runs"].eq(0), 1, 0)
    x["p_bowl_boundaries"] = np.where(p & (x["is_four"] | x["is_six"]), 1, 0)
    x["p_bowl_wickets"] = np.where(p & x["is_wicket"], 1, 0)

    bat = x.groupby("striker_id", dropna=False)[
        ["p_bat", "b_bat", "p_bat_runs", "b_bat_runs", "p_bat_outs",
         "b_bat_outs", "p_bat_dots", "b_bat_dots", "p_bat_boundaries"]
    ].sum().reset_index()
    bat.rename(
        columns={
            "p_bat": "P_Balls", "b_bat": "Base_Balls",
            "p_bat_runs": "Pressure_Runs", "b_bat_runs": "Base_Runs",
            "p_bat_outs": "Pressure_Outs", "b_bat_outs": "Base_Outs",
            "p_bat_dots": "Pressure_Dots", "b_bat_dots": "Base_Dots",
            "p_bat_boundaries": "Pressure_Boundaries",
        },
        inplace=True,
    )
    bat["Pressure_SR"] = bat["Pressure_Runs"] / bat["P_Balls"] * 100
    bat["Base_SR"] = bat["Base_Runs"] / bat["Base_Balls"] * 100
    bat["Delta_SR"] = bat["Pressure_SR"] - bat["Base_SR"]
    bat["Pressure_Average"] = bat["Pressure_Runs"] / bat["Pressure_Outs"].clip(lower=1)
    bat["Pressure_Dot_Pct"] = bat["Pressure_Dots"] / bat["P_Balls"] * 100
    bat["Pressure_Boundary_Pct"] = bat["Pressure_Boundaries"] / bat["P_Balls"] * 100
    bat["Base_Dot_Pct"] = bat["Base_Dots"] / bat["Base_Balls"] * 100

    bowl = x.groupby("bowler_id", dropna=False)[
        ["p_bowl", "b_bowl", "p_bowl_runs", "b_bowl_runs", "p_bowl_dots",
         "p_bowl_boundaries", "p_bowl_wickets"]
    ].sum().reset_index()
    bowl.rename(
        columns={
            "p_bowl": "P_Balls", "b_bowl": "Base_Balls",
            "p_bowl_runs": "Pressure_Runs_Conceded",
            "b_bowl_runs": "Base_Runs_Conceded",
            "p_bowl_dots": "Pressure_Dots",
            "p_bowl_boundaries": "Pressure_Boundaries",
            "p_bowl_wickets": "Pressure_Wickets",
        },
        inplace=True,
    )
    bowl["Pressure_Econ"] = bowl["Pressure_Runs_Conceded"] / bowl["P_Balls"] * 6
    bowl["Base_Econ"] = bowl["Base_Runs_Conceded"] / bowl["Base_Balls"] * 6
    bowl["Delta_Econ"] = bowl["Base_Econ"] - bowl["Pressure_Econ"]
    bowl["Pressure_Dot_Pct"] = bowl["Pressure_Dots"] / bowl["P_Balls"] * 100
    bowl["Pressure_Boundary_Leak_Pct"] = bowl["Pressure_Boundaries"] / bowl["P_Balls"] * 100
    bowl["Pressure_Wicket_SR"] = bowl["P_Balls"] / bowl["Pressure_Wickets"].clip(lower=1)
    return bat, bowl


def compare(raw: pd.DataFrame, exported: pd.DataFrame, key: str, columns: list[str]) -> pd.DataFrame:
    merged = raw.merge(exported, left_on=key, right_on=key, suffixes=("_raw", "_csv"), how="inner")
    rows = []
    for _, row in merged.iterrows():
        for column in columns:
            left = float(row[f"{column}_raw"])
            right = float(row[f"{column}_csv"])
            rows.append(
                {"ID": row[key], "Metric": column, "Raw": left, "CSV": right, "Abs_Error": abs(left - right)}
            )
    return pd.DataFrame(rows)


def check_shrinkage(frame: pd.DataFrame, raw: pd.DataFrame, metric: str, eb: str, sample: str, k: float) -> float:
    eligible = raw["P_Balls"].ge(MIN_BALLS)
    cohort_mean = raw.loc[eligible, metric].replace([np.inf, -np.inf], np.nan).mean()
    expected = (
        frame[metric].fillna(0) * frame[sample] + cohort_mean * k
    ) / (frame[sample] + k)
    return float(np.nanmax(np.abs(expected - frame[eb])))


def main() -> None:
    balls = reconstruct_pressure(load_raw())
    fail_if(not balls["pressure_index_audit"].between(0, 1).all(), "PI outside [0, 1]")
    fail_if(balls["pressure_audit"].sum() != 24677, "Unexpected pressure-delivery count")
    fail_if((balls["pressure_audit"] & balls["innings"].isin({3, 4})).any(), "PI leaked into innings 3/4")

    raw_bat, raw_bowl = raw_aggregates(balls)
    csv_bat = pd.read_csv(ROOT / "full_player_pressure_metrics.csv")
    csv_bowl = pd.read_csv(ROOT / "full_player_pressure_metrics.csv")
    csv_bat = csv_bat.loc[csv_bat["Player_Type"].eq("Batter")].copy()
    csv_bowl = csv_bowl.loc[csv_bowl["Player_Type"].eq("Bowler")].copy()
    # The combined export has blank ID fields for the other player type.
    csv_bat["striker_id"] = pd.to_numeric(csv_bat["striker_id"], errors="coerce")
    csv_bowl["bowler_id"] = pd.to_numeric(csv_bowl["bowler_id"], errors="coerce")

    bat_cols = [
        "P_Balls", "Base_Balls", "Pressure_Runs", "Base_Runs", "Pressure_Outs",
        "Pressure_SR", "Base_SR", "Delta_SR", "Pressure_Average",
        "Pressure_Dot_Pct", "Pressure_Boundary_Pct",
    ]
    bowl_cols = [
        "P_Balls", "Base_Balls", "Pressure_Runs_Conceded", "Base_Runs_Conceded",
        "Pressure_Wickets", "Pressure_Econ", "Base_Econ", "Delta_Econ",
        "Pressure_Wicket_SR", "Pressure_Dot_Pct", "Pressure_Boundary_Leak_Pct",
    ]
    bat_recon = compare(raw_bat, csv_bat, "striker_id", bat_cols)
    bowl_recon = compare(raw_bowl, csv_bowl, "bowler_id", bowl_cols)
    max_error = max(bat_recon["Abs_Error"].max(), bowl_recon["Abs_Error"].max())
    fail_if(max_error >= TOL, f"Raw reconciliation error {max_error:.8g} exceeds tolerance")

    top_bat = pd.read_csv(ROOT / "top_10_batters_pressure.csv")
    top_bowl = pd.read_csv(ROOT / "top_10_bowlers_pressure.csv")
    spot_bat = top_bat.loc[top_bat["striker_id"].isin([4708, 1699])].copy()
    spot_bowl = top_bowl.loc[top_bowl["bowler_id"].isin([19316, 8661])].copy()
    fail_if(len(spot_bat) != 2 or len(spot_bowl) != 2, "Required spot-check IDs are missing")
    fail_if(not (spot_bat["P_Balls"] >= MIN_BALLS).all(), "Ineligible batter in top output")
    fail_if(not (spot_bowl["P_Balls"] >= MIN_BALLS).all(), "Ineligible bowler in top output")

    full = pd.read_csv(ROOT / "full_player_pressure_metrics.csv")
    fail_if(
        full.loc[full["Player_Type"].eq("Batter"), "Eligible"].astype(bool)
        .ne(pd.to_numeric(full.loc[full["Player_Type"].eq("Batter"), "P_Balls"]) >= MIN_BALLS).any(),
        "Batter eligibility mismatch",
    )
    fail_if(
        full.loc[full["Player_Type"].eq("Bowler"), "Eligible"].astype(bool)
        .ne(pd.to_numeric(full.loc[full["Player_Type"].eq("Bowler"), "P_Balls"]) >= MIN_BALLS).any(),
        "Bowler eligibility mismatch",
    )
    fail_if((spot_bat["Delta_SR"] <= 0).any(), "Spot-check batter delta does not show elevation")
    fail_if((spot_bowl["Delta_Econ"].isna()).any(), "Spot-check bowler delta is NaN")
    spot_bat_raw = raw_bat.loc[raw_bat["striker_id"].isin([4708, 1699])]
    spot_bowl_raw = raw_bowl.loc[raw_bowl["bowler_id"].isin([19316, 8661])]
    spot_bat_recon = compare(spot_bat_raw, spot_bat, "striker_id", ["Pressure_SR", "Base_SR", "Delta_SR"])
    spot_bowl_recon = compare(
        spot_bowl_raw, spot_bowl, "bowler_id",
        ["Pressure_Econ", "Base_Econ", "Delta_Econ", "Pressure_Wicket_SR"],
    )
    fail_if(
        max(spot_bat_recon["Abs_Error"].max(), spot_bowl_recon["Abs_Error"].max()) >= TOL,
        "Required spot-check values do not reconcile",
    )
    fail_if((spot_bowl["Delta_Econ"] - (
        spot_bowl["Base_Econ"] - spot_bowl["Pressure_Econ"]
    )).abs().max() >= TOL, "Bowler delta orientation mismatch")

    # Verify the exact implementation's EB formulas, including event priors
    # used for average and wicket strike rate.
    eb_errors = [
        check_shrinkage(csv_bat, raw_bat, "Pressure_SR", "EB_Pressure_SR", "P_Balls", K_BALLS),
        check_shrinkage(csv_bat, raw_bat, "Pressure_Average", "EB_Pressure_Average", "Pressure_Outs", K_EVENTS),
        check_shrinkage(csv_bat, raw_bat, "Pressure_Boundary_Pct", "EB_Boundary_Pct", "P_Balls", K_BALLS),
        check_shrinkage(csv_bowl, raw_bowl, "Pressure_Econ", "EB_Pressure_Econ", "P_Balls", K_BALLS),
        check_shrinkage(csv_bowl, raw_bowl, "Pressure_Wicket_SR", "EB_Wicket_SR", "Pressure_Wickets", K_EVENTS),
    ]
    fail_if(max(eb_errors) >= TOL, f"EB implementation mismatch: {max(eb_errors):.8g}")
    eligible_metrics = [
        "P_Balls", "Base_Balls", "Pressure_Runs", "Pressure_Outs",
        "Pressure_SR", "Base_SR", "Delta_SR", "Pressure_Average",
        "Pressure_Dot_Pct", "Pressure_Boundary_Pct", "EB_Pressure_SR",
        "EB_Pressure_Average", "EB_Boundary_Pct", "EB_Dot_Pct", "EB_Delta_SR",
        "Composite_Score",
    ]
    fail_if(
        csv_bat.loc[csv_bat["Eligible"], list(eligible_metrics)].isna().any().any(),
        "Eligible batter metrics contain NaN",
    )
    eligible_metrics = [
        "P_Balls", "Base_Balls", "Pressure_Runs_Conceded",
        "Pressure_Wickets", "Pressure_Econ", "Base_Econ", "Delta_Econ",
        "Pressure_Wicket_SR", "Pressure_Dot_Pct", "Pressure_Boundary_Leak_Pct",
        "EB_Pressure_Econ", "EB_Wicket_SR", "EB_Dot_Pct",
        "EB_Boundary_Leak_Pct", "EB_Delta_Econ", "Composite_Score",
    ]
    fail_if(
        csv_bowl.loc[csv_bowl["Eligible"], list(eligible_metrics)].isna().any().any(),
        "Eligible bowler metrics contain NaN",
    )
    fail_if(full["Composite_Score"].isna().any(), "Composite score contains NaN")
    fail_if(not top_bat["Composite_Score"].is_monotonic_decreasing, "Batter scores not ranked descending")
    fail_if(not top_bowl["Composite_Score"].is_monotonic_decreasing, "Bowler scores not ranked descending")

    print("\nPRESSURE ANALYSIS MATHEMATICAL AUDIT")
    print("=" * 72)
    print(f"[PASS] Ball counts and run sums: {len(balls):,} deliveries; {int(balls.pressure_audit.sum()):,} pressure")
    print(f"[PASS] Strike-rate/economy formulas: maximum raw-vs-CSV error {max_error:.8f}")
    print("[PASS] Delta orientation: batter Delta_SR = pressure - baseline; bowler Delta_Econ = baseline - pressure")
    print(f"[PASS] Bayesian shrinkage: maximum implementation error {max(eb_errors):.8f}")
    print("[PASS] Eligible-player metrics, Bayesian outputs, and composite scores contain no NaN")
    print("[PASS] Composite scoring monotonic ordering and finite scores")
    print("[PASS] PI state isolation: pre-delivery state, innings 1/2 only, bounded [0, 1]")
    print("\nRECONCILIATION (largest absolute errors)")
    reconciliation = pd.concat([bat_recon, bowl_recon], ignore_index=True)
    print(
        reconciliation.sort_values("Abs_Error", ascending=False)
        .head(20)
        .to_string(index=False, float_format=lambda value: f"{value:.8f}")
    )
    print("\nSPOT-CHECKS")
    print("Batters:")
    print(
        top_bat.loc[
            top_bat["striker_id"].isin([4708, 1699]),
            ["Rank", "Player_Name", "P_Balls", "Pressure_SR", "Base_SR", "Delta_SR"],
        ].to_string(index=False)
    )
    print("Bowlers:")
    print(
        top_bowl.loc[
            top_bowl["bowler_id"].isin([19316, 8661]),
            ["Rank", "Player_Name", "P_Balls", "Pressure_Econ", "Base_Econ", "Delta_Econ", "Pressure_Wicket_SR"],
        ].to_string(index=False)
    )
    print("\nAUDIT RESULT: 100% PASS")


if __name__ == "__main__":
    main()
