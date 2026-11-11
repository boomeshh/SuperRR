"""Pressure analysis for the SupeRR Selector Season 2 challenge.

Run from the directory containing the supplied CSV files:
    python pressure_analysis.py

The script deliberately keeps the pressure-state reconstruction in delivery
order and uses only vectorized pandas/numpy operations for the calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MIN_PRESSURE_BALLS = 120
MAX_LEGAL_BALLS = 120
PRESSURE_THRESHOLD = 0.60
PRIOR_BALLS = 30.0
PRIOR_OUTS = 5.0


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)


def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, validate, and return balls, matches, and player-name mapping."""
    balls = pd.read_csv(root / "ball_by_ball.csv", low_memory=False)
    matches = pd.read_csv(root / "matches.csv", low_memory=False)
    mapping = pd.read_csv(root / "competition_team_player_mapping.csv", low_memory=False)

    required_balls = {
        "match_id", "innings", "overid", "innings_ball_num", "striker_id",
        "striker_name", "bowler_id", "bowler_name", "runs", "is_four",
        "is_six", "is_wicket", "dismissed_player_id", "extras_type",
        "extras1_runs", "extras_2_runs",
    }
    missing = required_balls - set(balls.columns)
    if missing:
        raise ValueError(f"ball_by_ball.csv is missing columns: {sorted(missing)}")
    if "match_id" not in matches:
        raise ValueError("matches.csv must contain match_id.")
    if "player_id" not in mapping or "player_name" not in mapping:
        raise ValueError(
            "competition_team_player_mapping.csv must contain player_id and player_name."
        )

    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    balls = balls.merge(
        matches[
            [
                "match_id", "match_date", "stage", "knock_out", "revised_target",
                "first_inn_revised_overs", "second_inn_revised_overs",
            ]
        ],
        on="match_id",
        how="left",
        validate="many_to_one",
    )
    balls = balls.loc[balls["match_date"].dt.year.ge(2022)].copy()

    _numeric(
        balls,
        [
            "innings", "overid", "innings_ball_num", "striker_id", "bowler_id",
            "dismissed_player_id", "runs", "extras1_runs", "extras_2_runs",
            "knock_out", "revised_target", "first_inn_revised_overs",
            "second_inn_revised_overs",
        ],
    )
    for column in ("is_four", "is_six", "is_wicket"):
        balls[column] = balls[column].fillna(False).astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    balls["stage"] = balls["stage"].fillna("").astype(str)
    balls["extras_type"] = balls["extras_type"].fillna("").astype(str).str.strip().str.lower()
    balls.sort_values(
        ["match_id", "innings", "innings_ball_num", "overid"],
        kind="mergesort",
        inplace=True,
    )

    # One stable name per player ID; the delivery name remains the fallback.
    mapping["player_id"] = pd.to_numeric(mapping["player_id"], errors="coerce")
    mapping["player_name"] = mapping["player_name"].fillna("").astype(str).str.strip()
    names = (
        mapping.loc[mapping["player_id"].notna() & mapping["player_name"].ne("")]
        .groupby("player_id")["player_name"]
        .agg(lambda values: values.value_counts().index[0])
        .rename("mapped_player_name")
    )
    balls = balls.merge(names, left_on="striker_id", right_index=True, how="left")
    balls.rename(columns={"mapped_player_name": "mapped_striker_name"}, inplace=True)
    balls = balls.merge(names, left_on="bowler_id", right_index=True, how="left")
    balls.rename(columns={"mapped_player_name": "mapped_bowler_name"}, inplace=True)
    return balls, matches, names


def build_pressure_index(balls: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct pre-delivery match state and calculate the 0-1 PI."""
    group = ["match_id", "innings"]
    balls["legal_delivery"] = ~balls["extras_type"].isin({"wide", "overstep"})
    # A delivery's extras columns contain placeholder values on ordinary balls.
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
        balls.groupby(group, sort=False)["legal_delivery"].cumsum().astype(int)
        - balls["legal_delivery"].astype(int)
    )

    first_innings_totals = (
        balls.loc[balls["innings"].eq(1)]
        .groupby("match_id")["total_runs"]
        .sum()
        .rename("first_innings_total")
    )
    balls = balls.merge(first_innings_totals, on="match_id", how="left")
    target = balls["revised_target"].where(
        balls["revised_target"].gt(0),
        balls["first_innings_total"] + 1,
    )
    runs_needed = (target - balls["runs_before"]).clip(lower=0)
    # Revised-over matches get their shortened legal-ball allowance.
    max_balls = np.where(
        balls["innings"].eq(2) & balls["second_inn_revised_overs"].gt(0),
        balls["second_inn_revised_overs"] * 6,
        MAX_LEGAL_BALLS,
    )
    balls_remaining = np.maximum(max_balls - balls["legal_before"], 0)
    rrr = np.where(
        balls["innings"].eq(2) & (balls_remaining > 0),
        runs_needed / balls_remaining * 6,
        0.0,
    )
    rrr = np.clip(rrr, 0.0, 36.0)
    rrr_stress = np.clip((rrr - 8.0) / 8.0, 0.0, 1.0)
    wicket_stress = np.clip(balls["wickets_before"] / 10.0, 0.0, 1.0)
    death_overs = np.clip((balls["overid"] - 14.0) / 6.0, 0.0, 1.0)
    chase_pi = 0.5 * rrr_stress + 0.3 * wicket_stress + 0.2 * death_overs

    collapse = np.where(
        balls["overid"].between(1, 8) & balls["wickets_before"].ge(3),
        balls["wickets_before"] / 4.0,
        0.0,
    )
    death = np.where(balls["overid"].ge(16), (balls["overid"] - 15.0) / 5.0, 0.0)
    setting_pi = np.maximum(collapse * 0.7, death * 0.8)
    balls["pressure_index"] = np.where(
        balls["innings"].eq(2), chase_pi, np.where(balls["innings"].eq(1), setting_pi, 0.0)
    )
    stakes = balls["knock_out"].eq(1) | balls["stage"].isin(
        {"Final", "Semi Final", "Eliminator", "Qualifier"}
    )
    balls["pressure_index"] = np.clip(
        balls["pressure_index"] * np.where(stakes, 1.15, 1.0), 0.0, 1.0
    )
    balls["is_pressure"] = balls["pressure_index"].ge(PRESSURE_THRESHOLD)
    balls.drop(columns=["first_innings_total"], inplace=True)
    return balls


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    return (series - series.mean()) / std if std and np.isfinite(std) else pd.Series(0.0, index=series.index)


def _shrink(value: pd.Series, sample: pd.Series, prior_mean: float, prior_n: float) -> pd.Series:
    return (value * sample + prior_mean * prior_n) / (sample + prior_n)


def batter_metrics(balls: pd.DataFrame) -> pd.DataFrame:
    """Build pressure/baseline batter metrics and an Empirical-Bayes rating."""
    legal = balls["legal_delivery"]
    pressure = balls["is_pressure"] & legal
    base = ~balls["is_pressure"] & legal
    balls = balls.copy()
    balls["p"] = pressure.astype(int)
    balls["b"] = base.astype(int)
    balls["p_runs"] = np.where(pressure, balls["runs"], 0)
    balls["b_runs"] = np.where(base, balls["runs"], 0)
    balls["p_outs"] = np.where(
        pressure & balls["is_wicket"] & balls["dismissed_player_id"].eq(balls["striker_id"]),
        1,
        0,
    )
    balls["b_outs"] = np.where(
        base & balls["is_wicket"] & balls["dismissed_player_id"].eq(balls["striker_id"]),
        1,
        0,
    )
    for prefix, mask in (("p", pressure), ("b", base)):
        balls[f"{prefix}_dots"] = np.where(mask & balls["runs"].eq(0), 1, 0)
        balls[f"{prefix}_boundaries"] = np.where(
            mask & (balls["is_four"] | balls["is_six"]), 1, 0
        )
    grouped = balls.groupby(["striker_id"], dropna=False)
    result = grouped[
        ["p", "p_runs", "p_outs", "p_dots", "p_boundaries", "b", "b_runs", "b_outs"]
    ].sum().reset_index()
    names = balls.groupby("striker_id", dropna=False).agg(
        Delivery_Name=("striker_name", lambda s: s.dropna().astype(str).mode().iat[0] if not s.dropna().empty else "")
    ).reset_index()
    mapped = balls[["striker_id", "mapped_striker_name"]].drop_duplicates("striker_id")
    result = result.merge(names, on="striker_id").merge(mapped, on="striker_id", how="left")
    result["Player_Name"] = result["mapped_striker_name"].fillna(result["Delivery_Name"])
    result["P_Balls"] = result.pop("p")
    result["Base_Balls"] = result.pop("b")
    result["Pressure_Runs"] = result.pop("p_runs")
    result["Pressure_Outs"] = result.pop("p_outs")
    result["Pressure_Dots"] = result.pop("p_dots")
    result["Pressure_Boundaries"] = result.pop("p_boundaries")
    result["Base_Runs"] = result.pop("b_runs")
    result["Base_Outs"] = result.pop("b_outs")
    result["Pressure_SR"] = result["Pressure_Runs"] / result["P_Balls"].replace(0, np.nan) * 100
    result["Pressure_Dot_Pct"] = result["Pressure_Dots"] / result["P_Balls"].replace(0, np.nan) * 100
    result["Pressure_Boundary_Pct"] = result["Pressure_Boundaries"] / result["P_Balls"].replace(0, np.nan) * 100
    result["Pressure_Average"] = result["Pressure_Runs"] / result["Pressure_Outs"].clip(lower=1)
    result["Base_SR"] = result["Base_Runs"] / result["Base_Balls"].replace(0, np.nan) * 100
    # Baseline dot counts are calculated separately to keep the output compact.
    base_dots = balls.groupby("striker_id")["b_dots"].sum()
    result["Base_Dot_Pct"] = result["striker_id"].map(base_dots) / result["Base_Balls"].replace(0, np.nan) * 100
    result["Delta_SR"] = result["Pressure_SR"] - result["Base_SR"]
    result["Eligible"] = result["P_Balls"].ge(MIN_PRESSURE_BALLS)

    eligible = result["Eligible"]
    means = {
        col: result.loc[eligible, col].replace([np.inf, -np.inf], np.nan).mean()
        for col in ["Pressure_SR", "Pressure_Average", "Pressure_Boundary_Pct", "Pressure_Dot_Pct", "Delta_SR"]
    }
    result["EB_Pressure_SR"] = _shrink(result["Pressure_SR"].fillna(0), result["P_Balls"], means["Pressure_SR"], PRIOR_BALLS)
    result["EB_Pressure_Average"] = _shrink(result["Pressure_Average"], result["Pressure_Outs"], means["Pressure_Average"], PRIOR_OUTS)
    result["EB_Boundary_Pct"] = _shrink(result["Pressure_Boundary_Pct"].fillna(0), result["P_Balls"], means["Pressure_Boundary_Pct"], PRIOR_BALLS)
    result["EB_Dot_Pct"] = _shrink(result["Pressure_Dot_Pct"].fillna(0), result["P_Balls"], means["Pressure_Dot_Pct"], PRIOR_BALLS)
    result["EB_Delta_SR"] = _shrink(result["Delta_SR"].fillna(0), result["P_Balls"], means["Delta_SR"], PRIOR_BALLS)
    result["Composite_Score"] = (
        0.35 * _zscore(result.loc[eligible, "EB_Pressure_SR"]).reindex(result.index, fill_value=0)
        + 0.25 * _zscore(result.loc[eligible, "EB_Pressure_Average"]).reindex(result.index, fill_value=0)
        + 0.20 * _zscore(result.loc[eligible, "EB_Delta_SR"]).reindex(result.index, fill_value=0)
        + 0.20 * _zscore(result.loc[eligible, "EB_Boundary_Pct"]).reindex(result.index, fill_value=0)
        - 0.10 * _zscore(result.loc[eligible, "EB_Dot_Pct"]).reindex(result.index, fill_value=0)
    )
    result["Composite_Score"] = (50 + 10 * result["Composite_Score"]).clip(0, 100)
    result["Player_Type"] = "Batter"
    return result


def bowler_metrics(balls: pd.DataFrame) -> pd.DataFrame:
    """Build pressure/baseline bowler metrics and an Empirical-Bayes rating."""
    legal = balls["legal_delivery"]
    pressure = balls["is_pressure"] & legal
    base = ~balls["is_pressure"] & legal
    balls = balls.copy()
    balls["p"] = pressure.astype(int)
    balls["b"] = base.astype(int)
    balls["p_runs"] = np.where(pressure, balls["bowler_runs"], 0)
    balls["b_runs"] = np.where(base, balls["bowler_runs"], 0)
    for prefix, mask in (("p", pressure), ("b", base)):
        balls[f"{prefix}_dots"] = np.where(mask & balls["bowler_runs"].eq(0), 1, 0)
        balls[f"{prefix}_boundaries"] = np.where(mask & (balls["is_four"] | balls["is_six"]), 1, 0)
        balls[f"{prefix}_wickets"] = np.where(mask & balls["is_wicket"], 1, 0)
    result = balls.groupby("bowler_id", dropna=False)[
        ["p", "p_runs", "p_dots", "p_boundaries", "p_wickets", "b", "b_runs"]
    ].sum().reset_index()
    names = balls.groupby("bowler_id", dropna=False).agg(
        Delivery_Name=("bowler_name", lambda s: s.dropna().astype(str).mode().iat[0] if not s.dropna().empty else "")
    ).reset_index()
    mapped = balls[["bowler_id", "mapped_bowler_name"]].drop_duplicates("bowler_id")
    result = result.merge(names, on="bowler_id").merge(mapped, on="bowler_id", how="left")
    result["Player_Name"] = result["mapped_bowler_name"].fillna(result["Delivery_Name"])
    result["P_Balls"] = result.pop("p")
    result["Base_Balls"] = result.pop("b")
    result["Pressure_Runs_Conceded"] = result.pop("p_runs")
    result["Pressure_Dots"] = result.pop("p_dots")
    result["Pressure_Boundaries"] = result.pop("p_boundaries")
    result["Pressure_Wickets"] = result.pop("p_wickets")
    result["Base_Runs_Conceded"] = result.pop("b_runs")
    result["Pressure_Econ"] = result["Pressure_Runs_Conceded"] / result["P_Balls"].replace(0, np.nan) * 6
    result["Pressure_Dot_Pct"] = result["Pressure_Dots"] / result["P_Balls"].replace(0, np.nan) * 100
    result["Pressure_Boundary_Leak_Pct"] = result["Pressure_Boundaries"] / result["P_Balls"].replace(0, np.nan) * 100
    result["Pressure_Wicket_SR"] = result["P_Balls"] / result["Pressure_Wickets"].clip(lower=1)
    result["Base_Econ"] = result["Base_Runs_Conceded"] / result["Base_Balls"].replace(0, np.nan) * 6
    result["Delta_Econ"] = result["Base_Econ"] - result["Pressure_Econ"]
    result["Eligible"] = result["P_Balls"].ge(MIN_PRESSURE_BALLS)
    eligible = result["Eligible"]
    means = {
        col: result.loc[eligible, col].replace([np.inf, -np.inf], np.nan).mean()
        for col in ["Pressure_Econ", "Pressure_Wicket_SR", "Pressure_Dot_Pct", "Pressure_Boundary_Leak_Pct", "Delta_Econ"]
    }
    result["EB_Pressure_Econ"] = _shrink(result["Pressure_Econ"].fillna(0), result["P_Balls"], means["Pressure_Econ"], PRIOR_BALLS)
    result["EB_Wicket_SR"] = _shrink(result["Pressure_Wicket_SR"], result["Pressure_Wickets"], means["Pressure_Wicket_SR"], PRIOR_OUTS)
    result["EB_Dot_Pct"] = _shrink(result["Pressure_Dot_Pct"].fillna(0), result["P_Balls"], means["Pressure_Dot_Pct"], PRIOR_BALLS)
    result["EB_Boundary_Leak_Pct"] = _shrink(result["Pressure_Boundary_Leak_Pct"].fillna(0), result["P_Balls"], means["Pressure_Boundary_Leak_Pct"], PRIOR_BALLS)
    result["EB_Delta_Econ"] = _shrink(result["Delta_Econ"].fillna(0), result["P_Balls"], means["Delta_Econ"], PRIOR_BALLS)
    result["Composite_Score"] = (
        0.35 * _zscore(-result.loc[eligible, "EB_Pressure_Econ"]).reindex(result.index, fill_value=0)
        + 0.25 * _zscore(-result.loc[eligible, "EB_Wicket_SR"]).reindex(result.index, fill_value=0)
        + 0.20 * _zscore(result.loc[eligible, "EB_Delta_Econ"]).reindex(result.index, fill_value=0)
        + 0.20 * _zscore(result.loc[eligible, "EB_Dot_Pct"]).reindex(result.index, fill_value=0)
        - 0.10 * _zscore(result.loc[eligible, "EB_Boundary_Leak_Pct"]).reindex(result.index, fill_value=0)
    )
    result["Composite_Score"] = (50 + 10 * result["Composite_Score"]).clip(0, 100)
    result["Player_Type"] = "Bowler"
    return result


def print_table(title: str, data: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n{title}")
    print(data[columns].to_string(index=False, formatters={
        column: (lambda value: f"{value:.2f}") for column in columns if column not in {"Rank", "Player_Name", "P_Balls"}
    }))


def main() -> None:
    balls, _, _ = load_data(ROOT)
    balls = build_pressure_index(balls)
    batters = batter_metrics(balls)
    bowlers = bowler_metrics(balls)
    top_batters = batters.loc[batters["Eligible"]].sort_values(
        ["Composite_Score", "P_Balls"], ascending=[False, False]
    ).head(10).copy()
    top_bowlers = bowlers.loc[bowlers["Eligible"]].sort_values(
        ["Composite_Score", "P_Balls"], ascending=[False, False]
    ).head(10).copy()
    top_batters.insert(0, "Rank", np.arange(1, len(top_batters) + 1))
    top_bowlers.insert(0, "Rank", np.arange(1, len(top_bowlers) + 1))

    print(f"Total deliveries analyzed: {len(balls):,}")
    print(f"Pressure deliveries identified: {int(balls['is_pressure'].sum()):,}")
    print_table(
        "TOP 10 BATTERS UNDER PRESSURE",
        top_batters,
        ["Rank", "Player_Name", "P_Balls", "Pressure_SR", "Delta_SR", "Pressure_Average", "Pressure_Boundary_Pct", "Composite_Score"],
    )
    print_table(
        "TOP 10 BOWLERS UNDER PRESSURE",
        top_bowlers,
        ["Rank", "Player_Name", "P_Balls", "Pressure_Econ", "Delta_Econ", "Pressure_Wicket_SR", "Pressure_Dot_Pct", "Composite_Score"],
    )
    all_metrics = pd.concat([batters, bowlers], ignore_index=True, sort=False)
    top_batters.to_csv(ROOT / "top_10_batters_pressure.csv", index=False)
    top_bowlers.to_csv(ROOT / "top_10_bowlers_pressure.csv", index=False)
    all_metrics.to_csv(ROOT / "full_player_pressure_metrics.csv", index=False)


if __name__ == "__main__":
    main()
