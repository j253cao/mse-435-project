"""
Policy validation for healthcare scheduling approaches (d, e, f).

Validates the scheduling methodology by comparing:
  d. Use certain admin. time for extenuating circumstances
  e. Overbook, taking no shows into account
  f. Account for uncertainty in examination durations

Complements policy_validation.py which covers (a) single room, (b) cluster, (c) blocked days.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from clinic_schedule_part3_column_generation import (
    ROOM_TO_IDX,
    blocked_intervals_for_date,
    is_blocked,
    load_appointments,
    load_room_assignments,
    parse_distance_matrix,
    room_preference_penalty,
)


def load_schedule(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date", "start", "end"])
    df["day_key"] = df["day_key"].astype(str)
    df["weekday"] = df["date"].dt.day_name()
    return df


def compute_metrics(
    schedule_df: pd.DataFrame, room_map: dict, dist: np.ndarray
) -> Dict[str, float]:
    switch_cost = 0.0
    pref_pen = 0.0
    if schedule_df.empty:
        return {"room_switch_distance": 0.0, "preference_penalty": 0.0}

    for provider, g in schedule_df.groupby("provider"):
        g = g.sort_values(["start", "end", "appt_id"]).reset_index(drop=True)
        for i, row in g.iterrows():
            pref_pen += room_preference_penalty(
                provider, row["date"], row["start"], row["room"], room_map, dist
            )
            if i > 0 and g.loc[i, "day_key"] == g.loc[i - 1, "day_key"]:
                switch_cost += dist[
                    ROOM_TO_IDX[g.loc[i - 1, "room"]], ROOM_TO_IDX[row["room"]]
                ]
    return {"room_switch_distance": float(switch_cost), "preference_penalty": float(pref_pen)}

def admin_intervals_for_date(dt: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    return blocked_intervals_for_date(dt)


def count_appts_in_admin(schedule_df: pd.DataFrame) -> int:
    count = 0
    for _, row in schedule_df.iterrows():
        start = pd.Timestamp(row["start"])
        dur_slots = int(row["dur_slots"])
        for i in range(dur_slots):
            t = start + pd.Timedelta(minutes=15 * i)
            if is_blocked(t):
                count += 1
                break
    return count


def apply_admin_time_policy(
    schedule_df: pd.DataFrame,
    room_map: dict,
    dist: np.ndarray,
    max_admin_slots_per_day: int = 4,
) -> Tuple[pd.DataFrame, Dict[str, object]]:

    out = schedule_df.copy()
    admin_capacity_used = 0
    admin_appointments_placed = 0
    provider_day_counts: Dict[str, int] = {}

    for (provider, day_key), g in out.groupby(["provider", "day_key"]):
        n = len(g)
        provider_day_counts[f"{provider}|{day_key}"] = n

    # Identify congested provider-days (above median)
    counts = list(provider_day_counts.values())
    median_count = float(np.median(counts)) if counts else 0
    congested = [k for k, v in provider_day_counts.items() if v >= max(median_count, 5)]

    info = {
        "admin_capacity_slots_available": len(congested) * max_admin_slots_per_day,
        "congested_provider_days": len(congested),
        "median_appts_per_provider_day": median_count,
        "admin_time_policy_note": (
            "Morning admin (9-9:30 Mon-Thu, 8-8:30 Fri) can absorb overflow. "
            "Metric: potential slots if policy is applied."
        ),
    }

    return out, info


def evaluate_admin_time_policy(
    schedule_df: pd.DataFrame, room_map: dict, dist: np.ndarray
) -> Dict[str, object]:
    """Evaluate policy d metrics."""
    _, info = apply_admin_time_policy(schedule_df, room_map, dist)
    baseline_metrics = compute_metrics(schedule_df, room_map, dist)
    appts_in_admin = count_appts_in_admin(schedule_df)
    return {
        **baseline_metrics,
        "appointments_touching_admin_in_baseline": appts_in_admin,
        **info,
    }


# ---------------------------------------------------------------------------
# Policy e: Overbook, taking no shows into account
# ---------------------------------------------------------------------------
def compute_no_show_rate(appointments_df: pd.DataFrame) -> float:
    no_show_col = "No Show Appts "
    if no_show_col not in appointments_df.columns:
        no_show_col = "No Show Appts"
    if no_show_col not in appointments_df.columns:
        return 0.10  # default assumption
    ns = (appointments_df[no_show_col].fillna("N").str.strip().str.upper() == "Y").sum()
    total = len(appointments_df)
    return float(ns / total) if total > 0 else 0.10


def apply_overbooking_policy(
    schedule_df: pd.DataFrame,
    appointments_df: pd.DataFrame,
    no_show_rate: float,
) -> Dict[str, object]:
    n_scheduled = len(schedule_df)
    if no_show_rate >= 1.0:
        no_show_rate = 0.10
    expected_shows = n_scheduled * (1 - no_show_rate)
    overbook_factor = 1.0 / (1 - no_show_rate) if no_show_rate < 1 else 1.0
    recommended_overbook = int(math.ceil(n_scheduled * (overbook_factor - 1)))
    additional_capacity = recommended_overbook

    return {
        "total_scheduled": n_scheduled,
        "no_show_rate": no_show_rate,
        "expected_attendees": expected_shows,
        "overbook_factor": overbook_factor,
        "recommended_additional_slots": additional_capacity,
        "effective_capacity_with_overbooking": n_scheduled + additional_capacity,
        "overbooking_policy_note": (
            "Overbook by 1/(1-r) to offset no-shows. "
            "e.g., 10%% no-show -> overbook ~11%% more."
        ),
    }


def evaluate_overbooking_policy(
    schedule_df: pd.DataFrame, appointments_df: pd.DataFrame
) -> Dict[str, object]:
    no_show_rate = compute_no_show_rate(appointments_df)
    return apply_overbooking_policy(schedule_df, appointments_df, no_show_rate)


def apply_duration_buffer_policy(
    schedule_df: pd.DataFrame,
    room_map: dict,
    dist: np.ndarray,
    buffer_slots: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, object]]:

    out = schedule_df.copy()
    out["dur_slots_buffered"] = out["dur_slots"].astype(int) + buffer_slots
    out["end_buffered"] = out["start"] + out["dur_slots_buffered"].apply(
        lambda s: pd.Timedelta(minutes=15 * int(s))
    )

    overlaps = 0
    for _, g in out.groupby(["provider", "day_key", "block_key"], sort=False):
        g = g.sort_values("start").reset_index(drop=True)
        for i in range(1, len(g)):
            prev_end = g.iloc[i - 1]["end_buffered"]
            curr_start = g.iloc[i]["start"]
            if curr_start < prev_end:
                overlaps += 1

    total_chain_transitions = sum(
        max(0, len(g) - 1)
        for _, g in out.groupby(["provider", "day_key", "block_key"], sort=False)
    )

    info = {
        "buffer_slots": buffer_slots,
        "buffer_minutes": buffer_slots * 15,
        "overlaps_with_buffer": overlaps,
        "total_chain_transitions": total_chain_transitions,
        "overlap_ratio": (
            overlaps / total_chain_transitions if total_chain_transitions else 0
        ),
        "duration_uncertainty_policy_note": (
            "Adding buffer to durations simulates variability. "
            "Overlaps indicate schedule would need slack or resequencing."
        ),
    }
    return out, info


def apply_percentile_duration_policy(
    schedule_df: pd.DataFrame,
    buffer_pct: float = 0.20,
) -> Dict[str, object]:
  
    out = schedule_df.copy()
    out["dur_slots_pct"] = (out["dur_slots"].astype(float) * (1 + buffer_pct)).apply(
        lambda x: max(1, int(math.ceil(x)))
    )
    out["end_pct"] = out["start"] + out["dur_slots_pct"].apply(
        lambda s: pd.Timedelta(minutes=15 * int(s))
    )

    overlaps = 0
    for _, g in out.groupby(["provider", "day_key", "block_key"], sort=False):
        g = g.sort_values("start").reset_index(drop=True)
        for i in range(1, len(g)):
            prev_end = g.iloc[i - 1]["end_pct"]
            curr_start = g.iloc[i]["start"]
            if curr_start < prev_end:
                overlaps += 1

    total_transitions = sum(
        max(0, len(g) - 1)
        for _, g in out.groupby(["provider", "day_key", "block_key"], sort=False)
    )
    return {
        "buffer_percent": buffer_pct,
        "overlaps_with_pct_buffer": overlaps,
        "total_transitions": total_transitions,
        "overlap_ratio_pct": overlaps / total_transitions if total_transitions else 0,
    }


def evaluate_duration_uncertainty_policy(
    schedule_df: pd.DataFrame, room_map: dict, dist: np.ndarray
) -> Dict[str, object]:
    _, info_1 = apply_duration_buffer_policy(schedule_df, room_map, dist, buffer_slots=1)
    _, info_2 = apply_duration_buffer_policy(schedule_df, room_map, dist, buffer_slots=2)
    info_pct = apply_percentile_duration_policy(schedule_df, buffer_pct=0.20)
    baseline = compute_metrics(schedule_df, room_map, dist)
    return {
        **baseline,
        "buffer_1_slot": info_1,
        "buffer_2_slots": info_2,
        "buffer_20_pct": info_pct,
    }

def evaluate_week_extended(
    week_label: str,
    schedule_df: pd.DataFrame,
    appointments_df: pd.DataFrame,
    room_map: dict,
    dist: np.ndarray,
) -> Dict[str, object]:
    return {
        "week": week_label,
        "policy_design_reasoning": {
            "admin_time": (
                "Admin slots (9-9:30 Mon-Thu, 8-8:30 Fri, etc.) are normally protected. "
                "Policy d allows using them for extenuating circumstances when congested."
            ),
            "overbooking": (
                "Empirical no-show rate from data. Overbook by 1/(1-r) to fill capacity "
                "that would otherwise be lost to no-shows."
            ),
            "duration_uncertainty": (
                "Examination durations vary. Add buffer (fixed slots or %) to nominal duration; "
                "overlaps indicate need for slack or resequencing under uncertainty."
            ),
        },
        "baseline": compute_metrics(schedule_df, room_map, dist),
        "policy_d_admin_time": evaluate_admin_time_policy(schedule_df, room_map, dist),
        "policy_e_overbooking": evaluate_overbooking_policy(schedule_df, appointments_df),
        "policy_f_duration_uncertainty": evaluate_duration_uncertainty_policy(
            schedule_df, room_map, dist
        ),
    }


def flatten_for_table(summary: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for wk in ("Week1", "Week2"):
        if wk not in summary:
            continue
        wd = summary[wk]
        base = wd.get("baseline", {})
        d = wd.get("policy_d_admin_time", {})
        e = wd.get("policy_e_overbooking", {})
        f = wd.get("policy_f_duration_uncertainty", {})

        rows.append({
            "week": wk,
            "policy": "baseline",
            "room_switch_distance": base.get("room_switch_distance"),
            "preference_penalty": base.get("preference_penalty"),
        })
        rows.append({
            "week": wk,
            "policy": "policy_d_admin_time",
            "room_switch_distance": d.get("room_switch_distance"),
            "preference_penalty": d.get("preference_penalty"),
            "appointments_touching_admin": d.get("appointments_touching_admin_in_baseline"),
            "congested_provider_days": d.get("congested_provider_days"),
        })
        rows.append({
            "week": wk,
            "policy": "policy_e_overbooking",
            "total_scheduled": e.get("total_scheduled"),
            "no_show_rate": e.get("no_show_rate"),
            "recommended_additional_slots": e.get("recommended_additional_slots"),
        })
        rows.append({
            "week": wk,
            "policy": "policy_f_duration_uncertainty",
            "room_switch_distance": f.get("room_switch_distance"),
            "overlaps_buffer_1": f.get("buffer_1_slot", {}).get("overlaps_with_buffer"),
            "overlaps_buffer_2": f.get("buffer_2_slots", {}).get("overlaps_with_buffer"),
        })
    return pd.DataFrame(rows)


def find_latest_run(base: Path) -> Path:
    outputs = base / "outputs"
    if not outputs.exists():
        raise FileNotFoundError(f"Outputs folder not found: {outputs}")
    runs = sorted(outputs.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError("No run_* folders found in outputs/")
    return runs[0]


def main() -> None:
    base = Path(__file__).resolve().parent
    inputs_dir = base / "inputs"
    run_dir = base / "outputs" / "run_20260320_173628"
    if not run_dir.exists():
        run_dir = find_latest_run(base)
        print(f"Using latest run: {run_dir}")

    dist = parse_distance_matrix(inputs_dir / "project435Winter2026.pdf")
    room_map_w1 = load_room_assignments(inputs_dir / "ProviderRoomAssignmentWeek1.docx")
    room_map_w2 = load_room_assignments(inputs_dir / "ProviderRoomAssignmentWeek2.docx")

    week1_appts = load_appointments(inputs_dir / "AppointmentDataWeek1.csv", "W1")
    week2_appts = load_appointments(inputs_dir / "AppointmentDataWeek2.csv", "W2")

    week1_sched = load_schedule(run_dir / "week1_schedule_output.csv")
    week2_sched = load_schedule(run_dir / "week2_schedule_output.csv")

    summary = {
        "config": {
            "policies": ["d_admin_time", "e_overbooking", "f_duration_uncertainty"],
            "source_run": str(run_dir),
        },
        "Week1": evaluate_week_extended(
            "Week1", week1_sched, week1_appts, room_map_w1, dist
        ),
        "Week2": evaluate_week_extended(
            "Week2", week2_sched, week2_appts, room_map_w2, dist
        ),
    }

    table_df = flatten_for_table(summary)
    with open(run_dir / "policy_validation_extended_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    table_df.to_csv(run_dir / "policy_validation_extended_table.csv", index=False)

    print(json.dumps(summary, indent=2, default=str))
    print("\nWrote:")
    print(f"- {run_dir / 'policy_validation_extended_summary.json'}")
    print(f"- {run_dir / 'policy_validation_extended_table.csv'}")


if __name__ == "__main__":
    main()