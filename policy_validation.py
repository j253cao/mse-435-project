from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from clinic_schedule_part3_column_generation import (
    ROOMS,
    ROOM_TO_IDX,
    load_room_assignments,
    parse_distance_matrix,
    room_preference_penalty,
)

CLUSTER_THRESHOLD = 2.0
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def load_schedule(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date", "start", "end"])
    df["day_key"] = df["day_key"].astype(str)
    df["weekday"] = df["date"].dt.day_name()
    return df


def compute_metrics(schedule_df: pd.DataFrame, room_map: dict, dist) -> Dict[str, float]:
    switch_cost = 0.0
    pref_pen = 0.0
    if schedule_df.empty:
        return {"room_switch_distance": 0.0, "preference_penalty": 0.0}

    for provider, g in schedule_df.groupby("provider"):
        g = g.sort_values(["start", "end", "appt_id"]).reset_index(drop=True)
        for i, row in g.iterrows():
            pref_pen += room_preference_penalty(provider, row["date"], row["start"], row["room"], room_map, dist)
            if i > 0 and g.loc[i, "day_key"] == g.loc[i - 1, "day_key"]:
                switch_cost += dist[ROOM_TO_IDX[g.loc[i - 1, "room"]], ROOM_TO_IDX[row["room"]]]

    return {"room_switch_distance": float(switch_cost), "preference_penalty": float(pref_pen)}


def provider_day_anchor_rooms(schedule_df: pd.DataFrame) -> Dict[tuple, str]:
    anchors: Dict[tuple, str] = {}
    for (provider, day_key), g in schedule_df.groupby(["provider", "day_key"]):
        room_counts = g["room"].value_counts()
        top_count = room_counts.max()
        tied = sorted(room_counts[room_counts == top_count].index.tolist())
        anchors[(provider, day_key)] = tied[0]
    return anchors


def apply_single_room_policy(schedule_df: pd.DataFrame) -> pd.DataFrame:
    anchors = provider_day_anchor_rooms(schedule_df)
    out = schedule_df.copy()
    out["room"] = out.apply(lambda r: anchors[(r["provider"], r["day_key"])], axis=1)
    return out


def rooms_within_threshold(anchor_room: str, dist, threshold: float) -> List[str]:
    a_idx = ROOM_TO_IDX[anchor_room]
    return [r for r in ROOMS if dist[a_idx, ROOM_TO_IDX[r]] <= threshold]


def apply_cluster_policy(schedule_df: pd.DataFrame, dist, threshold: float) -> pd.DataFrame:
    anchors = provider_day_anchor_rooms(schedule_df)
    out = schedule_df.copy()
    remapped_rooms: List[str] = []
    for _, row in out.iterrows():
        anchor = anchors[(row["provider"], row["day_key"])]
        cluster = rooms_within_threshold(anchor, dist, threshold)
        # Keep room if already in cluster; otherwise map to nearest cluster room.
        if row["room"] in cluster:
            remapped_rooms.append(row["room"])
            continue
        old_idx = ROOM_TO_IDX[row["room"]]
        best = min(cluster, key=lambda r: dist[old_idx, ROOM_TO_IDX[r]])
        remapped_rooms.append(best)
    out["room"] = remapped_rooms
    return out


def build_blocked_day_map(schedule_df: pd.DataFrame) -> Dict[str, str]:
    providers = sorted(schedule_df["provider"].unique().tolist())
    return {p: WEEKDAYS[i % len(WEEKDAYS)] for i, p in enumerate(providers)}


def apply_blocked_days_policy(schedule_df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float], Dict[str, str]]:
    blocked_map = build_blocked_day_map(schedule_df)
    out = schedule_df.copy()
    blocked_mask = out.apply(lambda r: blocked_map[r["provider"]] == r["weekday"], axis=1)
    blocked_appts = int(blocked_mask.sum())
    total_appts = int(len(out))
    feasible_df = out.loc[~blocked_mask].copy()
    info = {
        "blocked_appointments": blocked_appts,
        "total_appointments": total_appts,
        "blocked_ratio": float(blocked_appts / total_appts) if total_appts else 0.0,
    }
    return feasible_df, info, blocked_map


def evaluate_week(week_label: str, schedule_df: pd.DataFrame, room_map: dict, dist) -> Dict[str, object]:
    baseline = compute_metrics(schedule_df, room_map, dist)

    single_room_df = apply_single_room_policy(schedule_df)
    single_room = compute_metrics(single_room_df, room_map, dist)

    cluster_df = apply_cluster_policy(schedule_df, dist, CLUSTER_THRESHOLD)
    cluster = compute_metrics(cluster_df, room_map, dist)

    blocked_df, blocked_info, blocked_map = apply_blocked_days_policy(schedule_df)
    blocked_days = compute_metrics(blocked_df, room_map, dist)

    return {
        "week": week_label,
        "policy_design_reasoning": {
            "single_room_anchor_rule": (
                "Use provider-day modal room as anchor for deterministic and reproducible "
                "single-room behavior while preserving original appointment timing."
            ),
            "cluster_threshold_choice": (
                "Threshold 2.0 balances strict locality (<=1.0 can over-constrain) and "
                "overly loose movement (<=3.0 often behaves close to unconstrained)."
            ),
            "blocked_days_auto_strong": (
                "Round-robin weekday blocking is a stress-test policy that intentionally "
                "creates broad availability pressure instead of reflecting baseline operations."
            ),
            "comparison_fairness": (
                "All policies are evaluated as post-processing transforms on the same baseline "
                "schedule to isolate policy effects without rerunning optimization."
            ),
        },
        "baseline": baseline,
        "single_room_policy": single_room,
        "cluster_policy_threshold_2_0": cluster,
        "blocked_days_policy_auto_strong": {
            **blocked_days,
            **blocked_info,
        },
        "blocked_days_assignment": blocked_map,
    }


def flatten_for_table(summary: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for wk in ("Week1", "Week2"):
        week_data = summary[wk]
        rows.extend(
            [
                {"week": wk, "policy": "baseline", **week_data["baseline"]},
                {"week": wk, "policy": "single_room_policy", **week_data["single_room_policy"]},
                {"week": wk, "policy": "cluster_policy_threshold_2_0", **week_data["cluster_policy_threshold_2_0"]},
                {"week": wk, "policy": "blocked_days_policy_auto_strong", **week_data["blocked_days_policy_auto_strong"]},
            ]
        )
    return pd.DataFrame(rows)


def main() -> None:
    base = Path(__file__).resolve().parent
    run_dir = base / "outputs" / "run_20260320_173628"
    inputs_dir = base / "inputs"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    dist = parse_distance_matrix(inputs_dir / "project435Winter2026.pdf")
    room_map_w1 = load_room_assignments(inputs_dir / "ProviderRoomAssignmentWeek1.docx")
    room_map_w2 = load_room_assignments(inputs_dir / "ProviderRoomAssignmentWeek2.docx")

    week1_sched = load_schedule(run_dir / "week1_schedule_output.csv")
    week2_sched = load_schedule(run_dir / "week2_schedule_output.csv")

    summary = {
        "config": {
            "cluster_threshold": CLUSTER_THRESHOLD,
            "blocked_days_policy": "auto_strong_round_robin_weekday",
            "source_run": str(run_dir),
        },
        "Week1": evaluate_week("Week1", week1_sched, room_map_w1, dist),
        "Week2": evaluate_week("Week2", week2_sched, room_map_w2, dist),
    }

    table_df = flatten_for_table(summary)
    with open(run_dir / "policy_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    table_df.to_csv(run_dir / "policy_validation_table.csv", index=False)

    print(json.dumps(summary, indent=2))
    print("\nWrote:")
    print(f"- {run_dir / 'policy_validation_summary.json'}")
    print(f"- {run_dir / 'policy_validation_table.csv'}")


if __name__ == "__main__":
    main()
