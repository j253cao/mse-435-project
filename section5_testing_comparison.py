"""
Section 5: Testing and Comparison
=================================
Computes KPIs across multiple scheduling policies, generates comparison
tables and publication-quality graphs.

Policies evaluated:
  (a) Baseline column-generation schedule
  (b) Single-room: one room per provider-day
  (c) Room-cluster: rooms within distance threshold of anchor
  (d) Blocked-days: round-robin weekday blocking (stress test)
  (e) Admin-time: quantify overflow headroom in admin blocks
  (f) Overbooking: scale bookings by 1/(1 - no-show rate)
  (g) Duration uncertainty: buffer durations and count overlaps

Additionally runs cluster threshold sensitivity (0.5 to 4.0).
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOMS = [f"ER{i}" for i in range(1, 17)]
ROOM_TO_IDX = {r: i for i, r in enumerate(ROOMS)}

DISTANCE_MATRIX_UPPER = [
    (0,1,0.5),(0,2,1.5),(0,3,2.5),(0,4,4),(0,5,4.7),(0,6,2),(0,7,3),
    (0,8,4.7),(0,9,4.1),(0,10,3.3),(0,11,1.5),(0,12,3),(0,13,6.1),(0,14,11.5),(0,15,12),
    (1,2,1),(1,3,2),(1,4,3.5),(1,5,4.2),(1,6,1.5),(1,7,2.5),(1,8,4.2),
    (1,9,3.6),(1,10,2.8),(1,11,2),(1,12,3.5),(1,13,5.6),(1,14,11),(1,15,11.5),
    (2,3,1),(2,4,2.5),(2,5,3.2),(2,6,0.5),(2,7,1.5),(2,8,3.2),(2,9,2.6),
    (2,10,1.8),(2,11,3),(2,12,4.5),(2,13,4.6),(2,14,10),(2,15,10.5),
    (3,4,1.5),(3,5,2.2),(3,6,1.5),(3,7,1.5),(3,8,2.2),(3,9,1.6),
    (3,10,2.8),(3,11,4),(3,12,5.5),(3,13,3.6),(3,14,9),(3,15,9.5),
    (4,5,0.7),(4,6,3),(4,7,3),(4,8,2.5),(4,9,3.1),(4,10,4.3),
    (4,11,5.5),(4,12,6),(4,13,3.9),(4,14,7.5),(4,15,8),
    (5,6,3.7),(5,7,3.7),(5,8,3.2),(5,9,3.8),(5,10,5),
    (5,11,6.2),(5,12,6.7),(5,13,4.6),(5,14,6.8),(5,15,7.3),
    (6,7,1),(6,8,2.7),(6,9,2.1),(6,10,1.3),(6,11,2.5),(6,12,4),(6,13,4.1),(6,14,9.5),(6,15,10),
    (7,8,1.7),(7,9,1.1),(7,10,1.3),(7,11,2.5),(7,12,4),(7,13,3.1),(7,14,8.5),(7,15,9),
    (8,9,0.6),(8,10,1.8),(8,11,3.2),(8,12,5.7),(8,13,1.4),(8,14,6.8),(8,15,7.3),
    (9,10,1.2),(9,11,2.6),(9,12,5.1),(9,13,2),(9,14,7.4),(9,15,7.9),
    (10,11,1.8),(10,12,4.3),(10,13,2.8),(10,14,8.2),(10,15,8.7),
    (11,12,2.5),(11,13,4.6),(11,14,10),(11,15,10.5),
    (12,13,7.1),(12,14,12.5),(12,15,13),
    (13,14,5.4),(13,15,5.9),
    (14,15,0.5),
]


def build_distance_matrix() -> np.ndarray:
    mat = np.zeros((16, 16))
    for i, j, d in DISTANCE_MATRIX_UPPER:
        mat[i, j] = d
        mat[j, i] = d
    return mat


DIST = build_distance_matrix()

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# ---------------------------------------------------------------------------
# Schedule loading
# ---------------------------------------------------------------------------

def load_schedule(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date", "start", "end"])
    df["day_key"] = df["day_key"].astype(str)
    df["weekday"] = df["date"].dt.day_name()
    return df


def load_appointments(csv_path: Path, week_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    active = df[
        (df["Cancelled Appts"].fillna("N") != "Y")
        & (df["Deleted Appts"].fillna("N") != "Y")
    ].copy()
    return active


# ---------------------------------------------------------------------------
# Room-assignment helpers (self-contained, no docx dependency)
# ---------------------------------------------------------------------------

def _try_load_room_map(inputs_dir: Path, week: int) -> Optional[dict]:
    try:
        from clinic_schedule_part3_column_generation import load_room_assignments
        fname = f"ProviderRoomAssignmentWeek{week}.docx"
        path = inputs_dir / fname
        if path.exists():
            return load_room_assignments(path)
    except Exception:
        pass
    return None


def _room_pref_penalty(room: str, pref_rooms: List[str]) -> float:
    if not pref_rooms:
        return 0.0
    ridx = ROOM_TO_IDX[room]
    return min(DIST[ridx, ROOM_TO_IDX[p]] for p in pref_rooms)


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def compute_kpis(schedule_df: pd.DataFrame, label: str) -> Dict[str, object]:
    """Compute a comprehensive set of KPIs for a given schedule."""
    if schedule_df.empty:
        return {"label": label, "total_appointments": 0}

    n_appts = len(schedule_df)
    n_providers = schedule_df["provider"].nunique()
    n_days = schedule_df["day_key"].nunique()

    switch_dist = 0.0
    switch_count = 0
    provider_day_switches: Dict[str, int] = defaultdict(int)
    provider_day_switch_dist: Dict[str, float] = defaultdict(float)

    for provider, g in schedule_df.groupby("provider"):
        g = g.sort_values(["start", "end", "appt_id"]).reset_index(drop=True)
        for i in range(1, len(g)):
            if g.loc[i, "day_key"] == g.loc[i - 1, "day_key"]:
                d = DIST[ROOM_TO_IDX[g.loc[i - 1, "room"]], ROOM_TO_IDX[g.loc[i, "room"]]]
                switch_dist += d
                pd_key = f"{provider}|{g.loc[i, 'day_key']}"
                if d > 0:
                    switch_count += 1
                    provider_day_switches[pd_key] += 1
                    provider_day_switch_dist[pd_key] += d

    pd_switch_vals = list(provider_day_switches.values()) if provider_day_switches else [0]
    avg_switches_per_pd = float(np.mean(pd_switch_vals))
    max_switches_any_pd = max(pd_switch_vals)

    unique_rooms_per_pd = []
    for (prov, dk), g in schedule_df.groupby(["provider", "day_key"]):
        unique_rooms_per_pd.append(g["room"].nunique())
    avg_rooms_per_pd = float(np.mean(unique_rooms_per_pd))

    room_slot_usage = set()
    total_slots_used = 0
    for _, row in schedule_df.iterrows():
        dur = int(row["dur_slots"])
        for s in range(dur):
            t = row["start"] + pd.Timedelta(minutes=15 * s)
            room_slot_usage.add((row["room"], t))
            total_slots_used += 1

    total_available_room_slots = 0
    for dk in schedule_df["day_key"].unique():
        day_date = schedule_df.loc[schedule_df["day_key"] == dk, "date"].iloc[0]
        wd = day_date.day_name()
        if wd != "Friday":
            clinical_slots = 18  # 9:30-16:30 minus lunch/admin = ~18 usable quarter-hours
        else:
            clinical_slots = 18  # 8:30-15:00 minus lunch/admin = ~18
        total_available_room_slots += clinical_slots * len(ROOMS)
    room_util = total_slots_used / total_available_room_slots if total_available_room_slots else 0

    gaps = []
    for provider, g in schedule_df.groupby("provider"):
        g = g.sort_values(["start", "end", "appt_id"]).reset_index(drop=True)
        for i in range(1, len(g)):
            if g.loc[i, "day_key"] == g.loc[i - 1, "day_key"]:
                gap = (g.loc[i, "start"] - g.loc[i - 1, "end"]).total_seconds() / 60
                if gap > 0:
                    gaps.append(gap)
    avg_gap_min = float(np.mean(gaps)) if gaps else 0.0

    provider_util = []
    for (prov, dk), g in schedule_df.groupby(["provider", "day_key"]):
        total_dur_min = g["dur_slots"].sum() * 15
        first_start = g["start"].min()
        last_end = g["end"].max()
        span_min = (last_end - first_start).total_seconds() / 60
        if span_min > 0:
            provider_util.append(total_dur_min / span_min)
    avg_provider_util = float(np.mean(provider_util)) if provider_util else 0.0

    return {
        "label": label,
        "total_appointments": n_appts,
        "providers": n_providers,
        "clinic_days": n_days,
        "total_room_switch_distance": round(switch_dist, 1),
        "total_room_switches": switch_count,
        "avg_switches_per_provider_day": round(avg_switches_per_pd, 2),
        "max_switches_any_provider_day": max_switches_any_pd,
        "avg_unique_rooms_per_provider_day": round(avg_rooms_per_pd, 2),
        "room_utilization_pct": round(room_util * 100, 1),
        "avg_inter_appt_gap_min": round(avg_gap_min, 1),
        "avg_provider_utilization_pct": round(avg_provider_util * 100, 1),
    }


# ---------------------------------------------------------------------------
# Policy transforms (self-contained)
# ---------------------------------------------------------------------------

def provider_day_anchor_rooms(sched: pd.DataFrame) -> Dict[Tuple[str, str], str]:
    anchors = {}
    for (prov, dk), g in sched.groupby(["provider", "day_key"]):
        counts = g["room"].value_counts()
        top = counts.max()
        tied = sorted(counts[counts == top].index.tolist())
        anchors[(prov, dk)] = tied[0]
    return anchors


def apply_single_room(sched: pd.DataFrame) -> pd.DataFrame:
    anchors = provider_day_anchor_rooms(sched)
    out = sched.copy()
    out["room"] = out.apply(lambda r: anchors[(r["provider"], r["day_key"])], axis=1)
    return out


def apply_cluster(sched: pd.DataFrame, threshold: float) -> pd.DataFrame:
    anchors = provider_day_anchor_rooms(sched)
    out = sched.copy()
    rooms = []
    for _, row in out.iterrows():
        anchor = anchors[(row["provider"], row["day_key"])]
        a_idx = ROOM_TO_IDX[anchor]
        cluster = [r for r in ROOMS if DIST[a_idx, ROOM_TO_IDX[r]] <= threshold]
        if row["room"] in cluster:
            rooms.append(row["room"])
        else:
            old_idx = ROOM_TO_IDX[row["room"]]
            best = min(cluster, key=lambda r: DIST[old_idx, ROOM_TO_IDX[r]])
            rooms.append(best)
    out["room"] = rooms
    return out


def apply_blocked_days(sched: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    providers = sorted(sched["provider"].unique())
    blocked_map = {p: WEEKDAYS[i % len(WEEKDAYS)] for i, p in enumerate(providers)}
    mask = sched.apply(lambda r: blocked_map[r["provider"]] == r["weekday"], axis=1)
    blocked_count = int(mask.sum())
    total = len(sched)
    feasible = sched.loc[~mask].copy()
    info = {
        "blocked_appointments": blocked_count,
        "total_appointments": total,
        "blocked_ratio_pct": round(100 * blocked_count / total, 1) if total else 0,
    }
    return feasible, info


def compute_no_show_rate(appts_df: pd.DataFrame) -> float:
    col = "No Show Appts " if "No Show Appts " in appts_df.columns else "No Show Appts"
    if col not in appts_df.columns:
        return 0.10
    ns = (appts_df[col].fillna("N").str.strip().str.upper() == "Y").sum()
    return float(ns / len(appts_df)) if len(appts_df) > 0 else 0.10


def overbooking_analysis(sched: pd.DataFrame, appts_df: pd.DataFrame) -> dict:
    r = compute_no_show_rate(appts_df)
    n = len(sched)
    factor = 1 / (1 - r) if r < 1 else 1.0
    extra = int(math.ceil(n * (factor - 1)))
    return {
        "no_show_rate_pct": round(r * 100, 1),
        "scheduled_slots": n,
        "recommended_overbook_slots": extra,
        "effective_capacity": n + extra,
    }


def duration_uncertainty(sched: pd.DataFrame, buffer_slots: int) -> dict:
    out = sched.copy()
    out["dur_buffered"] = out["dur_slots"].astype(int) + buffer_slots
    out["end_buffered"] = out["start"] + out["dur_buffered"].apply(
        lambda s: pd.Timedelta(minutes=15 * int(s))
    )
    overlaps = 0
    total_trans = 0
    for _, g in out.groupby(["provider", "day_key", "block_key"], sort=False):
        g = g.sort_values("start").reset_index(drop=True)
        for i in range(1, len(g)):
            total_trans += 1
            if g.iloc[i]["start"] < g.iloc[i - 1]["end_buffered"]:
                overlaps += 1
    return {
        "buffer_minutes": buffer_slots * 15,
        "overlaps": overlaps,
        "total_chain_transitions": total_trans,
        "overlap_ratio_pct": round(100 * overlaps / total_trans, 1) if total_trans else 0,
    }


def admin_time_analysis(sched: pd.DataFrame) -> dict:
    """Count congested provider-days and potential admin overflow capacity."""
    pd_counts = {}
    for (prov, dk), g in sched.groupby(["provider", "day_key"]):
        pd_counts[f"{prov}|{dk}"] = len(g)
    vals = list(pd_counts.values())
    median_c = float(np.median(vals)) if vals else 0
    congested = sum(1 for v in vals if v >= max(median_c, 5))
    return {
        "congested_provider_days": congested,
        "admin_capacity_slots": congested * 4,
        "median_appts_per_provider_day": round(median_c, 1),
    }


# ---------------------------------------------------------------------------
# Full evaluation pipeline for one week
# ---------------------------------------------------------------------------

def evaluate_week(
    week_label: str,
    sched: pd.DataFrame,
    appts_df: Optional[pd.DataFrame],
    cluster_thresholds: List[float] = None,
) -> dict:
    if cluster_thresholds is None:
        cluster_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    results = {}

    results["baseline"] = compute_kpis(sched, f"{week_label} Baseline")

    sr = apply_single_room(sched)
    results["single_room"] = compute_kpis(sr, f"{week_label} Single-Room")

    for th in cluster_thresholds:
        cl = apply_cluster(sched, th)
        results[f"cluster_{th:.1f}"] = compute_kpis(cl, f"{week_label} Cluster <={th}")

    bl_sched, bl_info = apply_blocked_days(sched)
    results["blocked_days"] = {**compute_kpis(bl_sched, f"{week_label} Blocked-Days"), **bl_info}

    results["admin_time"] = admin_time_analysis(sched)

    if appts_df is not None:
        results["overbooking"] = overbooking_analysis(sched, appts_df)

    results["duration_uncertainty_15min"] = duration_uncertainty(sched, 1)
    results["duration_uncertainty_30min"] = duration_uncertainty(sched, 2)

    return results


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def build_main_comparison_table(w1_res: dict, w2_res: dict) -> pd.DataFrame:
    """Build the primary KPI comparison table (Table 4 candidate)."""
    policies = ["baseline", "single_room", "cluster_2.0", "blocked_days"]
    nice = {
        "baseline": "Baseline (CG)",
        "single_room": "Single-Room",
        "cluster_2.0": "Cluster (<=2.0 m)",
        "blocked_days": "Blocked-Days",
    }

    rows = []
    for pol in policies:
        for wk_label, res in [("Week 1", w1_res), ("Week 2", w2_res)]:
            d = res.get(pol, {})
            rows.append({
                "Policy": nice.get(pol, pol),
                "Week": wk_label,
                "Switch Distance (m)": d.get("total_room_switch_distance", ""),
                "# Room Switches": d.get("total_room_switches", ""),
                "Avg Switches/PD": d.get("avg_switches_per_provider_day", ""),
                "Max Switches (any PD)": d.get("max_switches_any_provider_day", ""),
                "Avg Rooms/PD": d.get("avg_unique_rooms_per_provider_day", ""),
                "Room Util. (%)": d.get("room_utilization_pct", ""),
                "Provider Util. (%)": d.get("avg_provider_utilization_pct", ""),
                "Avg Gap (min)": d.get("avg_inter_appt_gap_min", ""),
            })
    return pd.DataFrame(rows)


def build_sensitivity_table(w1_res: dict, w2_res: dict) -> pd.DataFrame:
    """Cluster threshold sensitivity table."""
    rows = []
    for th in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        key = f"cluster_{th:.1f}"
        for wk_label, res in [("Week 1", w1_res), ("Week 2", w2_res)]:
            d = res.get(key, {})
            rows.append({
                "Threshold (m)": th,
                "Week": wk_label,
                "Switch Distance (m)": d.get("total_room_switch_distance", ""),
                "# Room Switches": d.get("total_room_switches", ""),
                "Avg Rooms/PD": d.get("avg_unique_rooms_per_provider_day", ""),
            })
    return pd.DataFrame(rows)


def build_extended_policy_table(w1_res: dict, w2_res: dict) -> pd.DataFrame:
    """Admin time, overbooking, duration uncertainty summary."""
    rows = []
    for wk_label, res in [("Week 1", w1_res), ("Week 2", w2_res)]:
        adm = res.get("admin_time", {})
        rows.append({
            "Week": wk_label,
            "Metric": "Congested Provider-Days",
            "Value": adm.get("congested_provider_days", ""),
        })
        rows.append({
            "Week": wk_label,
            "Metric": "Admin Capacity Slots (potential)",
            "Value": adm.get("admin_capacity_slots", ""),
        })
        ob = res.get("overbooking", {})
        rows.append({
            "Week": wk_label,
            "Metric": "No-Show Rate (%)",
            "Value": ob.get("no_show_rate_pct", ""),
        })
        rows.append({
            "Week": wk_label,
            "Metric": "Recommended Overbook Slots",
            "Value": ob.get("recommended_overbook_slots", ""),
        })
        du15 = res.get("duration_uncertainty_15min", {})
        du30 = res.get("duration_uncertainty_30min", {})
        rows.append({
            "Week": wk_label,
            "Metric": "Overlap Ratio (+15 min, %)",
            "Value": du15.get("overlap_ratio_pct", ""),
        })
        rows.append({
            "Week": wk_label,
            "Metric": "Overlap Ratio (+30 min, %)",
            "Value": du30.get("overlap_ratio_pct", ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

COLORS = {
    "Baseline (CG)": "#2196F3",
    "Single-Room": "#4CAF50",
    "Cluster (<=2.0 m)": "#FF9800",
    "Blocked-Days": "#F44336",
}


def fig_bar_switch_distance(w1: dict, w2: dict, out_path: Path):
    """Grouped bar chart: room-switch distance by policy and week."""
    policies = ["baseline", "single_room", "cluster_2.0", "blocked_days"]
    nice = ["Baseline (CG)", "Single-Room", "Cluster (<=2.0 m)", "Blocked-Days"]
    w1_vals = [w1.get(p, {}).get("total_room_switch_distance", 0) for p in policies]
    w2_vals = [w2.get(p, {}).get("total_room_switch_distance", 0) for p in policies]

    x = np.arange(len(policies))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, w1_vals, width, label="Week 1", color="#2196F3")
    bars2 = ax.bar(x + width / 2, w2_vals, width, label="Week 2", color="#FF9800")
    ax.set_ylabel("Total Room-Switch Distance (m)", fontsize=12)
    ax.set_title("Room-Switch Distance by Scheduling Policy", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(nice, fontsize=10)
    ax.legend(fontsize=11)
    ax.bar_label(bars1, fmt="%.0f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.0f", padding=3, fontsize=8)
    ax.set_ylim(0, max(max(w1_vals), max(w2_vals)) * 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_bar_switch_count(w1: dict, w2: dict, out_path: Path):
    """Grouped bar chart: number of room switches."""
    policies = ["baseline", "single_room", "cluster_2.0", "blocked_days"]
    nice = ["Baseline (CG)", "Single-Room", "Cluster (<=2.0 m)", "Blocked-Days"]
    w1_vals = [w1.get(p, {}).get("total_room_switches", 0) for p in policies]
    w2_vals = [w2.get(p, {}).get("total_room_switches", 0) for p in policies]

    x = np.arange(len(policies))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, w1_vals, width, label="Week 1", color="#2196F3")
    ax.bar(x + width / 2, w2_vals, width, label="Week 2", color="#FF9800")
    ax.set_ylabel("Number of Room Switches", fontsize=12)
    ax.set_title("Room Switches by Scheduling Policy", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(nice, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(w1_vals), max(w2_vals)) * 1.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_bar_provider_utilization(w1: dict, w2: dict, out_path: Path):
    """Grouped bar chart: average provider utilization."""
    policies = ["baseline", "single_room", "cluster_2.0", "blocked_days"]
    nice = ["Baseline (CG)", "Single-Room", "Cluster (<=2.0 m)", "Blocked-Days"]
    w1_vals = [w1.get(p, {}).get("avg_provider_utilization_pct", 0) for p in policies]
    w2_vals = [w2.get(p, {}).get("avg_provider_utilization_pct", 0) for p in policies]

    x = np.arange(len(policies))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, w1_vals, width, label="Week 1", color="#2196F3")
    ax.bar(x + width / 2, w2_vals, width, label="Week 2", color="#FF9800")
    ax.set_ylabel("Provider Utilization (%)", fontsize=12)
    ax.set_title("Average Provider Utilization by Policy", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(nice, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_cluster_sensitivity(w1: dict, w2: dict, out_path: Path):
    """Line chart: switch distance vs cluster threshold."""
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    w1_vals = [w1.get(f"cluster_{t:.1f}", {}).get("total_room_switch_distance", 0) for t in thresholds]
    w2_vals = [w2.get(f"cluster_{t:.1f}", {}).get("total_room_switch_distance", 0) for t in thresholds]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, w1_vals, "o-", color="#2196F3", linewidth=2, markersize=7, label="Week 1")
    ax.plot(thresholds, w2_vals, "s-", color="#FF9800", linewidth=2, markersize=7, label="Week 2")

    w1_base = w1.get("baseline", {}).get("total_room_switch_distance", 0)
    w2_base = w2.get("baseline", {}).get("total_room_switch_distance", 0)
    ax.axhline(w1_base, color="#2196F3", linestyle="--", alpha=0.5, label="W1 Baseline")
    ax.axhline(w2_base, color="#FF9800", linestyle="--", alpha=0.5, label="W2 Baseline")

    ax.set_xlabel("Cluster Distance Threshold (m)", fontsize=12)
    ax.set_ylabel("Total Room-Switch Distance (m)", fontsize=12)
    ax.set_title("Cluster Policy Sensitivity Analysis", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(thresholds)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_cluster_sensitivity_switches(w1: dict, w2: dict, out_path: Path):
    """Line chart: number of switches vs cluster threshold."""
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    w1_vals = [w1.get(f"cluster_{t:.1f}", {}).get("total_room_switches", 0) for t in thresholds]
    w2_vals = [w2.get(f"cluster_{t:.1f}", {}).get("total_room_switches", 0) for t in thresholds]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, w1_vals, "o-", color="#2196F3", linewidth=2, markersize=7, label="Week 1")
    ax.plot(thresholds, w2_vals, "s-", color="#FF9800", linewidth=2, markersize=7, label="Week 2")
    ax.set_xlabel("Cluster Distance Threshold (m)", fontsize=12)
    ax.set_ylabel("Number of Room Switches", fontsize=12)
    ax.set_title("Room Switch Count vs Cluster Threshold", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(thresholds)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_duration_uncertainty(w1: dict, w2: dict, out_path: Path):
    """Bar chart: overlap ratios under duration buffers."""
    buffers = ["+15 min", "+30 min"]
    w1_vals = [
        w1.get("duration_uncertainty_15min", {}).get("overlap_ratio_pct", 0),
        w1.get("duration_uncertainty_30min", {}).get("overlap_ratio_pct", 0),
    ]
    w2_vals = [
        w2.get("duration_uncertainty_15min", {}).get("overlap_ratio_pct", 0),
        w2.get("duration_uncertainty_30min", {}).get("overlap_ratio_pct", 0),
    ]

    x = np.arange(len(buffers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, w1_vals, width, label="Week 1", color="#2196F3")
    bars2 = ax.bar(x + width / 2, w2_vals, width, label="Week 2", color="#FF9800")
    ax.set_ylabel("Overlap Ratio (%)", fontsize=12)
    ax.set_title("Schedule Fragility Under Duration Uncertainty", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(buffers, fontsize=11)
    ax.legend(fontsize=11)
    ax.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_overbooking(w1: dict, w2: dict, out_path: Path):
    """Stacked bar: scheduled vs recommended overbook slots."""
    labels = ["Week 1", "Week 2"]
    sched = [
        w1.get("overbooking", {}).get("scheduled_slots", 0),
        w2.get("overbooking", {}).get("scheduled_slots", 0),
    ]
    extra = [
        w1.get("overbooking", {}).get("recommended_overbook_slots", 0),
        w2.get("overbooking", {}).get("recommended_overbook_slots", 0),
    ]
    rates = [
        w1.get("overbooking", {}).get("no_show_rate_pct", 0),
        w2.get("overbooking", {}).get("no_show_rate_pct", 0),
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, sched, 0.5, label="Scheduled", color="#2196F3")
    ax.bar(x, extra, 0.5, bottom=sched, label="Overbook Slots", color="#FF9800")
    for i in range(len(labels)):
        ax.text(x[i], sched[i] + extra[i] + 5, f"No-show: {rates[i]:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("Appointment Slots", fontsize=12)
    ax.set_title("Overbooking Recommendation", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_cg_convergence(cg1_path: Path, cg2_path: Path, out_path: Path):
    """Line chart: CG objective convergence over iterations."""
    cg1 = pd.read_csv(cg1_path)
    cg2 = pd.read_csv(cg2_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(cg1["iteration"], cg1["rmp_objective"], "o-", color="#2196F3", linewidth=2, markersize=6)
    ax1.set_xlabel("CG Iteration", fontsize=11)
    ax1.set_ylabel("RMP Objective", fontsize=11)
    ax1.set_title("Week 1 – CG Convergence", fontsize=13, fontweight="bold")
    ax1.set_xticks(cg1["iteration"])

    ax1_twin = ax1.twinx()
    ax1_twin.bar(cg1["iteration"], cg1["columns_added"], alpha=0.3, color="#4CAF50", label="Columns Added")
    ax1_twin.set_ylabel("Columns Added", fontsize=11, color="#4CAF50")
    ax1_twin.tick_params(axis="y", labelcolor="#4CAF50")

    ax2.plot(cg2["iteration"], cg2["rmp_objective"], "s-", color="#FF9800", linewidth=2, markersize=6)
    ax2.set_xlabel("CG Iteration", fontsize=11)
    ax2.set_ylabel("RMP Objective", fontsize=11)
    ax2.set_title("Week 2 – CG Convergence", fontsize=13, fontweight="bold")
    ax2.set_xticks(cg2["iteration"])

    ax2_twin = ax2.twinx()
    ax2_twin.bar(cg2["iteration"], cg2["columns_added"], alpha=0.3, color="#4CAF50", label="Columns Added")
    ax2_twin.set_ylabel("Columns Added", fontsize=11, color="#4CAF50")
    ax2_twin.tick_params(axis="y", labelcolor="#4CAF50")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_gantt_provider_day(sched: pd.DataFrame, provider: str, day_key: str, title: str, out_path: Path):
    """Gantt chart for a single provider-day."""
    sub = sched[(sched["provider"] == provider) & (sched["day_key"] == day_key)].copy()
    if sub.empty:
        return
    sub = sub.sort_values("start").reset_index(drop=True)

    room_list = sorted(sub["room"].unique(), key=lambda r: ROOM_TO_IDX[r])
    room_y = {r: i for i, r in enumerate(room_list)}
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(room_list))

    fig, ax = plt.subplots(figsize=(12, max(3, len(room_list) * 0.8 + 1)))
    for _, row in sub.iterrows():
        y = room_y[row["room"]]
        start_h = row["start"].hour + row["start"].minute / 60
        dur_h = (row["end"] - row["start"]).total_seconds() / 3600
        bar = ax.barh(y, dur_h, left=start_h, height=0.6, color=cmap(y), edgecolor="black", linewidth=0.5)
        ax.text(start_h + dur_h / 2, y, row["appt_id"].split("_")[-1],
                ha="center", va="center", fontsize=7, fontweight="bold")

    ax.set_yticks(range(len(room_list)))
    ax.set_yticklabels(room_list, fontsize=10)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Exam Room", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(8, 17)
    ax.set_xticks(range(8, 18))
    ax.set_xticklabels([f"{h}:00" for h in range(8, 18)], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_room_utilization_heatmap(sched: pd.DataFrame, week_label: str, out_path: Path):
    """Heatmap of room utilization by day."""
    days = sorted(sched["day_key"].unique())
    usage = np.zeros((len(ROOMS), len(days)))

    for col_i, dk in enumerate(days):
        day_sub = sched[sched["day_key"] == dk]
        for _, row in day_sub.iterrows():
            r_idx = ROOM_TO_IDX[row["room"]]
            usage[r_idx, col_i] += int(row["dur_slots"])

    fig, ax = plt.subplots(figsize=(max(8, len(days) * 1.5), 8))
    im = ax.imshow(usage, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(days)))
    day_labels = [pd.Timestamp(d).strftime("%a %m/%d") for d in days]
    ax.set_xticklabels(day_labels, fontsize=10)
    ax.set_yticks(range(len(ROOMS)))
    ax.set_yticklabels(ROOMS, fontsize=10)
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Room", fontsize=12)
    ax.set_title(f"{week_label} – Room Utilization (15-min slots)", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Slots Used", fontsize=11)

    for i in range(len(ROOMS)):
        for j in range(len(days)):
            val = int(usage[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8,
                        color="white" if val > usage.max() * 0.6 else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_kpi_radar(w1: dict, w2: dict, out_path: Path):
    """Radar chart comparing baseline KPIs across weeks."""
    metrics = [
        ("Switch Distance", "total_room_switch_distance"),
        ("# Switches", "total_room_switches"),
        ("Avg Switches/PD", "avg_switches_per_provider_day"),
        ("Avg Rooms/PD", "avg_unique_rooms_per_provider_day"),
        ("Room Util (%)", "room_utilization_pct"),
        ("Provider Util (%)", "avg_provider_utilization_pct"),
    ]
    labels = [m[0] for m in metrics]
    w1_base = w1.get("baseline", {})
    w2_base = w2.get("baseline", {})
    w1_vals = [w1_base.get(m[1], 0) for m in metrics]
    w2_vals = [w2_base.get(m[1], 0) for m in metrics]

    maxvals = [max(a, b, 1) for a, b in zip(w1_vals, w2_vals)]
    w1_norm = [v / m for v, m in zip(w1_vals, maxvals)]
    w2_norm = [v / m for v, m in zip(w2_vals, maxvals)]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    w1_norm += w1_norm[:1]
    w2_norm += w2_norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, w1_norm, "o-", linewidth=2, label="Week 1", color="#2196F3")
    ax.fill(angles, w1_norm, alpha=0.15, color="#2196F3")
    ax.plot(angles, w2_norm, "s-", linewidth=2, label="Week 2", color="#FF9800")
    ax.fill(angles, w2_norm, alpha=0.15, color="#FF9800")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Baseline KPI Comparison (Normalized)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_provider_workload(sched: pd.DataFrame, week_label: str, out_path: Path):
    """Horizontal bar chart of total appointment slots per provider."""
    prov_load = sched.groupby("provider")["dur_slots"].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, len(prov_load) * 0.35)))
    bars = ax.barh(range(len(prov_load)), prov_load.values, color="#2196F3", edgecolor="white")
    ax.set_yticks(range(len(prov_load)))
    ax.set_yticklabels(prov_load.index, fontsize=9)
    ax.set_xlabel("Total Appointment Slots (15 min each)", fontsize=11)
    ax.set_title(f"{week_label} – Provider Workload Distribution", fontsize=13, fontweight="bold")
    ax.bar_label(bars, padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_multi_policy_gantt(sched_dict: Dict[str, pd.DataFrame], provider: str,
                           day_key: str, out_path: Path):
    """Side-by-side Gantt charts showing the same provider-day under different policies."""
    policy_names = list(sched_dict.keys())
    n_policies = len(policy_names)
    fig, axes = plt.subplots(n_policies, 1, figsize=(14, 2.5 * n_policies), sharex=True)
    if n_policies == 1:
        axes = [axes]

    for ax, pname in zip(axes, policy_names):
        s = sched_dict[pname]
        sub = s[(s["provider"] == provider) & (s["day_key"] == day_key)].copy()
        if sub.empty:
            ax.set_title(f"{pname} (no appointments)", fontsize=11)
            continue
        sub = sub.sort_values("start")
        room_list = sorted(sub["room"].unique(), key=lambda r: ROOM_TO_IDX[r])
        room_y = {r: i for i, r in enumerate(room_list)}
        cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(room_list), 1))

        for _, row in sub.iterrows():
            y = room_y[row["room"]]
            sh = row["start"].hour + row["start"].minute / 60
            dh = (row["end"] - row["start"]).total_seconds() / 3600
            ax.barh(y, dh, left=sh, height=0.6, color=cmap(y), edgecolor="black", linewidth=0.5)

        ax.set_yticks(range(len(room_list)))
        ax.set_yticklabels(room_list, fontsize=9)
        ax.set_title(f"{pname}", fontsize=11, fontweight="bold")
        ax.set_xlim(8, 17)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    axes[-1].set_xlabel("Hour of Day", fontsize=11)
    axes[-1].set_xticks(range(8, 18))
    axes[-1].set_xticklabels([f"{h}:00" for h in range(8, 18)], fontsize=9)
    fig.suptitle(f"Policy Comparison – {provider} on {day_key}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_run_dir(base: Path) -> Path:
    """Find the best available run output directory."""
    candidates = [
        base / "outputs" / "run_20260320_173628",
        base / "outputs" / "run_anastan",
    ]
    for c in candidates:
        if c.exists() and (c / "week1_schedule_output.csv").exists():
            return c
    outputs = base / "outputs"
    if outputs.exists():
        runs = sorted(outputs.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for r in runs:
            if (r / "week1_schedule_output.csv").exists():
                return r
    raise FileNotFoundError("No valid run output directory found in outputs/")


def main() -> None:
    base = Path(__file__).resolve().parent
    inputs_dir = base / "inputs"
    run_dir = find_run_dir(base)
    print(f"Using run directory: {run_dir}")

    out_dir = base / "outputs" / "section5_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    w1_sched = load_schedule(run_dir / "week1_schedule_output.csv")
    w2_sched = load_schedule(run_dir / "week2_schedule_output.csv")
    print(f"Loaded schedules: Week1={len(w1_sched)} appts, Week2={len(w2_sched)} appts")

    w1_appts, w2_appts = None, None
    try:
        w1_appts = load_appointments(inputs_dir / "AppointmentDataWeek1.csv", "W1")
        w2_appts = load_appointments(inputs_dir / "AppointmentDataWeek2.csv", "W2")
        print(f"Loaded appointment data: W1={len(w1_appts)}, W2={len(w2_appts)}")
    except Exception as e:
        print(f"Could not load raw appointment data: {e}")

    print("\n--- Computing KPIs across policies ---")
    w1_results = evaluate_week("Week 1", w1_sched, w1_appts)
    w2_results = evaluate_week("Week 2", w2_sched, w2_appts)

    print("\n--- Building comparison tables ---")
    main_table = build_main_comparison_table(w1_results, w2_results)
    sensitivity_table = build_sensitivity_table(w1_results, w2_results)
    extended_table = build_extended_policy_table(w1_results, w2_results)

    main_table.to_csv(out_dir / "table_main_kpi_comparison.csv", index=False)
    sensitivity_table.to_csv(out_dir / "table_cluster_sensitivity.csv", index=False)
    extended_table.to_csv(out_dir / "table_extended_policies.csv", index=False)

    full_results = {
        "source_run": str(run_dir),
        "Week1": w1_results,
        "Week2": w2_results,
    }
    with open(out_dir / "section5_full_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, default=str)

    print("\n--- Generating figures ---")

    fig_bar_switch_distance(w1_results, w2_results, fig_dir / "fig_switch_distance_by_policy.png")
    print("  [1/11] Switch distance bar chart")

    fig_bar_switch_count(w1_results, w2_results, fig_dir / "fig_switch_count_by_policy.png")
    print("  [2/11] Switch count bar chart")

    fig_bar_provider_utilization(w1_results, w2_results, fig_dir / "fig_provider_utilization.png")
    print("  [3/11] Provider utilization bar chart")

    fig_cluster_sensitivity(w1_results, w2_results, fig_dir / "fig_cluster_sensitivity_distance.png")
    print("  [4/11] Cluster sensitivity (distance)")

    fig_cluster_sensitivity_switches(w1_results, w2_results, fig_dir / "fig_cluster_sensitivity_switches.png")
    print("  [5/11] Cluster sensitivity (switches)")

    fig_duration_uncertainty(w1_results, w2_results, fig_dir / "fig_duration_uncertainty.png")
    print("  [6/11] Duration uncertainty")

    if w1_appts is not None:
        fig_overbooking(w1_results, w2_results, fig_dir / "fig_overbooking.png")
        print("  [7/11] Overbooking")
    else:
        print("  [7/11] Overbooking (skipped – no appointment data)")

    cg1_path = run_dir / "week1_cg_iterations.csv"
    cg2_path = run_dir / "week2_cg_iterations.csv"
    if cg1_path.exists() and cg2_path.exists():
        fig_cg_convergence(cg1_path, cg2_path, fig_dir / "fig_cg_convergence.png")
        print("  [8/11] CG convergence")
    else:
        print("  [8/11] CG convergence (skipped – no iteration files)")

    fig_kpi_radar(w1_results, w2_results, fig_dir / "fig_kpi_radar.png")
    print("  [9/11] KPI radar chart")

    fig_room_utilization_heatmap(w1_sched, "Week 1", fig_dir / "fig_room_heatmap_week1.png")
    fig_room_utilization_heatmap(w2_sched, "Week 2", fig_dir / "fig_room_heatmap_week2.png")
    print("  [10/11] Room utilization heatmaps")

    fig_provider_workload(w1_sched, "Week 1", fig_dir / "fig_provider_workload_week1.png")
    fig_provider_workload(w2_sched, "Week 2", fig_dir / "fig_provider_workload_week2.png")
    print("  [11/11] Provider workload charts")

    busiest_provider = w1_sched.groupby("provider").size().idxmax()
    busiest_day = w1_sched[w1_sched["provider"] == busiest_provider].groupby("day_key").size().idxmax()
    fig_gantt_provider_day(
        w1_sched, busiest_provider, busiest_day,
        f"Baseline Gantt – {busiest_provider} on {busiest_day}",
        fig_dir / "fig_gantt_baseline_sample.png",
    )
    print(f"  [Bonus] Gantt chart for {busiest_provider} on {busiest_day}")

    sr_sched = apply_single_room(w1_sched)
    cl_sched = apply_cluster(w1_sched, 2.0)
    fig_multi_policy_gantt(
        {"Baseline": w1_sched, "Single-Room": sr_sched, "Cluster (<=2.0 m)": cl_sched},
        busiest_provider, busiest_day,
        fig_dir / "fig_gantt_policy_comparison.png",
    )
    print(f"  [Bonus] Multi-policy Gantt comparison")

    print("\n" + "=" * 70)
    print("MAIN KPI COMPARISON TABLE")
    print("=" * 70)
    print(main_table.to_string(index=False))

    print("\n" + "=" * 70)
    print("CLUSTER THRESHOLD SENSITIVITY")
    print("=" * 70)
    print(sensitivity_table.to_string(index=False))

    print("\n" + "=" * 70)
    print("EXTENDED POLICIES (Admin, Overbooking, Uncertainty)")
    print("=" * 70)
    print(extended_table.to_string(index=False))

    print(f"\n\nAll outputs written to: {out_dir}")
    print(f"  Tables:  {out_dir}")
    print(f"  Figures: {fig_dir}")
    print(f"\nGenerated files:")
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            print(f"  - {f.relative_to(out_dir)}")


if __name__ == "__main__":
    main()
