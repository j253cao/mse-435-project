"""
Generate all figures for the final presentation slides 4 & 5.
Uses verified numbers directly from the run outputs -- no recomputation.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ── Style constants (matching slide sage-green theme) ──
BG = "#D6DFD0"
DARK = "#2D3A2D"
BLUE = "#2196F3"
ORANGE = "#FF9800"
RED = "#E53935"
GREY = "#78909C"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"],
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})

# ── Verified data (from policy_validation_summary.json & part3_results_summary.json) ──

POLICY_DATA = {
    "Baseline (CG)":      {"w1_dist": 339.7, "w2_dist": 600.9},
    "Single-Room":         {"w1_dist": 0.0,   "w2_dist": 0.0},
    "Cluster (<=2.0 m)":   {"w1_dist": 160.6, "w2_dist": 245.0},
    "Blocked-Days":        {"w1_dist": 270.5, "w2_dist": 408.2},
}

CG_RESULTS = {
    "Week1": {
        "final_lp": 51.9, "final_ip": 51.9,
        "runtime": 176.9, "iterations": 8,
        "pool_initial": 94, "pool_final": 160,
    },
    "Week2": {
        "final_lp": 118.8, "final_ip": 118.8,
        "runtime": 311.4, "iterations": 13,
        "pool_initial": 122, "pool_final": 247,
    },
}

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "outputs" / "presentation_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAAAAA")
    ax.spines["bottom"].set_color("#AAAAAA")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Slide 4: CG Convergence (dual-axis, side by side)
# ═══════════════════════════════════════════════════════════════════════════

def fig_cg_convergence():
    run_dir = BASE / "outputs" / "run_20260320_173628"
    cg1 = pd.read_csv(run_dir / "week1_cg_iterations.csv")
    cg2 = pd.read_csv(run_dir / "week2_cg_iterations.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=BG)

    for ax, cg, title, color in [
        (ax1, cg1, "Week 1 - CG Convergence", BLUE),
        (ax2, cg2, "Week 2 - CG Convergence", ORANGE),
    ]:
        ax.set_facecolor(BG)
        line = ax.plot(cg["iteration"], cg["rmp_objective"], "o-",
                       color=color, linewidth=2.2, markersize=6, zorder=5)
        for _, row in cg.iterrows():
            ax.annotate(f"{row['rmp_objective']:.1f}",
                        (row["iteration"], row["rmp_objective"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7.5, color=color, fontweight="bold")
        ax.set_xlabel("CG Iteration", fontsize=10)
        ax.set_ylabel("RMP Objective", fontsize=10, color=color)
        ax.set_title(title, fontsize=12, fontweight="bold", color=DARK)
        ax.set_xticks(cg["iteration"])
        ax.tick_params(axis="y", labelcolor=color)
        _clean_axes(ax)

        ax2r = ax.twinx()
        ax2r.bar(cg["iteration"], cg["columns_added"], alpha=0.35,
                 color="#43A047", zorder=2, width=0.6)
        ax2r.set_ylabel("Columns Added", fontsize=10, color="#43A047")
        ax2r.tick_params(axis="y", labelcolor="#43A047")
        ax2r.spines["top"].set_visible(False)
        ax2r.spines["left"].set_visible(False)
        ax2r.spines["right"].set_color("#43A047")
        for _, row in cg.iterrows():
            if row["columns_added"] > 0:
                ax2r.text(row["iteration"], row["columns_added"] + 0.4,
                          str(int(row["columns_added"])),
                          ha="center", fontsize=7.5, color="#43A047", fontweight="bold")

    fig.tight_layout(w_pad=3)
    path = OUT_DIR / "slide4_cg_convergence.png"
    fig.savefig(path, dpi=250, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1] {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Slide 5 top-right: Room-Switch Distance bar chart
# ═══════════════════════════════════════════════════════════════════════════

def fig_switch_distance_bar():
    policies = list(POLICY_DATA.keys())
    w1 = [POLICY_DATA[p]["w1_dist"] for p in policies]
    w2 = [POLICY_DATA[p]["w2_dist"] for p in policies]

    x = np.arange(len(policies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
    ax.set_facecolor(BG)

    bars1 = ax.bar(x - width / 2, w1, width, label="Week 1",
                   color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    bars2 = ax.bar(x + width / 2, w2, width, label="Week 2",
                   color=ORANGE, edgecolor="white", linewidth=0.8, zorder=3)

    ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=9, fontweight="bold", color=BLUE)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9, fontweight="bold", color=ORANGE)

    ax.set_ylabel("Total Room-Switch Distance (m)", fontsize=11)
    ax.set_title("Room-Switch Distance by Scheduling Policy",
                 fontsize=13, fontweight="bold", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=10)
    ax.legend(fontsize=10, framealpha=0.7)
    ax.set_ylim(0, max(max(w1), max(w2)) * 1.22)
    _clean_axes(ax)
    ax.grid(axis="y", alpha=0.15, zorder=0)

    fig.tight_layout()
    path = OUT_DIR / "slide5_switch_distance_bar.png"
    fig.savefig(path, dpi=250, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2] {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Slide 5 bottom: Three policy boxes (d, e, f)
# ═══════════════════════════════════════════════════════════════════════════

def fig_policy_boxes():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8), facecolor=BG)

    # ── Panel 1: Admin Time (Policy d) ──
    ax = axes[0]
    ax.set_facecolor(BG)
    cats = ["Congested\nProvider-Days", "Admin Capacity\nSlots"]
    v1, v2 = [39, 156], [40, 160]
    x = np.arange(len(cats))
    w = 0.32
    ax.bar(x - w/2, v1, w, label="Week 1", color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + w/2, v2, w, label="Week 2", color=ORANGE, edgecolor="white", linewidth=0.8, zorder=3)
    for i, (a, b) in enumerate(zip(v1, v2)):
        ax.text(i - w/2, a + 4, str(a), ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=BLUE)
        ax.text(i + w/2, b + 4, str(b), ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=ORANGE)
    ax.set_ylim(0, 195)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_title("Policy (d): Admin Time", fontsize=13, fontweight="bold", pad=10, color=DARK)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.7)
    _clean_axes(ax)
    ax.tick_params(left=False, labelleft=False)
    ax.grid(axis="y", alpha=0.15, zorder=0)
    ax.text(0.5, -0.18, "Baseline uses 0 admin slots",
            transform=ax.transAxes, ha="center", fontsize=9, fontstyle="italic", color=GREY)

    # ── Panel 2: Overbooking (Policy e) ──
    ax = axes[1]
    ax.set_facecolor(BG)
    labs = ["Week 1", "Week 2"]
    sched, over, rates = [427, 499], [34, 43], [7.3, 7.8]
    ax.bar(labs, sched, 0.5, label="Scheduled", color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(labs, over, 0.5, bottom=sched, label="Overbook (+)", color=RED, edgecolor="white", linewidth=0.8, zorder=3)
    for i, (s, o) in enumerate(zip(sched, over)):
        ax.text(i, s + o + 8, f"+{o} slots", ha="center", fontsize=11, fontweight="bold", color=RED)
        ax.text(i, s / 2, str(s), ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax.set_ylim(0, 600)
    ax.set_title("Policy (e): Overbooking", fontsize=13, fontweight="bold", pad=10, color=DARK)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.7)
    _clean_axes(ax)
    ax.tick_params(left=False, labelleft=False)
    ax.grid(axis="y", alpha=0.15, zorder=0)
    ax.text(0.5, -0.18, "No-show rate: 7.3% (W1), 7.8% (W2)",
            transform=ax.transAxes, ha="center", fontsize=9, fontstyle="italic", color=GREY)

    # ── Panel 3: Duration Uncertainty (Policy f) ──
    ax = axes[2]
    ax.set_facecolor(BG)
    bufs = ["+15 min", "+30 min"]
    d1, d2 = [61.9, 71.5], [66.6, 74.5]
    x = np.arange(len(bufs))
    w = 0.32
    ax.bar(x - w/2, d1, w, label="Week 1", color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + w/2, d2, w, label="Week 2", color=ORANGE, edgecolor="white", linewidth=0.8, zorder=3)
    for i, (a, b) in enumerate(zip(d1, d2)):
        ax.text(i - w/2, a + 1.5, f"{a}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=BLUE)
        ax.text(i + w/2, b + 1.5, f"{b}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=ORANGE)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Overlap Ratio (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(bufs, fontsize=10)
    ax.set_title("Policy (f): Duration Uncertainty", fontsize=13, fontweight="bold", pad=10, color=DARK)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.7)
    _clean_axes(ax)
    ax.grid(axis="y", alpha=0.15, zorder=0)
    ax.axhline(50, color=RED, linestyle="--", linewidth=1, alpha=0.5, zorder=2)
    ax.text(1.35, 51, "fragile", fontsize=8, color=RED, fontstyle="italic", ha="left")
    ax.text(0.5, -0.18, "Needs built-in slack or resequencing",
            transform=ax.transAxes, ha="center", fontsize=9, fontstyle="italic", color=GREY)

    fig.tight_layout(w_pad=2.5)
    path = OUT_DIR / "slide5_policy_boxes.png"
    fig.savefig(path, dpi=250, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3] {path.name}")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"Output folder: {OUT_DIR}\n")
    fig_cg_convergence()
    fig_switch_distance_bar()
    fig_policy_boxes()
    print(f"\nDone. All figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
