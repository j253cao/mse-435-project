"""Generate the three policy-box figures for slide 5."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

BG = "#D6DFD0"
DARK = "#2D3A2D"
BLUE = "#2196F3"
ORANGE = "#FF9800"
RED = "#E53935"
GREEN = "#43A047"
GREY = "#78909C"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"],
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})

fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8), facecolor=BG)

# ── Panel 1: Admin Time (Policy d) ──
ax = axes[0]
ax.set_facecolor(BG)

categories = ["Congested\nProvider-Days", "Admin Capacity\nSlots"]
w1 = [39, 156]
w2 = [40, 160]
x = np.arange(len(categories))
width = 0.32
ax.bar(x - width/2, w1, width, label="Week 1", color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
ax.bar(x + width/2, w2, width, label="Week 2", color=ORANGE, edgecolor="white", linewidth=0.8, zorder=3)
for i, (v1, v2) in enumerate(zip(w1, w2)):
    ax.text(i - width/2, v1 + 4, str(v1), ha="center", va="bottom", fontsize=11, fontweight="bold", color=BLUE)
    ax.text(i + width/2, v2 + 4, str(v2), ha="center", va="bottom", fontsize=11, fontweight="bold", color=ORANGE)
ax.set_ylim(0, 195)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.set_title("Policy (d): Admin Time", fontsize=13, fontweight="bold", pad=10, color=DARK)
ax.legend(fontsize=9, loc="upper left", framealpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#AAAAAA")
ax.spines["bottom"].set_color("#AAAAAA")
ax.tick_params(left=False, labelleft=False)
ax.grid(axis="y", alpha=0.15, zorder=0)

ax.text(0.5, -0.18, "Baseline uses 0 admin slots", transform=ax.transAxes,
        ha="center", fontsize=9, fontstyle="italic", color=GREY)

# ── Panel 2: Overbooking (Policy e) ──
ax = axes[1]
ax.set_facecolor(BG)

labels = ["Week 1", "Week 2"]
scheduled = [427, 499]
overbook = [34, 43]
rates = [7.3, 7.8]

bars1 = ax.bar(labels, scheduled, 0.5, label="Scheduled", color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
bars2 = ax.bar(labels, overbook, 0.5, bottom=scheduled, label="Overbook (+)", color=RED, edgecolor="white", linewidth=0.8, zorder=3)
for i, (s, o, r) in enumerate(zip(scheduled, overbook, rates)):
    ax.text(i, s + o + 8, f"+{o} slots", ha="center", fontsize=11, fontweight="bold", color=RED)
    ax.text(i, s/2, str(s), ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax.text(i, s + o/2, f"", ha="center", fontsize=9, color="white")

ax.set_ylim(0, 600)
ax.set_title("Policy (e): Overbooking", fontsize=13, fontweight="bold", pad=10, color=DARK)
ax.legend(fontsize=9, loc="upper left", framealpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#AAAAAA")
ax.spines["bottom"].set_color("#AAAAAA")
ax.tick_params(left=False, labelleft=False)
ax.grid(axis="y", alpha=0.15, zorder=0)

ax.text(0.5, -0.18, "No-show rate: 7.3% (W1), 7.8% (W2)", transform=ax.transAxes,
        ha="center", fontsize=9, fontstyle="italic", color=GREY)

# ── Panel 3: Duration Uncertainty (Policy f) ──
ax = axes[2]
ax.set_facecolor(BG)

buffers = ["+15 min", "+30 min"]
w1_vals = [61.9, 71.5]
w2_vals = [66.6, 74.5]
x = np.arange(len(buffers))
width = 0.32
b1 = ax.bar(x - width/2, w1_vals, width, label="Week 1", color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
b2 = ax.bar(x + width/2, w2_vals, width, label="Week 2", color=ORANGE, edgecolor="white", linewidth=0.8, zorder=3)
for i, (v1, v2) in enumerate(zip(w1_vals, w2_vals)):
    ax.text(i - width/2, v1 + 1.5, f"{v1}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color=BLUE)
    ax.text(i + width/2, v2 + 1.5, f"{v2}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color=ORANGE)

ax.set_ylim(0, 100)
ax.set_ylabel("Overlap Ratio (%)", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(buffers, fontsize=10)
ax.set_title("Policy (f): Duration Uncertainty", fontsize=13, fontweight="bold", pad=10, color=DARK)
ax.legend(fontsize=9, loc="upper left", framealpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#AAAAAA")
ax.spines["bottom"].set_color("#AAAAAA")
ax.grid(axis="y", alpha=0.15, zorder=0)

ax.axhline(50, color=RED, linestyle="--", linewidth=1, alpha=0.5, zorder=2)
ax.text(1.35, 51, "fragile", fontsize=8, color=RED, fontstyle="italic", ha="left")

ax.text(0.5, -0.18, "Needs built-in slack or resequencing", transform=ax.transAxes,
        ha="center", fontsize=9, fontstyle="italic", color=GREY)

fig.tight_layout(w_pad=2.5)
out = Path(__file__).resolve().parent / "outputs" / "section5_results" / "figures" / "fig_slide5_policy_boxes.png"
fig.savefig(out, dpi=250, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
