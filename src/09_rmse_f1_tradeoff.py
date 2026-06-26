# -*- coding: utf-8 -*-
"""
09_rmse_f1_tradeoff.py
======================
Produces a 2x2 RMSE–F1 trade-off scatter figure (one panel per scenario).

x-axis : Wet-day RMSE (mm)
y-axis : F1 (wet-day classification)
points : methods (mean ± std across seeds where applicable)
panels : 4 masking scenarios

Key message:
  Methods like Mean and WGAN-GP may show moderate RMSE (some wet days predicted
  with non-zero values) but their F1 collapses because they produce 100% false wet
  rate. The Pareto-optimal region (low RMSE AND high F1) is exclusively occupied
  by DLPIF.
"""

import os, sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass
warnings.filterwarnings("ignore")

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SCENARIO_ORDER = ["10pct", "20pct", "block7d", "block30d"]
SCENARIO_LABELS = {
    "10pct":    "Random 10%",
    "20pct":    "Random 20%",
    "block7d":  "Block 7d",
    "block30d": "Block 30d",
}

# Method family → (display label, pattern in CSV, marker, color, zorder)
METHOD_FAMILIES = [
    ("Mean",          "mean",               "s",  "#9E9E9E", 2),
    ("Linear",        "linear",             "^",  "#78909C", 2),
    ("KNN",           "knn",                "D",  "#90A4AE", 2),
    ("MICE",          "mice",               "P",  "#B0BEC5", 2),
    ("WGAN-GP (raw)", "WGAN-GP_raw",        "o",  "#EF5350", 3),
    ("PrecipFix",     "PrecipFix",          "h",  "#FF7043", 3),
    ("Precip2Stage",  "Precip2Stage_clean", "v",  "#42A5F5", 4),
    ("DLPIF",         "AmountRF_clean",     "*",  "#1B5E20", 5),
]

MARKER_SIZE   = {"s":70,"^":70,"D":60,"P":70,"o":70,"h":70,"v":70,"*":200}
LABEL_OFFSETS = {
    "Mean":          ( 0.08, -0.018),
    "Linear":        ( 0.08,  0.006),
    "KNN":           ( 0.08, -0.012),
    "MICE":          ( 0.08,  0.010),
    "WGAN-GP (raw)": ( 0.08,  0.005),
    "PrecipFix":     ( 0.08,  0.005),
    "Precip2Stage":  (-0.10, -0.022),
    "DLPIF":         ( 0.08,  0.008),
}

BG_COLOR   = "#0D1117"
PANEL_COLOR= "#161B22"
GRID_COLOR = "#21262D"
TEXT_COLOR = "#E6EDF3"
SPINE_COLOR= "#30363D"


def load_and_aggregate():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "clean_full_evaluation.csv"))

    rows = []
    for label, pattern, marker, color, zo in METHOD_FAMILIES:
        is_det = (pattern in ("mean", "linear", "knn", "mice"))

        if is_det:
            sub = df[df["method"] == pattern]
        else:
            sub = df[df["method"].str.contains(pattern, regex=False)]

        for scen in SCENARIO_ORDER:
            scen_sub = sub[sub["scenario"] == scen]
            if scen_sub.empty:
                continue

            f1_vals   = scen_sub["f1"].dropna().values
            rmse_vals = scen_sub["rmse_wet"].dropna().values

            if len(f1_vals) == 0 or len(rmse_vals) == 0:
                continue

            rows.append(dict(
                scenario      = scen,
                label         = label,
                marker        = marker,
                color         = color,
                zorder        = zo,
                f1_mean       = np.mean(f1_vals),
                f1_std        = np.std(f1_vals, ddof=1) if len(f1_vals) > 1 else 0.0,
                rmse_mean     = np.mean(rmse_vals),
                rmse_std      = np.std(rmse_vals, ddof=1) if len(rmse_vals) > 1 else 0.0,
                n_seeds       = len(f1_vals),
                deterministic = is_det,
            ))

    return pd.DataFrame(rows)

def draw_panel(ax, data, scenario, show_legend=False, show_xlabel=True,
               show_ylabel=True, panel_label=""):
    """Draw one scatter panel for a given scenario."""
    sub = data[data["scenario"] == scenario]

    # Dark panel background
    ax.set_facecolor(PANEL_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # Reference lines: median RMSE and F1 = 0.5
    all_rmse = sub["rmse_mean"].values
    if len(all_rmse) > 0:
        ax.axhline(0.5, color="#444D56", linewidth=0.8, linestyle="--", zorder=1)

    # Scatter each method
    for _, row in sub.sort_values("zorder").iterrows():
        # Error bars for seeded methods
        xerr = row["rmse_std"] if row["rmse_std"] > 1e-6 else None
        yerr = row["f1_std"]   if row["f1_std"]   > 1e-6 else None

        if xerr is not None or yerr is not None:
            ax.errorbar(
                row["rmse_mean"], row["f1_mean"],
                xerr=xerr, yerr=yerr,
                fmt="none",
                ecolor=row["color"], elinewidth=1.2, capsize=3,
                capthick=1.0, alpha=0.7, zorder=row["zorder"],
            )

        ms = MARKER_SIZE.get(row["marker"], 70)
        ax.scatter(
            row["rmse_mean"], row["f1_mean"],
            marker=row["marker"], s=ms,
            color=row["color"],
            edgecolors="white" if row["label"] == "DLPIF" else row["color"],
            linewidths=1.5 if row["label"] == "DLPIF" else 0,
            zorder=row["zorder"],
            alpha=1.0,
        )

        # Label
        dx, dy = LABEL_OFFSETS.get(row["label"], (0.06, 0.006))
        ax.annotate(
            row["label"],
            xy=(row["rmse_mean"], row["f1_mean"]),
            xytext=(row["rmse_mean"] + dx, row["f1_mean"] + dy),
            fontsize=7, color=row["color"], fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground=PANEL_COLOR)],
            zorder=10,
        )

    # "Pareto-optimal" region annotation on first panel only
    if scenario == "10pct":
        ax.annotate(
            "Pareto-optimal\n(low RMSE + high F1)",
            xy=(5.12, 0.742), xytext=(5.5, 0.65),
            fontsize=6.5, color="#1B5E20", alpha=0.9,
            arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.0),
            path_effects=[pe.withStroke(linewidth=2, foreground=PANEL_COLOR)],
        )

    # Panel letter + title
    ax.set_title(
        f"  {panel_label}  {SCENARIO_LABELS[scenario]}",
        loc="left", fontsize=9.5, color=TEXT_COLOR, fontweight="bold", pad=4,
    )

    if show_xlabel:
        ax.set_xlabel("Wet-day RMSE (mm)", fontsize=8.5, color=TEXT_COLOR, labelpad=4)
    if show_ylabel:
        ax.set_ylabel("F1 (wet-day occurrence)", fontsize=8.5, color=TEXT_COLOR, labelpad=4)

    # Axis limits with padding
    x_min = max(0, sub["rmse_mean"].min() - 0.5)
    x_max = sub["rmse_mean"].max() + 1.2
    y_min = max(0, sub["f1_mean"].min() - 0.06)
    y_max = min(1.0, sub["f1_mean"].max() + 0.07)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

def build_legend_handles():
    handles = []
    for label, pattern, marker, color, _ in METHOD_FAMILIES:
        ms = MARKER_SIZE.get(marker, 70)
        h = Line2D([0], [0],
                   marker=marker, color="none",
                   markerfacecolor=color,
                   markeredgecolor="white" if label == "DLPIF" else color,
                   markeredgewidth=1.2 if label == "DLPIF" else 0,
                   markersize=np.sqrt(ms) * 0.75,
                   label=label)
        handles.append(h)
    return handles

def main():
    print("=" * 65)
    print("  09_rmse_f1_tradeoff.py — RMSE–F1 Trade-off Figure")
    print("=" * 65)

    data = load_and_aggregate()
    print(f"  {len(data)} (method × scenario) points loaded.")

    fig = plt.figure(figsize=(12, 9), facecolor=BG_COLOR)
    fig.subplots_adjust(hspace=0.38, wspace=0.30,
                        left=0.07, right=0.97,
                        top=0.90, bottom=0.10)

    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4),
    ]
    panel_letters = ["(a)", "(b)", "(c)", "(d)"]

    for ax, scen, letter in zip(axes, SCENARIO_ORDER, panel_letters):
        show_x = scen in ("block7d", "block30d")
        show_y = scen in ("10pct",   "block7d")
        draw_panel(ax, data, scen,
                   show_xlabel=show_x, show_ylabel=show_y,
                   panel_label=letter)

    # ── Main title ────────────────────────────────────────────────────────────
    fig.text(0.50, 0.96,
             "RMSE–F1 Trade-off: Continuous Accuracy vs. Event-Level Reconstruction",
             ha="center", va="top",
             fontsize=12, color=TEXT_COLOR, fontweight="bold")
    fig.text(0.50, 0.93,
             "Error bars show ±1 std across seeds 42, 123, 456  ·  "
             "Methods in upper-left quadrant achieve both low RMSE and high F1",
             ha="center", va="top", fontsize=8.5, color="#8B949E")

    # ── Shared legend (bottom) ────────────────────────────────────────────────
    handles = build_legend_handles()
    fig.legend(
        handles=handles, ncol=8,
        loc="lower center", bbox_to_anchor=(0.50, 0.01),
        frameon=True, framealpha=0.15,
        facecolor="#21262D", edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR, fontsize=8,
        handletextpad=0.4, columnspacing=0.8,
    )

    out_path = os.path.join(FIG_DIR, "rmse_f1_tradeoff.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)

    print(f"\n  -> figures/rmse_f1_tradeoff.png")
    print(f"     Path: {out_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
