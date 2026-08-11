"""
Figure 12 - RQ4b hero figure: does the validation-selected, frozen
interaction gate recover NetBlock-30d's catastrophic collapse without
materially disturbing ordinary scenarios? Rebuilt from
results/rq4_context_availability/gate_applied/{original_scenarios_gated,
original_scenarios_bootstrap}.csv.

(a) Delta F1 = F1_Gated - F1_DLPIF per scenario, with the seed=42 paired
    block-bootstrap 95% CI as error bars (n_boot=10000). NetBlock-30d's
    bar is annotated with its point estimate; Block-30d's small but
    statistically significant COST (p=0.008) is annotated too -- the gate
    is not a free win, and that is reported directly on the figure, not
    only in the table.
(b) Gate trigger rate per scenario -- the evidence that recovery comes
    from selective activation (0-4.2% under every ordinary/MNAR scenario)
    rather than blanket fallback use, only reaching 100% at total context
    collapse.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATED_DIR = os.path.join(REPO, "results", "rq4_context_availability", "gate_applied")
scen_df = pd.read_csv(os.path.join(GATED_DIR, "original_scenarios_gated.csv"))
boot_df = pd.read_csv(os.path.join(GATED_DIR, "original_scenarios_bootstrap.csv"))

SCEN_ORDER = ["10pct", "mar_meteo", "mnar_wet",
             "mnar_intensity_moderate", "mnar_intensity_severe",
             "block7d", "block30d", "netblock30d"]
SCEN_LABELS = ["MCAR-10", "MAR-\nMeteo", "MNAR-\nWet",
              "MNAR-Int.\nModerate", "MNAR-Int.\nSevere",
              "Block-7d", "Block-30d", "NetBlock-\n30d"]

COL_POS = "#009E73"   # bluish green -- improvement
COL_NEG = "#D55E00"   # vermillion -- cost
COL_ZERO = "#888888"
COL_TRIGGER = "#0072B2"

mean_delta = scen_df.groupby("scenario")["delta_f1"].mean().reindex(SCEN_ORDER)
trigger = scen_df.groupby("scenario")["gate_trigger_rate"].mean().reindex(SCEN_ORDER) * 100
boot_idx = boot_df.set_index("scenario").reindex(SCEN_ORDER)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

# ── Panel (a): Delta F1 with bootstrap CI ───────────────────────────────
x = np.arange(len(SCEN_ORDER))
colors = [COL_POS if v > 0.005 else (COL_NEG if v < -0.005 else COL_ZERO) for v in mean_delta]
yerr_low = (boot_idx["delta_f1"] - boot_idx["delta_f1_ci_low"]).to_numpy()
yerr_high = (boot_idx["delta_f1_ci_high"] - boot_idx["delta_f1"]).to_numpy()
ax1.bar(x, mean_delta, color=colors, edgecolor="white", linewidth=0.8,
       yerr=[yerr_low, yerr_high], capsize=4, ecolor="#333333")
ax1.axhline(0, color="#333333", lw=1.0)
ax1.set_xticks(x); ax1.set_xticklabels(SCEN_LABELS, fontsize=9)
ax1.set_ylabel("$\\Delta$F1 = F1$_{Gated}$ - F1$_{DLPIF}$", fontsize=11)
ax1.set_title("(a) Recovery vs preservation\n(seed=42 bootstrap 95% CI)", fontsize=12, fontweight="bold")
ax1.grid(True, axis="y", alpha=0.3)
ax1.set_ylim(-0.18, 0.68)

nb_i = SCEN_ORDER.index("netblock30d")
ax1.text(nb_i, 0.25, f"+{mean_delta.iloc[nb_i]:.3f}\n(p<.0001)",
        fontsize=10, fontweight="bold", color="white", ha="center", va="center")
b30_i = SCEN_ORDER.index("block30d")
ax1.annotate(f"{mean_delta.iloc[b30_i]:.3f} (p=0.008)", xy=(b30_i, mean_delta.iloc[b30_i]),
            xytext=(b30_i + 0.35, mean_delta.iloc[b30_i] - 0.10), fontsize=8.5, color=COL_NEG,
            arrowprops=dict(arrowstyle="->", color=COL_NEG, lw=1.0))

# ── Panel (b): gate trigger rate ────────────────────────────────────────
ax2.bar(x, trigger, color=COL_TRIGGER, edgecolor="white", linewidth=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(SCEN_LABELS, fontsize=9)
ax2.set_ylabel("Gate trigger rate (%)", fontsize=11)
ax2.set_title("(b) Selectivity: how often the gate\nactually routes to fallback", fontsize=12, fontweight="bold")
ax2.grid(True, axis="y", alpha=0.3)
for xi, v in zip(x, trigger):
    if v > 0.5:
        ax2.text(xi, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8.5)

fig.suptitle("A validation-selected interaction gate recovers catastrophic collapse\n"
             "while leaving ordinary scenarios almost untouched",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.005,
        "Gate frozen on validation data BEFORE this test run (tau_neighbor=0.75, tau_local=0.9, "
        "fallback=Linear interpolation); no threshold was tuned on these results.",
        ha="center", fontsize=8.5, style="italic", color="#444444")
fig.tight_layout(rect=[0, 0.03, 1, 0.90])

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(out_dir, f"Figure_12_GateRecovery.{ext}"),
               bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("Saved Figure 12 (png/pdf/svg)")
