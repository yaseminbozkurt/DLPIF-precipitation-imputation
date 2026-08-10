"""
Figure 10 - Split-conformal prediction interval coverage (RQ3), two panels.
Rebuilt from results/rq3_conformal/{test_conformal_oracle,test_conformal_
endtoend}.csv.

(a) Oracle (Stage-2 alone, ground-truth-wet positions) vs end-to-end
    (frozen Stage-1 decision gates Stage-2; predicted-dry -> degenerate
    {0} interval) PICP per scenario, mean over 3 seeds, vs the 90%
    nominal target. NetBlock-30d is the clearest case for the argument
    motivating RQ4: Stage-2 alone is close to nominal, but Stage-1's
    occurrence collapse (F1=0) drags end-to-end coverage down substantially.
(b) Extreme-event (p95-conditional) PICP, oracle population only, with the
    number of masked p95 test observations the estimate is based on
    annotated above each bar -- several scenarios have single-digit n and
    are descriptive only (see caption); MNAR-Intensity-Moderate (n=24) and
    -Severe (n=47) are well-powered and both land far below nominal.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RQ3 = os.path.join(REPO, "results", "rq3_conformal")
oracle = pd.read_csv(os.path.join(RQ3, "test_conformal_oracle.csv"))
e2e = pd.read_csv(os.path.join(RQ3, "test_conformal_endtoend.csv"))

COL_ORACLE = "#56B4E9"    # sky blue
COL_E2E = "#D55E00"       # vermillion
COL_NOMINAL = "#333333"
COL_P95 = "#CC79A7"       # reddish purple

SCEN_ORDER = ["10pct", "mar_meteo", "mnar_wet",
             "mnar_intensity_moderate", "mnar_intensity_severe",
             "block7d", "block30d", "netblock30d"]
SCEN_LABELS = ["MCAR-10", "MAR-Meteo", "MNAR-Wet",
              "MNAR-Int.\nModerate", "MNAR-Int.\nSevere",
              "Block-7d", "Block-30d", "NetBlock-30d"]

oracle_agg = oracle.groupby("scenario")[["picp", "picp_p95", "n_p95"]].mean().reindex(SCEN_ORDER)
e2e_agg = e2e.groupby("scenario")["picp"].mean().reindex(SCEN_ORDER)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

# ── Panel (a): oracle vs end-to-end PICP ────────────────────────────────
x = np.arange(len(SCEN_ORDER))
w = 0.36
ax1.bar(x - w / 2, oracle_agg["picp"], width=w, color=COL_ORACLE,
       edgecolor="white", linewidth=0.8, label="Oracle (Stage-2 alone)")
ax1.bar(x + w / 2, e2e_agg, width=w, color=COL_E2E,
       edgecolor="white", linewidth=0.8, label="End-to-end (Stage-1 gated)")
ax1.axhline(0.90, ls="--", lw=1.6, color=COL_NOMINAL, label="90% nominal target")
ax1.set_xticks(x); ax1.set_xticklabels(SCEN_LABELS, fontsize=9)
ax1.set_ylabel("PICP (empirical coverage)", fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.set_title("(a) Oracle vs end-to-end coverage", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9, loc="lower left", framealpha=1.0)
ax1.grid(True, axis="y", alpha=0.3)

# Annotate the netblock30d oracle/end-to-end gap -- the RQ4 bridge.
nb_idx = SCEN_ORDER.index("netblock30d")
ax1.annotate("", xy=(nb_idx + w / 2, e2e_agg.iloc[nb_idx] + 0.02),
            xytext=(nb_idx - w / 2, oracle_agg["picp"].iloc[nb_idx] - 0.02),
            arrowprops=dict(arrowstyle="->", color=COL_NOMINAL, lw=1.3))
ax1.text(nb_idx, oracle_agg["picp"].iloc[nb_idx] + 0.05,
        "Stage-1 collapse\ndrags coverage down", fontsize=7.5, ha="center",
        style="italic", color="#444444")

# ── Panel (b): p95-conditional PICP (oracle), with n annotated ─────────
bars = ax2.bar(x, oracle_agg["picp_p95"], width=0.6, color=COL_P95,
               edgecolor="white", linewidth=0.8)
ax2.axhline(0.90, ls="--", lw=1.6, color=COL_NOMINAL, label="90% nominal target")
for xi, (picp_p95, n) in enumerate(zip(oracle_agg["picp_p95"], oracle_agg["n_p95"])):
    if np.isnan(picp_p95):
        continue
    ax2.text(xi, picp_p95 + 0.03, f"n={int(n)}", ha="center", fontsize=8.5)
ax2.set_xticks(x); ax2.set_xticklabels(SCEN_LABELS, fontsize=9)
ax2.set_ylabel("PICP$_{p95}$ (extreme-event coverage, oracle)", fontsize=11)
ax2.set_ylim(0, 1.05)
ax2.set_title("(b) Extreme-event (≥p95) coverage", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9, loc="upper right", framealpha=1.0)
ax2.grid(True, axis="y", alpha=0.3)

fig.suptitle("Stage-2 split-conformal interval coverage under increasing MNAR severity",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.005,
        "Panel (b): estimates based on fewer than ~15 masked p95 events (all scenarios except\n"
        "MNAR-Intensity-Moderate/Severe) should be interpreted descriptively, not as precise rates.",
        ha="center", fontsize=8.5, style="italic", color="#444444")
fig.tight_layout(rect=[0, 0.04, 1, 0.94])

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(out_dir, f"Figure_10_ConformalCoverage.{ext}"),
               bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("Saved Figure 10 (png/pdf/svg)")
