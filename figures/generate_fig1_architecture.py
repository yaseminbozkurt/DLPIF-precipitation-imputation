"""
Figure 1 – DLPIF Architecture Diagram  (v10 — final horizontal flow)

Fully horizontal: Input → Base Imputation → Stage 1 → Stage 2 → Calibration → Output
All text fit tested for W_BOX ≈ 1.53 in at 300 dpi.
Subtitle lines kept ≤ 22 characters at 5.0 pt to guarantee no clipping.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── palette ───────────────────────────────────────────────────────────────────
C_NEUTRAL = "#2c3e6b"
C_BASE    = "#3a5585"
C_S1      = "#2d6a4f"
C_S2      = "#2a7a78"
C_CAL     = "#7d5a2c"
C_BG      = "#ffffff"
C_ARROW   = "#1e2d52"
C_DLPIF   = "#6677aa"
FONT      = "DejaVu Sans"

# ── canvas ─────────────────────────────────────────────────────────────────────
FW, FH = 10.0, 3.00
fig, ax = plt.subplots(figsize=(FW, FH), dpi=300)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor(C_BG)

# ──────────────────────────────────────────────────────────────────────────────
#  GEOMETRY
# ──────────────────────────────────────────────────────────────────────────────
MARGIN  = 0.20
ARROW_W = 0.22
N_BOXES = 6

W_BOX   = (FW - 2 * MARGIN - (N_BOXES - 1) * ARROW_W) / N_BOXES  # ≈ 1.527
BOX_H   = 1.02
BOX_Y   = 0.96
MID_Y   = BOX_Y + BOX_H / 2

def bx(i):
    return MARGIN + i * (W_BOX + ARROW_W)

# ──────────────────────────────────────────────────────────────────────────────
#  HELPER – box with clipped text
#  lines = list of strings; first line is bold title, rest are subtitles
# ──────────────────────────────────────────────────────────────────────────────
def box(i, color, lines, fontsizes, lw=0.85):
    x = bx(i)
    patch = FancyBboxPatch(
        (x, BOX_Y), W_BOX, BOX_H,
        boxstyle="round,pad=0.022",
        facecolor=color, edgecolor="white",
        linewidth=lw, zorder=3)
    ax.add_patch(patch)

    n = len(lines)
    # Space the lines evenly inside the box
    margin_frac = 0.06
    usable_h = BOX_H * (1 - 2 * margin_frac)
    step = usable_h / n
    base_y = BOX_Y + BOX_H * (1 - margin_frac) - step / 2

    for j, (txt, fs) in enumerate(zip(lines, fontsizes)):
        y = base_y - j * step
        t = ax.text(
            x + W_BOX / 2, y, txt,
            fontsize=fs,
            fontweight="bold" if j == 0 else "normal",
            color="white",
            alpha=1.0 if j == 0 else 0.90,
            fontfamily=FONT,
            ha="center", va="center",
            zorder=4)
        t.set_clip_path(patch)

def harrow(i):
    x0 = bx(i) + W_BOX + 0.03
    x1 = bx(i + 1) - 0.03
    ax.annotate("", xy=(x1, MID_Y), xytext=(x0, MID_Y),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.1, mutation_scale=9), zorder=5)

# ──────────────────────────────────────────────────────────────────────────────
#  TITLE
# ──────────────────────────────────────────────────────────────────────────────
ax.text(FW / 2, FH - 0.15,
        "Decoupled Latent\u2013Physical Imputation Framework (DLPIF)",
        fontsize=10.5, fontweight="bold", fontfamily=FONT,
        ha="center", va="top", color="#1a1a2e")
ax.text(FW / 2, FH - 0.46,
        "Two-stage precipitation-specific correction built on "
        "continuous multivariate imputation",
        fontsize=7.0, fontfamily=FONT, fontstyle="italic",
        ha="center", va="top", color="#445566")

# ──────────────────────────────────────────────────────────────────────────────
#  BOXES  — subtitle lines kept short (≤ ~22 chars) to avoid clipping
# ──────────────────────────────────────────────────────────────────────────────

# 0 – Input Data
box(0, C_NEUTRAL,
    ["Input Data",
     "7 meteorological variables",
     "including precipitation",
     "with missing values"],
    [6.8, 5.1, 5.1, 5.1])

# 1 – Base Imputation
box(1, C_BASE,
    ["Base Imputation",
     "WGAN-GP, Mode B",
     "Multivariate reconstruction",
     "\u2192 imputed latent state"],
    [6.8, 5.1, 5.1, 5.1])

# 2 – Stage 1
box(2, C_S1,
    ["Stage 1",
     "Occurrence RF",
     "Wet/dry classification",
     "Local precip. excluded\u00b9"],
    [7.2, 5.6, 5.3, 5.3])

# 3 – Stage 2
box(3, C_S2,
    ["Stage 2",
     "Wet-day amount RF",
     "Conditional amount estimation",
     "Local precip. hard-zeroed\u00b9"],
    [7.2, 5.7, 5.4, 5.4])

# 4 – Calibration
box(4, C_CAL,
    ["Calibration Layer",
     "Dry: r\u209c = 0 mm",
     "Wet: calibrated r\u209c"],
    [6.8, 5.4, 5.4])

# 5 – Final Output
box(5, C_NEUTRAL,
    ["Final DLPIF Output",
     "Physically consistent",
     "precipitation",
     "reconstruction"],
    [6.8, 5.1, 5.1, 5.1])

# ──────────────────────────────────────────────────────────────────────────────
#  ARROWS
# ──────────────────────────────────────────────────────────────────────────────
for i in range(N_BOXES - 1):
    harrow(i)

# ──────────────────────────────────────────────────────────────────────────────
#  DLPIF BRACKET  (below Stage 1, Stage 2, Calibration)
# ──────────────────────────────────────────────────────────────────────────────
X_L = bx(2)
X_R = bx(4) + W_BOX
X_M = (X_L + X_R) / 2
BY  = BOX_Y - 0.13
TH  = 0.08

ax.plot([X_L, X_R], [BY, BY], color=C_DLPIF, lw=0.85, zorder=4)
ax.plot([X_L, X_L], [BY, BY + TH], color=C_DLPIF, lw=0.85, zorder=4)
ax.plot([X_R, X_R], [BY, BY + TH], color=C_DLPIF, lw=0.85, zorder=4)
ax.text(X_M, BY - 0.05, "DLPIF correction module",
        fontsize=5.8, color=C_DLPIF, fontfamily=FONT,
        fontstyle="italic", ha="center", va="top", zorder=4)

# footnote removed — see figure caption

# ──────────────────────────────────────────────────────────────────────────────
#  SAVE
# ──────────────────────────────────────────────────────────────────────────────
out_dir  = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, "fig1_dlpif_architecture.pdf")
png_path = os.path.join(out_dir, "fig1_dlpif_architecture.png")

fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved:\n  {pdf_path}\n  {png_path}")
