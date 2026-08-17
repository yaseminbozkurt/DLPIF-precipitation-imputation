"""
Figure 1 - DLPIF Architecture Diagram (v11 - parallel branches)

Redrawn from a sequential "Input -> Backbone -> Stage1 -> Stage2 ->
Calibration -> Output" chain to a parallel architecture: from the raw
corrupted input, a continuous base imputer reconstructs the six
non-precipitation variables while DLPIF's Stage 1/Stage 2 reconstruct
precipitation directly from the same raw corrupted records -- the two
branches never read each other's output and merge only at the final
combined state. This matches the proven backbone-independence result
(Table 9 / Section 5.8): DLPIF's PRECIP reconstruction is bit-identical
regardless of what, if anything, the base-imputer branch produces.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# -- palette --------------------------------------------------------------
C_NEUTRAL = "#2c3e6b"
C_BASE    = "#3a5585"
C_S1      = "#2d6a4f"
C_S2      = "#2a7a78"
C_BG      = "#ffffff"
C_ARROW   = "#1e2d52"
C_BASE_LBL = "#3a5585"
C_DLPIF_LBL = "#2d6a4f"
FONT      = "DejaVu Sans"

# -- canvas -----------------------------------------------------------------
FW, FH = 10.5, 4.6
fig, ax = plt.subplots(figsize=(FW, FH), dpi=300)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor(C_BG)

# -- title --------------------------------------------------------------
ax.text(FW / 2, FH - 0.12,
        "Decoupled Learning-Based Precipitation Imputation Framework (DLPIF)",
        fontsize=10.5, fontweight="bold", fontfamily=FONT,
        ha="center", va="top", color="#1a1a2e")
ax.text(FW / 2, FH - 0.42,
        "Parallel reconstruction: a continuous base imputer and DLPIF's occurrence-amount "
        "pathway operate independently on the same raw corrupted input",
        fontsize=6.6, fontfamily=FONT, fontstyle="italic",
        ha="center", va="top", color="#445566")

# -- geometry -----------------------------------------------------------
ROW_TOP_Y = 2.55      # base-imputer branch, box bottom
ROW_BOT_Y = 0.55       # DLPIF branch, box bottom
BOX_H = 0.95
IO_BOX_H = ROW_TOP_Y + BOX_H - ROW_BOT_Y   # tall Input/Output boxes span both rows

def box(x, y, w, h, color, lines, fontsizes, lw=0.85):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.022",
        facecolor=color, edgecolor="white",
        linewidth=lw, zorder=3)
    ax.add_patch(patch)
    n = len(lines)
    margin_frac = 0.07
    usable_h = h * (1 - 2 * margin_frac)
    step = usable_h / n
    base_y = y + h * (1 - margin_frac) - step / 2
    for j, (txt, fs) in enumerate(zip(lines, fontsizes)):
        yy = base_y - j * step
        t = ax.text(x + w / 2, yy, txt, fontsize=fs,
                    fontweight="bold" if j == 0 else "normal",
                    color="white", alpha=1.0 if j == 0 else 0.90,
                    fontfamily=FONT, ha="center", va="center", zorder=4)
        t.set_clip_path(patch)
    return dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2)

def diag_arrow(p0, p1, color=C_ARROW):
    arr = FancyArrowPatch(p0, p1, arrowstyle="-|>", color=color,
                          lw=1.05, mutation_scale=9,
                          connectionstyle="arc3,rad=0.0", zorder=5,
                          shrinkA=0, shrinkB=0)
    ax.add_patch(arr)

def harrow(p0, p1, color=C_ARROW):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1,
                                mutation_scale=9), zorder=5)

# -- Input (tall, left) --------------------------------------------------
IN_X, IN_W = 0.20, 1.35
inp = box(IN_X, ROW_BOT_Y, IN_W, IO_BOX_H, C_NEUTRAL,
         ["Input Data", "7 meteorological", "variables incl.", "precipitation,",
          "with missing values"],
         [6.6, 5.0, 5.0, 5.0, 5.0])

# -- Output (tall, right) -------------------------------------------------
OUT_W = 1.55
OUT_X = FW - 0.20 - OUT_W
out = box(OUT_X, ROW_BOT_Y, OUT_W, IO_BOX_H, C_NEUTRAL,
          ["Final Output", "Combined 7-variable", "reconstruction",
           "(6 vars + PRECIP)"],
          [6.6, 4.9, 4.9, 4.9])

# -- top branch: Base Imputation (single wide box) ------------------------
MID_X0 = IN_X + IN_W + 0.55
MID_X1 = OUT_X - 0.55
top_box = box(MID_X0, ROW_TOP_Y, MID_X1 - MID_X0, BOX_H, C_BASE,
             ["Base Imputation",
              "Any continuous multivariate imputer (e.g., WGAN-GP)",
              "Reconstructs the 6 non-precipitation variables",
              "(TMAX, TMIN, TMEAN, humidity, pressure, wind)"],
             [7.0, 5.0, 5.4, 5.2])

# -- bottom branch: Stage 1 -> Stage 2 ------------------------------------
BOT_GAP = 0.22
BOT_W = (MID_X1 - MID_X0 - BOT_GAP) / 2
s1_box = box(MID_X0, ROW_BOT_Y, BOT_W, BOX_H, C_S1,
            ["Stage 1: Occurrence RF", "Wet/dry classification",
             "Local precip. excluded", "Raw corrupted records only"],
            [7.0, 5.6, 5.3, 5.0])
s2_box = box(MID_X0 + BOT_W + BOT_GAP, ROW_BOT_Y, BOT_W, BOX_H, C_S2,
            ["Stage 2: Amount RF", "Conditional wet-day amount",
             "Local precip. hard-zeroed", "Dry -> 0; wet -> RF estimate"],
            [7.0, 5.4, 5.2, 5.0])

harrow((s1_box['x'] + s1_box['w'] + 0.02, s1_box['cy']),
      (s2_box['x'] - 0.02, s2_box['cy']))

# -- split arrows: Input -> top branch / bottom branch --------------------
diag_arrow((inp['x'] + inp['w'] + 0.02, inp['cy'] + 0.35),
          (top_box['x'] - 0.02, top_box['cy']))
diag_arrow((inp['x'] + inp['w'] + 0.02, inp['cy'] - 0.35),
          (s1_box['x'] - 0.02, s1_box['cy']))

# -- merge arrows: top branch / bottom branch -> Output --------------------
diag_arrow((top_box['x'] + top_box['w'] + 0.02, top_box['cy']),
          (out['x'] - 0.02, out['cy'] + 0.35))
diag_arrow((s2_box['x'] + s2_box['w'] + 0.02, s2_box['cy']),
          (out['x'] - 0.02, out['cy'] - 0.35))

# -- branch labels ---------------------------------------------------------
ax.text(top_box['cx'], ROW_TOP_Y + BOX_H + 0.10,
        "independent of DLPIF — imputes context variables only",
        fontsize=5.8, color=C_BASE_LBL, fontfamily=FONT, fontstyle="italic",
        ha="center", va="bottom", zorder=4)
ax.text((s1_box['cx'] + s2_box['cx']) / 2, ROW_BOT_Y - 0.10,
        "DLPIF — reconstructs PRECIP directly; independent of the base imputer's output",
        fontsize=5.8, color=C_DLPIF_LBL, fontfamily=FONT, fontstyle="italic",
        ha="center", va="top", zorder=4)

# -- save -------------------------------------------------------------------
out_dir  = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, "fig1_dlpif_architecture.pdf")
png_path = os.path.join(out_dir, "fig1_dlpif_architecture.png")

fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved:\n  {pdf_path}\n  {png_path}")
