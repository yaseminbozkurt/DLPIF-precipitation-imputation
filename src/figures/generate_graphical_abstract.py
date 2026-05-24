"""
Graphical Abstract Generator for DLPIF paper
EMS (Environmental Modelling & Software)
Output: 2000 x 800 px  PNG + PDF

Usage: run from the repository root:
    python src/figures/generate_graphical_abstract.py
Output is written to figures/figures/ (same directory as submitted manuscript figures).
"""

import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Canvas ─────────────────────────────────────────────────────────────────────
W_PX, H_PX = 2000, 800
DPI = 200
fig_w = W_PX / DPI    # 10 in
fig_h = H_PX / DPI    # 4 in

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# ── Colours ────────────────────────────────────────────────────────────────────
BG_TOP    = '#0d1b2a'
BG_BOT    = '#162032'
ACCENT1   = '#2980b9'
ACCENT2   = '#1abc9c'
WHITE     = '#ffffff'
LIGHT     = '#ecf0f1'
ARROW_CLR = '#7fb3d3'

BOX_PAIRS = [
    ('#12304e', '#2471a3'),   # 0 Input data
    ('#0f3d2e', '#1abc9c'),   # 1 Base imputation
    ('#2c1654', '#8e44ad'),   # 2 Stage 1
    ('#4a2c00', '#e67e22'),   # 3 Stage 2
    ('#133052', '#2980b9'),   # 4 Physical consistency
    ('#0c2d1c', '#27ae60'),   # 5 Final output
]

STEP_COLORS = ['#2471a3', '#1abc9c', '#8e44ad', '#e67e22', '#2980b9', '#27ae60']

BOX_LABELS = [
    'Incomplete\nmeteorological\ndata',
    'Base\nimputation',
    'Stage 1\nWet/dry\nclassification',
    'Stage 2\nWet-day amount\nestimation',
    'Physical\nconsistency',
    'Final precipitation\nreconstruction',
]

# Step number labels shown inside coloured circle on each box
STEP_NUMS = ['0', '1', '2', '3', '4', '5']

# ── Background ────────────────────────────────────────────────────────────────
ax.add_patch(mpatches.Rectangle((0, 0), 1, 1,
    color=BG_TOP, zorder=0, transform=ax.transAxes))

grad_bg = np.linspace(0, 1, 200).reshape(200, 1)
bg_cmap = LinearSegmentedColormap.from_list('bg', [BG_TOP, BG_BOT])
ax.imshow(grad_bg, aspect='auto', extent=[0, 1, 0, 1],
          cmap=bg_cmap, origin='lower', zorder=1, alpha=0.55,
          transform=ax.transAxes)

# ── Top accent strip ───────────────────────────────────────────────────────────
accent_grad = np.linspace(0, 1, 600).reshape(1, 600)
accent_cmap = LinearSegmentedColormap.from_list('acc', [ACCENT1, ACCENT2, '#8e44ad'])
ax.imshow(accent_grad, aspect='auto', extent=[0, 1, 0.965, 0.990],
          cmap=accent_cmap, origin='lower', zorder=10,
          transform=ax.transAxes)

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.880,
        'DLPIF: Physically Informed Precipitation Imputation',
        ha='center', va='center',
        fontsize=12, fontweight='bold', color=WHITE, zorder=20,
        transform=ax.transAxes)

ax.text(0.5, 0.818,
        'Decoupling precipitation occurrence and amount reconstruction',
        ha='center', va='center',
        fontsize=8.5, color=ACCENT2, zorder=20,
        fontstyle='italic', transform=ax.transAxes)

# ── Box layout ─────────────────────────────────────────────────────────────────
N   = 6
MX  = 0.018       # left/right margin
BOX_Y  = 0.455    # vertical centre of flow boxes
BOX_H  = 0.365    # box height
ARW    = 0.020    # gap for arrow
TOT_W  = 1 - 2*MX
BOX_W  = (TOT_W - (N-1)*ARW) / N

# ── Helpers ────────────────────────────────────────────────────────────────────
def add_rounded_rect(ax, x, y, w, h, color, ec, lw, alpha, rx=0.01, zorder=5):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0,rounding_size={rx}',
                       fc=color, ec=ec, lw=lw, alpha=alpha,
                       zorder=zorder, transform=ax.transAxes)
    ax.add_patch(p)
    return p

def draw_box(i, x_left):
    dark_c, light_c = BOX_PAIRS[i]
    step_c = STEP_COLORS[i]
    x0 = x_left
    y0 = BOX_Y - BOX_H / 2

    # Shadow
    add_rounded_rect(ax, x0+0.003, y0-0.009, BOX_W, BOX_H,
                     'black', 'none', 0, 0.30, zorder=4)

    # Gradient fill via a narrow imshow tile
    box_grad = np.linspace(0, 1, 120).reshape(120, 1)
    box_cmap = LinearSegmentedColormap.from_list('box', [dark_c, light_c])
    ax.imshow(box_grad, aspect='auto',
              extent=[x0, x0+BOX_W, y0, y0+BOX_H],
              cmap=box_cmap, origin='lower', zorder=5, alpha=0.96,
              transform=ax.transAxes)

    # Border
    border_lw = 2.0 if i == N-1 else 1.3
    border_ec = ACCENT2 if i == N-1 else light_c
    add_rounded_rect(ax, x0, y0, BOX_W, BOX_H,
                     'none', border_ec, border_lw, 0.88, zorder=6)

    # Top highlight sheen
    rx = 0.01
    ax.plot([x0+rx, x0+BOX_W-rx], [y0+BOX_H, y0+BOX_H],
            color='white', lw=0.7, alpha=0.20, zorder=7,
            transform=ax.transAxes, solid_capstyle='round')

    # Step indicator circle  (top centre of box)
    cx = x0 + BOX_W / 2
    circle_y = y0 + BOX_H - 0.048
    circ = plt.Circle((cx, circle_y), 0.022,
                       transform=ax.transAxes,
                       fc=step_c, ec='white', lw=0.8,
                       zorder=8, clip_on=False)
    ax.add_patch(circ)
    ax.text(cx, circle_y, STEP_NUMS[i],
            ha='center', va='center',
            fontsize=6.5, fontweight='bold', color=WHITE, zorder=9,
            transform=ax.transAxes)

    # Label text
    label_y = BOX_Y - 0.025
    ax.text(cx, label_y, BOX_LABELS[i],
            ha='center', va='center',
            fontsize=7.0, color=WHITE, fontweight='bold',
            zorder=8, transform=ax.transAxes,
            multialignment='center', linespacing=1.45)

    # Small "DLPIF" badge on final box
    if i == N-1:
        badge_y = y0 + 0.022
        add_rounded_rect(ax, cx-0.04, badge_y, 0.080, 0.030,
                         ACCENT2, 'none', 0, 0.18, rx=0.005, zorder=7)
        ax.text(cx, badge_y+0.015, 'DLPIF output',
                ha='center', va='center', fontsize=5.6,
                color=ACCENT2, fontstyle='italic',
                zorder=9, transform=ax.transAxes)

def draw_arrow(x_left):
    xm = x_left + ARW / 2
    ax.annotate('',
                xy=(x_left + ARW, BOX_Y), xytext=(x_left, BOX_Y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(
                    arrowstyle='->', color=ARROW_CLR,
                    lw=1.8, mutation_scale=16),
                zorder=8)

# ── Draw all boxes and arrows ──────────────────────────────────────────────────
for i in range(N):
    x_start = MX + i * (BOX_W + ARW)
    draw_box(i, x_start)
    if i < N-1:
        draw_arrow(x_start + BOX_W)

# ── Footer bar ────────────────────────────────────────────────────────────────
FY = 0.085
FH = 0.105
add_rounded_rect(ax, MX, FY-FH/2, 1-2*MX, FH,
                 '#091f33', ACCENT1, 0.9, 0.88, rx=0.008, zorder=5)

# Decorative left accent block
add_rounded_rect(ax, MX, FY-FH/2, 0.006, FH,
                 ACCENT2, 'none', 0, 0.95, rx=0.004, zorder=6)

footer = (
    'Reduces drizzle-like artefacts'
    '  \u00b7  '
    'Restores wet/dry intermittency'
    '  \u00b7  '
    'Improves extreme-event reconstruction'
)
ax.text(0.5, FY, footer,
        ha='center', va='center', fontsize=7.5,
        color=LIGHT, zorder=10, transform=ax.transAxes)

# ── Bottom accent strip ────────────────────────────────────────────────────────
ax.imshow(accent_grad, aspect='auto', extent=[0, 1, 0.008, 0.022],
          cmap=accent_cmap, origin='lower', zorder=10,
          transform=ax.transAxes)

# ── Save ──────────────────────────────────────────────────────────────────────
# Output directory: figures/figures/ at the repository root (same folder as submitted manuscript figures).
# This script must be run from the repository root: python src/figures/generate_graphical_abstract.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.normpath(os.path.join(_script_dir, '..', '..', 'figures', 'figures'))
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, 'Graphical_Abstract_DLPIF.png')
pdf_path = os.path.join(out_dir, 'Graphical_Abstract_DLPIF.pdf')

fig.savefig(png_path, dpi=DPI, bbox_inches='tight',
            facecolor=BG_TOP, edgecolor='none',
            metadata={'Title': 'DLPIF Graphical Abstract'})

fig.savefig(pdf_path, bbox_inches='tight',
            facecolor=BG_TOP, edgecolor='none',
            metadata={'Title': 'DLPIF Graphical Abstract'})

plt.close(fig)
print("[OK]  PNG saved: " + png_path)
print("[OK]  PDF saved: " + pdf_path)
print("      Size: %d x %d px @ %d dpi" % (W_PX, H_PX, DPI))
