import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration
OUTPUT_DIR = "figures/EMS"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_PREFIX = "Figure_1_DLPIF_Workflow"

# Fonts & Style
FONT = "Arial"
plt.rcParams["font.family"] = FONT

# Color Progression: Blue -> Teal -> Green
C_INPUT = "#34495e"    # Dark blue/grey
C_BASE  = "#2980b9"    # Blue
C_STG1  = "#008080"    # Teal
C_STG2  = "#27ae60"    # Green
C_CAL   = "#2ecc71"    # Lighter green
C_OUT   = "#2c3e50"    # Dark neutral

C_TEXT  = "white"
C_ARROW = "#555555"

# Canvas setup
FW, FH = 11.5, 3.5
fig, ax = plt.subplots(figsize=(FW, FH), dpi=300)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")

# Title and Subtitle
ax.text(FW/2, FH - 0.3, "Decoupled Latent-Physical Imputation Framework (DLPIF)", 
        ha="center", va="center", fontsize=15, fontweight="bold", color="#222222")
ax.text(FW/2, FH - 0.7, "Physically informed decoupling of precipitation occurrence and amount reconstruction", 
        ha="center", va="center", fontsize=12, style="italic", color="#444444")

# Box setup
boxes = [
    {"x": 0.2, "w": 2.0, "color": C_INPUT, "t_main": "Multivariate meteorological\nobservations\nwith missing values", "t_sub": ""},
    {"x": 2.5, "w": 1.5, "color": C_BASE,  "t_main": "Base continuous\nimputation", "t_sub": ""},
    {"x": 4.3, "w": 1.5, "color": C_STG1,  "t_main": "Stage 1", "t_sub": "wet/dry classification"},
    {"x": 6.1, "w": 1.6, "color": C_STG2,  "t_main": "Stage 2", "t_sub": "conditional wet-day\namount estimation"},
    {"x": 8.0, "w": 1.5, "color": C_CAL,   "t_main": "Physical\nConsistency\nEnforcement", "t_sub": ""},
    {"x": 9.8, "w": 1.5, "color": C_OUT,  "t_main": "Final precipitation\nreconstruction", "t_sub": ""}
]

BOX_H = 1.0
BOX_Y = 1.0
CORNER = 0.15

def draw_box(x, y, w, h, c, t_main, t_sub):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={CORNER}",
        facecolor=c, edgecolor="none", zorder=3
    )
    ax.add_patch(box)
    
    # Text positioning
    cx = x + w/2
    if t_sub:
        cy_m = y + h/2 + 0.15
        cy_s = y + h/2 - 0.15
        ax.text(cx, cy_m, t_main, ha="center", va="center", color=C_TEXT, fontsize=11, fontweight="bold")
        ax.text(cx, cy_s, t_sub, ha="center", va="center", color=C_TEXT, fontsize=10)
    else:
        cy_m = y + h/2
        ax.text(cx, cy_m, t_main, ha="center", va="center", color=C_TEXT, fontsize=10, fontweight="bold")

# Draw arrows
def draw_arrow(x1, x2, y):
    arr = mpatches.FancyArrowPatch(
        (x1, y), (x2, y),
        arrowstyle="-|>,head_length=6,head_width=4",
        color=C_ARROW, linewidth=2.5, zorder=2,
        mutation_scale=2
    )
    ax.add_patch(arr)

# Draw all elements
for i, b in enumerate(boxes):
    draw_box(b["x"], BOX_Y, b["w"], BOX_H, b["color"], b["t_main"], b["t_sub"])
    
    # Draw arrow to next box
    if i < len(boxes) - 1:
        next_b = boxes[i+1]
        x_end_box = b["x"] + b["w"]
        x_start_next = next_b["x"]
        
        # gap = 0.4
        draw_arrow(x_end_box + 0.03, x_start_next - 0.05, BOX_Y + BOX_H/2)

plt.tight_layout()

# Save
for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}.{ext}")
    plt.savefig(out_path, format=ext, dpi=300, bbox_inches="tight", transparent=False)
    print(f"Saved: {out_path}")

plt.close()
