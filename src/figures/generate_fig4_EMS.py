import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "figures/EMS"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_PREFIX = "Figure_4_Bias_vs_F1"

sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["font.size"] = 11

def clean_method(m):
    if m.startswith('WGAN-GP_raw'): return 'WGAN-GP raw'
    if m.startswith('PrecipFix'): return 'PrecipFix'
    if m.startswith('Precip2Stage'): return 'Precip2Stage'
    if m.startswith('AmountRF'): return 'DLPIF'
    if m.startswith('SAITS'): return 'SAITS'
    return m

# Load Data
df_main = pd.read_csv('results/clean_full_evaluation.csv')
saits_files = glob.glob('src/results_dl/evaluation_saits_precip_seed*.csv')
if saits_files:
    df_saits = pd.concat([pd.read_csv(f) for f in saits_files], ignore_index=True)
    df_all = pd.concat([df_main, df_saits], ignore_index=True)
else:
    df_all = df_main

if 'F1' in df_all.columns and 'f1' in df_all.columns:
    df_all['F1'] = df_all['F1'].fillna(df_all['f1'])
elif 'f1' in df_all.columns:
    df_all.rename(columns={'f1': 'F1'}, inplace=True)

df_all['Method'] = df_all['method'].apply(clean_method)
df_all['abs_bias'] = df_all['bias'].abs()

# ONLY keep correction methods
keep_methods = ['PrecipFix', 'Precip2Stage', 'SAITS', 'DLPIF']
df_all = df_all[df_all['Method'].isin(keep_methods)]

# Ensure scenario categorical
scenarios = ['10pct', '20pct', 'block7d', 'block30d']
scen_labels = ['10%', '20%', '7d Block', '30d Block']
scen_map = dict(zip(scenarios, scen_labels))
df_all['Scenario'] = df_all['scenario'].map(scen_map)

# Group by method and scenario to plot means
df_agg = df_all.groupby(['Method', 'Scenario'])[['abs_bias', 'F1']].mean().reset_index()

fig, ax = plt.subplots(figsize=(7, 6))

markers = {"10%": "o", "20%": "s", "7d Block": "^", "30d Block": "D"}
palette = sns.color_palette("colorblind", len(keep_methods))

sns.scatterplot(
    data=df_agg,
    x="abs_bias", y="F1",
    hue="Method", style="Method",
    palette=palette,
    s=170, alpha=0.9, ax=ax, edgecolor='w', linewidth=0.5
)

# Highlight DLPIF with black edgecolor
for collection in ax.collections:
    # ax.collections has paths. Since it's mapped by hue/style, we can find DLPIF
    # An easier way is just to redraw DLPIF on top
    pass

df_dlpif = df_agg[df_agg['Method'] == 'DLPIF']
sns.scatterplot(
    data=df_dlpif,
    x="abs_bias", y="F1",
    color=palette[keep_methods.index('DLPIF')],
    marker='X', # The style mapped to DLPIF usually, or we can just let it inherit if we use edgecolor
    s=170, alpha=1.0, ax=ax, edgecolor='black', linewidth=1.2, legend=False
)

# Add annotations with specific textcoords offsets
offsets_dlpif = {
    "10%": {"xy": (-10, -4), "ha": "right", "va": "center"},
    "20%": {"xy": (0, 14), "ha": "center", "va": "bottom"},
    "7d": {"xy": (14, 0), "ha": "left", "va": "center"},
    "30d": {"xy": (0, -20), "ha": "center", "va": "top"}
}

for idx, row in df_agg.iterrows():
    if row['Method'] == 'DLPIF':
        scen_clean = row['Scenario'].replace(' Block', '')
        conf = offsets_dlpif.get(scen_clean, {"xy": (5, 5), "ha": "left", "va": "center"})
        
        ax.annotate(
            scen_clean, 
            (row['abs_bias'], row['F1']),
            xytext=conf["xy"],
            textcoords='offset points',
            fontsize=12, alpha=0.9, fontweight='semibold',
            ha=conf["ha"],
            va=conf["va"]
        )

ax.set_xlabel("Absolute Wet-Day Frequency Bias")
ax.set_ylabel("F1 Score")
ax.set_ylim(0.3, 1.0)
ax.set_xlim(left=-0.01)

# Keep only Method legend, remove duplicates
handles, labels = ax.get_legend_handles_labels()
unique_h, unique_l = [], []
seen = set()
for h, l in zip(handles, labels):
    if l in keep_methods and l not in seen:
        unique_h.append(h)
        unique_l.append(l)
        seen.add(l)

ax.legend(handles=unique_h, labels=unique_l, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, title="Method")
sns.despine(ax=ax)

plt.tight_layout()

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}.{ext}")
    plt.savefig(out_path, format=ext, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close()
