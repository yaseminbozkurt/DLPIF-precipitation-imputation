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
FILE_PREFIX = "Figure_2_WetDayFrequency"

# Seaborn settings
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
    if m in ['mean', 'linear', 'knn', 'mice']:
        return m.capitalize()
    return m

# 1. Load Data
df_main = pd.read_csv('results/clean_full_evaluation.csv')

saits_files = glob.glob('src/results_dl/evaluation_saits_precip_seed*.csv')
if saits_files:
    df_saits = pd.concat([pd.read_csv(f) for f in saits_files], ignore_index=True)
    df_all = pd.concat([df_main, df_saits], ignore_index=True)
else:
    df_all = df_main

df_all['Method'] = df_all['method'].apply(clean_method)

# Filter scenarios
scenarios = ['10pct', '20pct', 'block7d', 'block30d']
scen_titles = ['10% Missing', '20% Missing', '7-Day Block', '30-Day Block']

df_agg = df_all.groupby(['Method', 'scenario'])[['freq_pred', 'freq_gt']].mean().reset_index()

# Method order
method_order = ['Mean', 'Linear', 'Knn', 'Mice', 'WGAN-GP raw', 'SAITS', 'PrecipFix', 'Precip2Stage', 'DLPIF']
method_order = [m for m in method_order if m in df_agg['Method'].values]

fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)

# Seaborn colorblind palette
colors = sns.color_palette("colorblind", len(method_order))

for i, scen in enumerate(scenarios):
    ax = axes[i]
    df_sub = df_agg[df_agg['scenario'] == scen].set_index('Method')
    df_sub = df_sub.reindex(method_order)
    
    # Plot horizontal bars
    sns.barplot(
        data=df_sub.reset_index(),
        x="freq_pred",
        y="Method",
        ax=ax,
        order=method_order,
        palette=colors,
        edgecolor=".2"
    )
    
    # Ground truth line
    gt_val = df_sub['freq_gt'].iloc[0]
    ax.axvline(gt_val, color='black', linestyle='--', linewidth=2, label='Ground Truth' if i==0 else "")
    
    ax.set_title(scen_titles[i], fontweight='bold')
    ax.set_xlabel("Wet-day Frequency")
    if i == 0:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("")
        
    ax.set_xlim(0, 1.05)
    
    # Despine
    sns.despine(ax=ax, left=True, bottom=False)

fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=1, frameon=False)
plt.tight_layout()

# Save
for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}.{ext}")
    plt.savefig(out_path, format=ext, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close()
