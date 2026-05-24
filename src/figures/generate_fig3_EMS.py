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
FILE_PREFIX = "Figure_3_F1_Performance"

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

# Standardize F1 column name
if 'F1' in df_all.columns and 'f1' in df_all.columns:
    df_all['F1'] = df_all['F1'].fillna(df_all['f1'])
elif 'f1' in df_all.columns:
    df_all.rename(columns={'f1': 'F1'}, inplace=True)

df_all['Method'] = df_all['method'].apply(clean_method)

# Method order
method_order = ['Linear', 'Precip2Stage', 'SAITS', 'DLPIF']
method_order = [m for m in method_order if m in df_all['Method'].values]

scenarios = ['10pct', '20pct', 'block7d', 'block30d']
scen_labels = ['10%', '20%', '7d Block', '30d Block']
df_all['scenario'] = pd.Categorical(df_all['scenario'], categories=scenarios, ordered=True)
df_all = df_all.dropna(subset=['scenario'])
df_all = df_all[df_all['Method'].isin(method_order)]

fig, ax = plt.subplots(figsize=(7, 5))

markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<"]
colors = sns.color_palette("colorblind", len(method_order))

for i, m in enumerate(method_order):
    sub = df_all[df_all['Method'] == m]
    if sub['F1'].isna().all(): continue
    
    # Custom styling for perfect overlap illusion
    lw = 2.2
    ms = 9
    ls = '-'
    mfc = colors[i]
    
    if m == 'Precip2Stage':
        lw = 5.0
        ms = 12
    elif m == 'DLPIF':
        lw = 2.0
        ms = 7
        ls = '--'
        mfc = 'white' # Hollow marker for DLPIF so Precip2Stage shows through
        
    sns.lineplot(
        data=sub, x='scenario', y='F1', 
        label=m, ax=ax,
        marker=markers[i%len(markers)], markersize=ms,
        linewidth=lw, linestyle=ls, color=colors[i],
        markerfacecolor=mfc,
        errorbar='sd', err_style='bars', err_kws={'capsize': 5, 'elinewidth': 1.5}
    )

ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(scen_labels)
ax.set_ylabel("F1 Score")
ax.set_xlabel("Missingness Scenario")
ax.set_ylim(0.3, 1.0)
ax.legend(title="Method", loc='lower right', frameon=True, fontsize=10, title_fontsize=11)
sns.despine(ax=ax)

plt.tight_layout()

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}.{ext}")
    plt.savefig(out_path, format=ext, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close()
