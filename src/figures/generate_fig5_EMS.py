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
FILE_PREFIX = "Figure_5_ExtremeEvents"

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

# Load Data
df_main = pd.read_csv('results/clean_full_evaluation.csv')
saits_files = glob.glob('src/results_dl/evaluation_saits_extreme_seed*.csv')
if saits_files:
    df_saits = pd.concat([pd.read_csv(f) for f in saits_files], ignore_index=True)
    df_all = pd.concat([df_main, df_saits], ignore_index=True)
else:
    df_all = df_main

df_all['Method'] = df_all['method'].apply(clean_method)

# Filter for only strongest methods
keep_methods = ['Linear', 'WGAN-GP raw', 'SAITS', 'Precip2Stage', 'DLPIF']
df_all = df_all[df_all['Method'].isin(keep_methods)]

scenarios = ['10pct', '20pct', 'block7d', 'block30d']
scen_labels = ['10%', '20%', '7d Block', '30d Block']
scen_map = dict(zip(scenarios, scen_labels))
df_all['Scenario'] = df_all['scenario'].map(scen_map)
df_all['Scenario'] = pd.Categorical(df_all['Scenario'], categories=scen_labels, ordered=True)
df_all = df_all.dropna(subset=['Scenario'])

# Aggregate
df_agg = df_all.groupby(['Method', 'Scenario'])[['mae_p95', 'rmse_p95']].mean().reset_index()

# Set order
df_agg['Method'] = pd.Categorical(df_agg['Method'], categories=keep_methods, ordered=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

palette = sns.color_palette("colorblind", len(keep_methods))

# Plot MAE (5a)
sns.barplot(
    data=df_agg, x='Scenario', y='mae_p95', hue='Method',
    ax=axes[0], palette=palette, edgecolor=".2"
)
axes[0].set_title("(a) Extreme Precipitation MAE\n(Events ≥ 95th percentile (16.74 mm/day))", fontweight='bold')
axes[0].set_ylabel("MAE (mm/day)")
axes[0].set_xlabel("")
axes[0].legend_.remove()

# Plot RMSE (5b)
sns.barplot(
    data=df_agg, x='Scenario', y='rmse_p95', hue='Method',
    ax=axes[1], palette=palette, edgecolor=".2"
)
axes[1].set_title("(b) Extreme Precipitation RMSE\n(Events ≥ 95th percentile (16.74 mm/day))", fontweight='bold')
axes[1].set_ylabel("RMSE (mm/day)")
axes[1].set_xlabel("")
axes[1].legend(title="Method", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=9, title_fontsize=10)

for ax in axes:
    sns.despine(ax=ax)

plt.tight_layout(w_pad=2.0)

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}.{ext}")
    plt.savefig(out_path, format=ext, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close()
