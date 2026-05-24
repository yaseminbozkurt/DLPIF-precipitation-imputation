import pandas as pd
import glob
dfs_p = [pd.read_csv(f) for f in glob.glob('src/results_dl/evaluation_saits_precip_seed*.csv')]
dfs_e = [pd.read_csv(f) for f in glob.glob('src/results_dl/evaluation_saits_extreme_seed*.csv')]
df_p = pd.concat(dfs_p).groupby('scenario').mean(numeric_only=True).reset_index()
df_e = pd.concat(dfs_e).groupby('scenario').mean(numeric_only=True).reset_index()

s10 = df_p[df_p['scenario']=='10pct'].iloc[0]
e10 = df_e[df_e['scenario']=='10pct'].iloc[0]

s30 = df_p[df_p['scenario']=='block30d'].iloc[0]
e30 = df_e[df_e['scenario']=='block30d'].iloc[0]

row = f"| SAITS | {s10['bias']:+7.4f} | {s10['F1']:.4f} | {e10['rmse_p95']:.2f} | {s30['bias']:+7.4f} | {s30['F1']:.4f} | {e30['rmse_p95']:.2f} |"
print(row)
