import pandas as pd
df = pd.read_csv('results/clean_full_evaluation.csv')

def get_clean_method(m):
    if 'AmountRF_clean' in m: return 'DLPIF'
    if 'Precip2Stage' in m: return 'Precip2Stage'
    if 'PrecipFix' in m: return 'PrecipFix'
    if 'WGAN-GP_raw' in m: return 'WGAN-GP raw'
    if 'SAITS' in m: return 'SAITS'
    return m.capitalize() if m in ['mean', 'linear', 'knn', 'mice'] else m

df['Method_Group'] = df['method'].apply(get_clean_method)
agg = df.groupby(['Method_Group', 'scenario']).mean(numeric_only=True).reset_index()

# TABLE 1
methods_t1 = ['Mean', 'Linear', 'Knn', 'Mice', 'WGAN-GP raw', 'SAITS', 'PrecipFix', 'Precip2Stage', 'DLPIF']
print('=== TABLE 1 ===')
print('| Method | Bias (10%) | F1 (10%) | Extreme RMSE (10%) | Bias (30d) | F1 (30d) | Extreme RMSE (30d) |')
print('|---|---|---|---|---|---|---|')

for m in methods_t1:
    sub = agg[agg['Method_Group'] == m]
    if len(sub) == 0: continue
    row = f'| {m} |'
    
    sub_10 = sub[sub['scenario'] == '10pct']
    if len(sub_10) > 0:
        row += f" {sub_10['bias'].values[0]:+7.4f} | {sub_10['f1'].values[0]:.4f} | {sub_10['rmse_p95'].values[0]:.2f} |"
    else:
        row += ' nan | nan | nan |'
        
    sub_30 = sub[sub['scenario'] == 'block30d']
    if len(sub_30) > 0:
        row += f" {sub_30['bias'].values[0]:+7.4f} | {sub_30['f1'].values[0]:.4f} | {sub_30['rmse_p95'].values[0]:.2f} |"
    else:
        row += ' nan | nan | nan |'
        
    print(row)

# TABLE 2
print('\n=== TABLE 2 ===')
methods_t2 = ['WGAN-GP raw', 'PrecipFix', 'Precip2Stage', 'DLPIF']
strats = {
    'WGAN-GP raw': ('Raw latent', 'Raw latent'),
    'PrecipFix': ('Calibration', 'Calibrated latent'),
    'Precip2Stage': ('Random Forest', 'Raw latent'),
    'DLPIF': ('Random Forest', 'Random Forest')
}

print('| Variant | Occurrence strategy | Amount strategy | Bias (10%) | F1 (10%) | Wet-day RMSE (10%) | Extreme RMSE (10%) |')
print('|---|---|---|---|---|---|---|')
for m in methods_t2:
    sub = agg[(agg['Method_Group'] == m) & (agg['scenario'] == '10pct')]
    if len(sub) == 0: continue
    
    occ, amt = strats.get(m, ('-', '-'))
    b = sub['bias'].values[0]
    f1 = sub['f1'].values[0]
    rw = sub['rmse_wet'].values[0]
    rx = sub['rmse_p95'].values[0]
    
    print(f'| {m} | {occ} | {amt} | {b:+7.4f} | {f1:.4f} | {rw:.2f} | {rx:.2f} |')
