import pandas as pd
import numpy as np

clean = pd.read_csv('results/clean_full_evaluation.csv')
multi = pd.read_csv('results/multiseed_clean_evaluation.csv')

def fam(m):
    m = str(m)
    if m.lower() == 'mean': return 'Mean'
    if m.lower() == 'linear': return 'Linear'
    if m.lower() == 'knn': return 'KNN'
    if m.lower() == 'mice': return 'MICE'
    if m.startswith('WGAN-GP_raw'): return 'WGAN-GP'
    if m.startswith('PrecipFix'): return 'PrecipFix'
    if m.startswith('Precip2Stage_clean'): return 'Precip2Stage'
    if m.startswith('AmountRF_clean'): return 'DLPIF'
    return m

clean['family'] = clean['method'].map(fam)
multi['family'] = multi['method'].map(fam)
SCENARIOS = ['10pct','20pct','block7d','block30d']

print("=== BIAS DEFINITION CHECK (should be freq_pred - freq_gt) ===")
for _, r in clean[clean['family']=='Mean'].iterrows():
    computed = round(r['freq_pred'] - r['freq_gt'], 4)
    stored = round(r['bias'], 4)
    match = 'OK' if abs(computed - stored) < 0.001 else 'MISMATCH'
    print(f"  {r['scenario']}: stored={stored}, computed={computed}  [{match}]")

print()
print("=== FIGURE 3: BIAS VALUES PER METHOD/SCENARIO ===")
for sc in SCENARIOS:
    print(f"\n--- {sc} ---")
    for f in ['Mean','Linear','KNN','MICE','WGAN-GP','PrecipFix','Precip2Stage','DLPIF']:
        src = multi if f in ['DLPIF','Precip2Stage'] else clean
        rows = src[(src['family']==f) & (src['scenario']==sc)]
        if not rows.empty:
            b = rows['bias'].mean()
            fp = rows['freq_pred'].mean()
            fg = rows['freq_gt'].mean()
            print(f"  {f:<15}: bias={b:+.4f}  freq_pred={fp:.4f}  freq_gt={fg:.4f}")

print()
print("=== FIGURE 4: F1 VALUES ===")
for sc in SCENARIOS:
    print(f"\n--- {sc} ---")
    for f in ['Linear','KNN','MICE','WGAN-GP','PrecipFix','Precip2Stage','DLPIF']:
        src = multi if f in ['DLPIF','Precip2Stage'] else clean
        rows = src[(src['family']==f) & (src['scenario']==sc)]
        if not rows.empty:
            print(f"  {f:<15}: F1={rows['f1'].mean():.4f} +/- {rows['f1'].std():.4f}")

print()
print("=== FIGURE 5: ABLATION RMSE/MAE ===")
for sc in SCENARIOS:
    print(f"\n--- {sc} ---")
    for f in ['PrecipFix','Precip2Stage','DLPIF']:
        src = multi if f in ['DLPIF','Precip2Stage'] else clean
        rows = src[(src['family']==f) & (src['scenario']==sc)]
        if not rows.empty:
            print(f"  {f:<15}: rmse_wet={rows['rmse_wet'].mean():.4f}, mae_wet={rows['mae_wet'].mean():.4f}")

print()
print("=== FIGURE 6: P95 MAE/RMSE ===")
for sc in SCENARIOS:
    print(f"\n--- {sc} ---")
    for f in ['Linear','KNN','MICE','WGAN-GP','PrecipFix','Precip2Stage','DLPIF']:
        src = multi if f in ['DLPIF','Precip2Stage'] else clean
        rows = src[(src['family']==f) & (src['scenario']==sc)]
        if not rows.empty:
            print(f"  {f:<15}: mae_p95={rows['mae_p95'].mean():.4f}, rmse_p95={rows['rmse_p95'].mean():.4f}")

print()
print("=== FIGURE 2: MASK PERCENTAGES ===")
import numpy as np
rng = np.random.default_rng(42)
n = 140
m10 = np.zeros(n); m10[rng.choice(n, size=round(n*0.10), replace=False)] = 1
rng2 = np.random.default_rng(42)
m10b = np.zeros(n); m10b[rng2.choice(n, size=round(n*0.10), replace=False)] = 1
rng3 = np.random.default_rng(42)
m20 = np.zeros(n); m20[rng3.choice(n, size=round(n*0.10), replace=False)] = 1
rng4 = np.random.default_rng(42)
m20b = np.zeros(n); m20b[rng4.choice(n, size=round(n*0.20), replace=False)] = 1
b7 = np.zeros(n)
for s in [8,32,61,95,121]: b7[s:s+7] = 1
b30 = np.zeros(n)
for s in [18,88]: b30[s:s+30] = 1
print(f"  10pct mask: {m10b.sum()} days = {m10b.mean()*100:.1f}%")
print(f"  20pct mask: {m20b.sum()} days = {m20b.mean()*100:.1f}%")
print(f"  block7d mask: {b7.sum()} days = {b7.mean()*100:.1f}%")
print(f"  block30d mask: {b30.sum()} days = {b30.mean()*100:.1f}%")
print()
print("  From data n_masked values:")
for sc in SCENARIOS:
    r = clean[clean['scenario']==sc]['n_masked'].iloc[0]
    print(f"  {sc}: n_masked={r}")

print()
print("=== DLPIF vs Precip2Stage F1 IDENTICAL CHECK ===")
for sc in SCENARIOS:
    d = multi[(multi['family']=='DLPIF') & (multi['scenario']==sc)]['f1'].mean()
    p = multi[(multi['family']=='Precip2Stage') & (multi['scenario']==sc)]['f1'].mean()
    print(f"  {sc}: DLPIF F1={d:.4f}, Precip2Stage F1={p:.4f}, diff={d-p:.6f}")

print()
print("=== CHECK Fig3: Are bias values ≈ 0.68 or freq_pred ≈ 1.0? ===")
r = clean[(clean['family']=='Mean') & (clean['scenario']=='10pct')]
print(f"  Mean/10pct: freq_gt={r['freq_gt'].values[0]}, freq_pred={r['freq_pred'].values[0]}, bias={r['bias'].values[0]}")
print(f"  So bias = {r['freq_pred'].values[0]} - {r['freq_gt'].values[0]} = {r['freq_pred'].values[0]-r['freq_gt'].values[0]:.3f}")
print(f"  freq_gt (observed) = {r['freq_gt'].values[0]:.3f} = 32.2% wet days in test set")
