"""
baselines_dl/run_multiseed_saits_v2.py
========================================
Orchestrator: runs train_saits_v2.py for seeds [42, 7, 123]
then aggregates F1, CSI, bias, RMSE_wet mean ± std into:
  results_dl/saits_v2/saits_v2_multiseed_summary.csv

Nothing existing is modified.
"""

import os, sys, subprocess, io, pandas as pd, numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON      = sys.executable
TRAIN_SCRIPT = os.path.join(PROJECT_DIR, "baselines_dl", "train_saits_v2.py")
V2_RESULTS  = os.path.join(PROJECT_DIR, "results_dl", "saits_v2")
SEEDS       = [42, 7, 123]
METRICS     = ["F1", "CSI", "bias", "RMSE_wet"]


class SafeStream(io.TextIOBase):
    def __init__(self, wrapped):
        self._w = wrapped
    def write(self, s):
        safe = s.encode("cp1254", errors="replace").decode("cp1254")
        self._w.write(safe)
        return len(s)
    def flush(self):
        self._w.flush()


def run(seed):
    print(f"\n{'='*60}")
    print(f"  TRAINING  seed={seed}")
    print(f"{'='*60}")
    proc = subprocess.Popen(
        [PYTHON, TRAIN_SCRIPT, "--seed", str(seed)],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    safe = SafeStream(sys.stdout)
    for line in proc.stdout:
        safe.write(line)
    proc.wait()
    return proc.returncode


def aggregate():
    print(f"\n{'='*60}")
    print("  AGGREGATING v2 results ...")
    print(f"{'='*60}\n")

    frames = []
    for s in SEEDS:
        p = os.path.join(V2_RESULTS, f"evaluation_v2_seed{s}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p); df["seed"] = s; frames.append(df)
        else:
            print(f"  WARNING: {p} not found — skipping seed {s}")

    if not frames:
        print("  No evaluation files found. Exiting.")
        return

    all_df = pd.concat(frames, ignore_index=True)

    # Relabel method to "SAITS_v2"
    all_df["method_group"] = "SAITS_v2"

    rows = []
    for (mg, scen), grp in all_df.groupby(["method_group", "scenario"]):
        row = {"method": mg, "scenario": scen, "n_seeds": grp["seed"].nunique()}
        for col in METRICS:
            if col in grp.columns:
                vals = grp[col].dropna()
                row[f"mean_{col}"] = round(float(vals.mean()), 4) if len(vals) else np.nan
                row[f"std_{col}"]  = round(float(vals.std()),  4) if len(vals) > 1 else 0.0
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["scenario", "method"])
    out = os.path.join(V2_RESULTS, "saits_v2_multiseed_summary.csv")
    summary.to_csv(out, index=False)
    print(f"  Saved: {out}\n")

    # Pretty print
    cols = ["method", "scenario", "n_seeds"] + \
           [c for c in summary.columns if c.startswith("mean_") or c.startswith("std_")]
    print(summary[cols].to_string(index=False))

    # F1 mean pivot
    try:
        pivot = summary.pivot(index="method", columns="scenario", values="mean_F1").round(4)
        print("\n  mean_F1 by method x scenario:")
        print(pivot.to_string())
    except Exception:
        pass

    print(f"\n  ALL DONE.")
    print(f"  Summary CSV -> {out}")


def main():
    os.makedirs(V2_RESULTS, exist_ok=True)

    for seed in SEEDS:
        rc = run(seed)
        if rc != 0:
            print(f"\n  ERROR: seed {seed} failed with exit code {rc}. Stopping.")
            sys.exit(1)

    aggregate()


if __name__ == "__main__":
    main()
