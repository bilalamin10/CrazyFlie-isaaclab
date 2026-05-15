"""Post-process: compute success rate at multiple thresholds from time series .npy files.

This re-derives success rates from the saved per-step tracking errors,
without needing to rerun eval. Lets you see how the metric changes with threshold.

Usage:
    python scripts/compute_success_thresholds.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

LOGS = Path("logs/rsl_rl")
THRESHOLDS = [0.20, 0.30, 0.50, 1.00]  # meters

records = []
for p in sorted(LOGS.glob("*_seed*/*/eval-on-*_timeseries.npy")):
    # Parse train_shape, seed from the parent_parent folder name
    run_folder = p.parent.parent.name   # e.g. "Hover_seed0"
    train_shape, seed_str = run_folder.split("_seed")
    seed = int(seed_str)

    # Parse eval_shape from the file name
    eval_shape = p.name.replace("eval-on-", "").replace("_timeseries.npy", "")

    errors = np.load(p)
    if len(errors) == 0:
        print(f"  Warning: empty file {p}")
        continue

    row = {
        "train_shape": train_shape,
        "eval_shape": eval_shape,
        "seed": seed,
        "mean_err": errors.mean(),
        "p95_err": np.percentile(errors, 95),
    }
    for thr in THRESHOLDS:
        row[f"success_{int(thr*100)}cm"] = (errors < thr).mean()
    records.append(row)

df = pd.DataFrame(records)
print(f"Loaded {len(df)} cells.\n")

# Pivot tables for each threshold
shapes = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]
for thr in THRESHOLDS:
    col = f"success_{int(thr*100)}cm"
    print(f"\n{'='*70}")
    print(f"Success rate at threshold {thr*100:.0f} cm — mean across seeds")
    print(f"{'='*70}")
    pivot = df.groupby(["train_shape", "eval_shape"])[col].mean().unstack()
    pivot = pivot.reindex(index=shapes, columns=shapes)
    print(pivot.round(3).to_string())

# Save full CSV for further plotting
out = Path("logs/success_thresholds.csv")
df.to_csv(out, index=False)
print(f"\nSaved {out}")
