# scripts/plot_error_distributions.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LOGS = Path("logs/rsl_rl")
SHAPES = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]

# Pick one training shape, show error distribution across all eval shapes
TRAIN = "Hover"   # change to whichever you want
fig, ax = plt.subplots(figsize=(10, 6))

data = []
labels = []
for eval_shape in SHAPES:
    all_errors = []
    for seed in [0, 1, 2]:
        pattern = f"{TRAIN}_seed{seed}/*/eval-on-{eval_shape}_timeseries.npy"
        for p in LOGS.glob(pattern):
            all_errors.append(np.load(p))
    if all_errors:
        data.append(np.concatenate(all_errors))
        labels.append(eval_shape)

ax.boxplot(data, labels=labels, showfliers=False)
ax.set_ylabel("Tracking error (m)")
ax.set_title(f"Tracking error distribution — policy trained on {TRAIN}\n(boxes across all timesteps and 3 seeds)")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"logs/plots/distribution_{TRAIN}.png", dpi=150, bbox_inches="tight")
plt.show()