# scripts/plot_bar_per_train.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("logs/eval_matrix.csv")
SHAPES = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]

# 5-panel figure: one panel per training shape
fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
x = np.arange(len(SHAPES))

for ax, train in zip(axes, SHAPES):
    sub = df[df.train_shape == train]
    means = [sub[sub.eval_shape == s].tracking_err_mean.mean() for s in SHAPES]
    stds  = [sub[sub.eval_shape == s].tracking_err_mean.std()  for s in SHAPES]
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", edgecolor="black")
    # highlight the diagonal (eval == train) cell
    bars[SHAPES.index(train)].set_color("orange")
    ax.set_xticks(x)
    ax.set_xticklabels(SHAPES, rotation=30, ha="right")
    ax.set_title(f"Trained on {train}")
    ax.grid(axis="y", alpha=0.3)
    if ax is axes[0]:
        ax.set_ylabel("Mean tracking error (m)")

fig.suptitle("Cross-trajectory generalization (mean ± std over 3 seeds)", fontsize=14)
plt.tight_layout()
plt.savefig("logs/plots/bar_per_train.png", dpi=150, bbox_inches="tight")
plt.show()