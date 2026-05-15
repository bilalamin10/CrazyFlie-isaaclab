# scripts/plot_scatter_two_metrics.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/eval_matrix.csv")
SHAPES = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]
colors = {s: c for s, c in zip(SHAPES, plt.cm.tab10.colors)}

fig, ax = plt.subplots(figsize=(10, 7))
for train in SHAPES:
    sub = df[df.train_shape == train]
    ax.scatter(sub.tracking_err_mean, sub.success_rate,
               c=[colors[train]], label=f"Train: {train}",
               s=80, alpha=0.7, edgecolor="black", linewidth=0.5)

ax.set_xlabel("Mean tracking error (m)")
ax.set_ylabel("Success rate (fraction within 30 cm)")
ax.set_title("Tracking error vs success rate — all 75 cells")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("logs/plots/scatter_two_metrics.png", dpi=150, bbox_inches="tight")
plt.show()