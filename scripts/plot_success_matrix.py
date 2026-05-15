import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("logs/eval_matrix.csv")
shapes = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]

# Success rate heatmap
mean = df.groupby(["train_shape", "eval_shape"])["success_rate"].mean().unstack()
std = df.groupby(["train_shape", "eval_shape"])["success_rate"].std().unstack()
mean = mean.reindex(index=shapes, columns=shapes)
std = std.reindex(index=shapes, columns=shapes)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(mean.values, cmap="RdYlGn", vmin=0, vmax=mean.values.max())

# Annotate each cell with mean ± std
for i in range(len(shapes)):
    for j in range(len(shapes)):
        m, s = mean.values[i, j], std.values[i, j]
        text = f"{m:.2f}\n±{s:.2f}"
        # White text on dark cells, black on light
        color = "white" if m > mean.values.max() * 0.6 else "black"
        ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

ax.set_xticks(range(len(shapes)))
ax.set_yticks(range(len(shapes)))
ax.set_xticklabels(shapes)
ax.set_yticklabels(shapes)
ax.set_xlabel("Evaluation trajectory")
ax.set_ylabel("Training trajectory")
ax.set_title("Cross-trajectory generalization: mean success_rate (m)\nlower is better, mean ± std over 3 seeds")
plt.colorbar(im, ax=ax, label="Mean success_rate (m)")
plt.tight_layout()
plt.savefig("logs/success_rate_matrix.png", dpi=150, bbox_inches="tight")
print("Saved logs/success_rate_matrix.png")
plt.show()