import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("logs/lookahead_sweep.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Tracking error vs lookahead
ax = axes[0]
for shape in df.eval_shape.unique():
    sub = df[df.eval_shape == shape]
    grouped = sub.groupby("lookahead")["tracking_err_mean"].agg(["mean", "std"])
    ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                marker="o", capsize=4, linewidth=2, label=f"Eval on {shape}")
ax.set_xlabel("Lookahead (seconds)", fontsize=12)
ax.set_ylabel("Mean tracking error (m)", fontsize=12)
ax.set_title("Effect of setpoint lookahead on tracking error\n(Static-trained policy, mean ± std over 3 seeds)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Success rate vs lookahead
ax = axes[1]
for shape in df.eval_shape.unique():
    sub = df[df.eval_shape == shape]
    grouped = sub.groupby("lookahead")["success_rate"].agg(["mean", "std"])
    ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                marker="o", capsize=4, linewidth=2, label=f"Eval on {shape}")
ax.set_xlabel("Lookahead (seconds)", fontsize=12)
ax.set_ylabel("Success rate (within 30 cm)", fontsize=12)
ax.set_title("Effect of setpoint lookahead on success rate")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("logs/plots/lookahead_sweep.png", dpi=150, bbox_inches="tight")
print("Saved logs/plots/lookahead_sweep.png")
plt.show()