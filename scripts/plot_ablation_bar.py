# scripts/plot_ablation_bar.py
import matplotlib.pyplot as plt

conditions = ["Baseline\n(LA=0)", "LA=1.0", "LA=2.0\n(extrapolated)", "Action history"]
errors = [1.384, 1.355, 1.361, 1.236]  # fill in your real numbers
colors = ["gray", "lightblue", "lightblue", "orange"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(conditions, errors, color=colors, edgecolor="black")
ax.set_ylabel("Mean tracking error (m)")
ax.set_title("Static-trained policy on Circle: ablation of design choices\n(Lower is better)")
ax.axhline(y=1.384, color="red", linestyle="--", alpha=0.5, label="Baseline")
ax.set_ylim(0, 1.6)
ax.grid(True, alpha=0.3, axis="y")
ax.legend()
for bar, err in zip(bars, errors):
    ax.text(bar.get_x() + bar.get_width()/2, err + 0.02, f"{err:.3f}",
            ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("logs/plots/ablation_bar.png", dpi=150, bbox_inches="tight")
plt.show()