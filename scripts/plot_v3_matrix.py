"""
Final 4×4 matrix plot + side-by-side v2 vs v3 comparison.
Train: Hover3D, Circle_v3, Lemniscate_v3, Lissajous_v3
Eval:  Hover, Circle, Lemniscate, Lissajous  (no Static)
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT    = Path("logs/plots")
LOGS   = Path("logs/rsl_rl")
OUT.mkdir(parents=True, exist_ok=True)

EVAL_SHAPES  = ["Hover", "Circle", "Lemniscate", "Lissajous"]
TRAIN_ROWS_V3 = ["Hover3D", "Circle_v3", "Lemniscate_v3", "Lissajous_v3"]
TRAIN_ROWS_V2 = ["Circle",  "Lemniscate", "Lissajous"]
SEEDS        = [0, 1, 2]
DT           = 1.0 / 60.0
COLORS       = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ── helpers ──────────────────────────────────────────────────────────────────
def load_v3_pivot():
    df = pd.read_csv("logs/eval_matrix_v3.csv")
    pivot = (df.groupby(["train", "eval_shape"])["tracking_err_mean"]
               .agg(["mean", "std"]).unstack())
    mean = pivot["mean"].reindex(index=TRAIN_ROWS_V3, columns=EVAL_SHAPES)
    std  = pivot["std"].reindex(index=TRAIN_ROWS_V3, columns=EVAL_SHAPES)
    return mean, std


def load_v2_pivot():
    df = pd.read_csv("logs/eval_matrix_v2.csv")
    pivot = (df.groupby(["train_shape", "eval_shape"])["tracking_err_mean"]
               .agg(["mean", "std"]).unstack())
    mean = pivot["mean"].reindex(index=TRAIN_ROWS_V2, columns=EVAL_SHAPES)
    std  = pivot["std"].reindex(index=TRAIN_ROWS_V2, columns=EVAL_SHAPES)
    return mean, std


def load_timeseries(exp_name: str, eval_shape: str, seed: int):
    """Load timeseries from seed folder (direct) or timestamped subfolder."""
    seed_dir = LOGS / f"{exp_name}_seed{seed}"
    # Direct in seed folder
    p = seed_dir / f"eval-on-{eval_shape}_timeseries.npy"
    if p.exists():
        return np.load(p)
    # Timestamped subfolder
    matches = sorted(seed_dir.glob(f"*/eval-on-{eval_shape}_timeseries.npy"))
    if matches:
        return np.load(matches[-1])
    return None


def get_mean_std_ts(exp_name: str, eval_shape: str):
    series = [s for seed in SEEDS
              for s in [load_timeseries(exp_name, eval_shape, seed)]
              if s is not None and len(s) > 0]
    if not series:
        return None, None, None
    min_len = min(len(s) for s in series)
    stacked = np.stack([s[:min_len] for s in series])
    t = np.arange(min_len) * DT
    return t, stacked.mean(0), stacked.std(0)


# ── Plot 1: Final 4×4 heatmap (v3, mean ± std) ───────────────────────────────
def plot_heatmap_v3():
    mean, std = load_v3_pivot()

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(mean.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=0.0, vmax=2.5)

    ax.set_xticks(range(4))
    ax.set_xticklabels(EVAL_SHAPES, rotation=20, ha="right", fontsize=12)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["Hover3D\n(position ctrl)", "Circle", "Lemniscate", "Lissajous"],
                       fontsize=11)
    ax.set_xlabel("Eval trajectory", fontsize=12)
    ax.set_ylabel("Train trajectory", fontsize=12)
    ax.set_title(
        "Mean ± std tracking error (m) — final 4×4 matrix\n"
        "(3 seeds · lower = better · blue = diagonal)",
        fontsize=13,
    )

    for i in range(4):
        for j in range(4):
            m_val = mean.values[i, j]
            s_val = std.values[i, j]
            if not np.isnan(m_val):
                txt_color = "white" if m_val > 1.8 else "black"
                s_str = f"±{s_val:.2f}" if not np.isnan(s_val) and s_val > 0 else "1 seed"
                ax.text(j, i, f"{m_val:.2f}\n{s_str}",
                        ha="center", va="center", fontsize=9,
                        color=txt_color, fontweight="bold", linespacing=1.4)

    # Diagonal highlight (skip Hover3D row 0 — no diagonal concept)
    for k in range(1, 4):   # Circle=1, Lemniscate=2, Lissajous=3
        # eval index matches train index offset by 1 (Hover is col 0)
        eval_col = k   # Circle→1, Lemniscate→2, Lissajous→3
        rect = plt.Rectangle((eval_col - 0.5, k - 0.5), 1, 1,
                              linewidth=2.5, edgecolor="blue", facecolor="none")
        ax.add_patch(rect)

    # Separator below Hover3D row
    ax.axhline(0.5, color="white", linewidth=2, linestyle="--")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Mean tracking error (m)")
    plt.tight_layout()
    p = OUT / "heatmap_final_4x4.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {p}")


# ── Plot 2: V2 vs V3 side-by-side bar chart ───────────────────────────────────
def plot_v2_v3_comparison():
    mean_v3, std_v3 = load_v3_pivot()
    mean_v2, std_v2 = load_v2_pivot()

    # Load v2 Hover separately
    df_v2 = pd.read_csv("logs/eval_matrix_v2.csv")
    hover_v2_mean = (df_v2[df_v2.train_shape == "Hover"]
                     .groupby("eval_shape")["tracking_err_mean"]
                     .mean().reindex(EVAL_SHAPES))
    hover_v2_std  = (df_v2[df_v2.train_shape == "Hover"]
                     .groupby("eval_shape")["tracking_err_mean"]
                     .std().reindex(EVAL_SHAPES).fillna(0))

    # 4 panels: Hover, Circle, Lemniscate, Lissajous
    compare = [
        ("Hover",      "Hover",      "Hover3D"),
        ("Circle",     "Circle",     "Circle_v3"),
        ("Lemniscate", "Lemniscate", "Lemniscate_v3"),
        ("Lissajous",  "Lissajous",  "Lissajous_v3"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    for ax, (label, v2_key, v3_key) in zip(axes, compare):
        if v2_key == "Hover":
            v2_vals = hover_v2_mean.values
            v2_stds = hover_v2_std.values
        else:
            v2_vals = mean_v2.loc[v2_key, EVAL_SHAPES].values
            v2_stds = std_v2.loc[v2_key, EVAL_SHAPES].fillna(0).values

        v3_vals = mean_v3.loc[v3_key, EVAL_SHAPES].values
        v3_stds = std_v3.loc[v3_key, EVAL_SHAPES].fillna(0).values

        x = np.arange(4)
        w = 0.35
        v2_label = "Hover 2D" if label == "Hover" else "v2"
        v3_label = "Hover 3D" if label == "Hover" else "v3"

        ax.bar(x - w/2, v2_vals, w, yerr=v2_stds, capsize=4,
               label=v2_label, color="#4C72B0", alpha=0.85, edgecolor="black")
        ax.bar(x + w/2, v3_vals, w, yerr=v3_stds, capsize=4,
               label=v3_label, color="#DD8452", alpha=0.85, edgecolor="black")

        ax.set_title(f"Train: {label}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(EVAL_SHAPES, rotation=20, ha="right", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 2.6)
        if ax is axes[0]:
            ax.set_ylabel("Mean tracking error (m)", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("V2 vs V3 training: tracking error comparison (all 4 policies)\n"
                 "(lower = better · error bars = std over 3 seeds)",
                 fontsize=13)
    plt.tight_layout()
    p = OUT / "v2_vs_v3_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {p}")


# ── Plot 3: Timeseries per training shape (v3) ────────────────────────────────
def plot_timeseries_v3():
    exp_map = {
        "Hover3D":       "Hover3D",
        "Circle_v3":     "Circle_v3",
        "Lemniscate_v3": "Lemniscate_v3",
        "Lissajous_v3":  "Lissajous_v3",
    }

    for label, exp_name in exp_map.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        any_data = False

        for eval_shape, color in zip(EVAL_SHAPES, COLORS):
            t, mean, std = get_mean_std_ts(exp_name, eval_shape)
            if t is None:
                print(f"  [!] Missing: {exp_name} → {eval_shape}")
                continue
            any_data = True
            ax.plot(t, mean, color=color, linewidth=2,
                    label=f"Eval: {eval_shape}")
            if std is not None and std.max() > 0:
                ax.fill_between(t, mean - std, mean + std,
                                color=color, alpha=0.15)

        if not any_data:
            plt.close()
            continue

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Mean tracking error (m)", fontsize=12)
        ax.set_title(
            f"Policy trained on {label} — tracking error over time\n"
            "(mean ± std over 3 seeds)",
            fontsize=13,
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        p = OUT / f"timeseries_v3_{label}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[✓] {p}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Final 4×4 heatmap ───────────────────────────")
    plot_heatmap_v3()

    print("\n── V2 vs V3 comparison ─────────────────────────")
    plot_v2_v3_comparison()

    print("\n── Timeseries plots ────────────────────────────")
    plot_timeseries_v3()

    print(f"\nDone. Plots in: {OUT}")