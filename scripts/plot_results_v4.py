"""
V4 results: annealB as the standard training recipe.

Generates:
  1. logs/eval_matrix_v4.csv      — collected from per-seed eval JSONs
  2. heatmap_v4_4x4.png           — final 4x4 matrix, mean ± std over 3 seeds
  3. v3_vs_v4_comparison.png      — before/after bar chart per training shape
  4. timeseries_v4_{Cond}.png     — 4 timeseries plots, mean ± std over 3 seeds

Train conditions (rows):
  Hover3D_annealB   — 3D-cuboid position controller
  Circle_annealB    — circle expert
  Lemniscate_annealB
  Lissajous_annealB
Eval shapes (cols): Hover, Circle, Lemniscate, Lissajous

Usage:
    python scripts/plot_results_v4.py
"""

import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
LOGS        = Path("logs/rsl_rl")
OUT         = Path("logs/plots")
CSV_V4      = Path("logs/eval_matrix_v4.csv")
CSV_V3      = Path("logs/eval_matrix_v3.csv")
SEEDS       = [0, 1, 2]
DT          = 1.0 / 60.0
EVAL_SHAPES = ["Hover", "Circle", "Lemniscate", "Lissajous"]
SHAPE_LOWER = {s: s.lower() for s in EVAL_SHAPES}
COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

TRAIN_CONDITIONS = [
    "Hover3D_annealB",
    "Circle_annealB",
    "Lemniscate_annealB",
    "Lissajous_annealB",
]
ROW_LABELS = [
    "Hover3D\n(position ctrl)",
    "Circle",
    "Lemniscate",
    "Lissajous",
]

OUT.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def find_json(cond: str, seed: int, eval_shape: str) -> Path | None:
    """Eval JSON: directly in seed dir, or in a timestamped subfolder."""
    seed_dir = LOGS / f"{cond}_seed{seed}"
    lower = SHAPE_LOWER[eval_shape]
    candidates = [
        seed_dir / f"eval-on-{lower}.json",
        seed_dir / f"eval-on-{eval_shape}.json",
        *sorted(seed_dir.glob(f"*/eval-on-{lower}.json")),
        *sorted(seed_dir.glob(f"*/eval-on-{eval_shape}.json")),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_timeseries(cond: str, seed: int, eval_shape: str) -> Path | None:
    seed_dir = LOGS / f"{cond}_seed{seed}"
    lower = SHAPE_LOWER[eval_shape]
    candidates = [
        seed_dir / f"eval-on-{lower}_timeseries.npy",
        seed_dir / f"eval-on-{eval_shape}_timeseries.npy",
        *sorted(seed_dir.glob(f"*/eval-on-{lower}_timeseries.npy")),
        *sorted(seed_dir.glob(f"*/eval-on-{eval_shape}_timeseries.npy")),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def get_ts_mean_std(cond: str, eval_shape: str):
    series = []
    for seed in SEEDS:
        p = find_timeseries(cond, seed, eval_shape)
        if p is not None:
            s = np.load(p)
            if len(s) > 0:
                series.append(s)
    if not series:
        return None, None, None
    min_len = min(len(s) for s in series)
    stacked = np.stack([s[:min_len] for s in series])
    t = np.arange(min_len) * DT
    std = stacked.std(0) if stacked.shape[0] > 1 else np.zeros(min_len)
    return t, stacked.mean(0), std


# ── Step 1: Build eval_matrix_v4.csv ─────────────────────────────────────────
def build_csv():
    rows = []
    for cond in TRAIN_CONDITIONS:
        for seed in SEEDS:
            for eval_shape in EVAL_SHAPES:
                p = find_json(cond, seed, eval_shape)
                if p is None:
                    print(f"  [!] Missing: {cond} seed{seed} → {eval_shape}")
                    continue
                d = json.loads(p.read_text())
                rows.append({
                    "train":             cond,
                    "eval_shape":        eval_shape,
                    "seed":              seed,
                    "tracking_err_mean": d.get("Metrics/tracking_err_mean"),
                    "tracking_err_p95":  d.get("Metrics/tracking_err_p95"),
                    "success_rate":      d.get("Metrics/success_rate"),
                })
    with open(CSV_V4, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[✓] {CSV_V4}  ({len(rows)} rows)")
    return pd.DataFrame(rows)


# ── Step 2: 4×4 heatmap with mean ± std ──────────────────────────────────────
def plot_heatmap(df: pd.DataFrame):
    agg  = (df.groupby(["train", "eval_shape"])["tracking_err_mean"]
              .agg(["mean", "std"]).unstack())
    mean = agg["mean"].reindex(index=TRAIN_CONDITIONS, columns=EVAL_SHAPES)
    std  = agg["std"].reindex(index=TRAIN_CONDITIONS, columns=EVAL_SHAPES)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(mean.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=0.0, vmax=2.0)

    ax.set_xticks(range(4))
    ax.set_xticklabels(EVAL_SHAPES, rotation=20, ha="right", fontsize=12)
    ax.set_yticks(range(4))
    ax.set_yticklabels(ROW_LABELS, fontsize=11)
    ax.set_xlabel("Eval trajectory", fontsize=12)
    ax.set_ylabel("Train condition (all with penalty annealing)", fontsize=12)
    ax.set_title(
        "Mean ± std tracking error (m) — final matrix, annealB recipe\n"
        "(3 seeds · lower = better · blue = diagonal)",
        fontsize=13,
    )

    for i in range(4):
        for j in range(4):
            m_val = mean.values[i, j]
            s_val = std.values[i, j]
            if not np.isnan(m_val):
                txt_color = "white" if m_val > 1.5 else "black"
                s_str = f"±{s_val:.2f}" if not np.isnan(s_val) else ""
                ax.text(j, i, f"{m_val:.2f}\n{s_str}",
                        ha="center", va="center", fontsize=10,
                        color=txt_color, fontweight="bold", linespacing=1.4)

    # Diagonal: experts on their own task (rows 1-3 map to cols 1-3)
    for k in range(1, 4):
        rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                              linewidth=2.5, edgecolor="blue",
                              facecolor="none")
        ax.add_patch(rect)

    # Separator below position-controller row
    ax.axhline(0.5, color="white", linewidth=2, linestyle="--")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Mean tracking error (m)")
    plt.tight_layout()
    p = OUT / "heatmap_v4_4x4.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {p}")


# ── Step 3: v3 (fixed) vs v4 (annealB) comparison ────────────────────────────
def plot_v3_vs_v4(df_v4: pd.DataFrame):
    df_v3 = pd.read_csv(CSV_V3)

    # Mapping: (panel title, v3 train name, v4 train name)
    compare = [
        ("Hover3D",    "Hover3D",       "Hover3D_annealB"),
        ("Circle",     "Circle_v3",     "Circle_annealB"),
        ("Lemniscate", "Lemniscate_v3", "Lemniscate_annealB"),
        ("Lissajous",  "Lissajous_v3",  "Lissajous_annealB"),
    ]

    agg3 = (df_v3.groupby(["train", "eval_shape"])["tracking_err_mean"]
                 .agg(["mean", "std"]))
    agg4 = (df_v4.groupby(["train", "eval_shape"])["tracking_err_mean"]
                 .agg(["mean", "std"]))

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    for ax, (label, k3, k4) in zip(axes, compare):
        v3_m = [agg3.loc[(k3, s), "mean"] if (k3, s) in agg3.index else np.nan
                for s in EVAL_SHAPES]
        v3_s = [agg3.loc[(k3, s), "std"]  if (k3, s) in agg3.index else 0
                for s in EVAL_SHAPES]
        v4_m = [agg4.loc[(k4, s), "mean"] if (k4, s) in agg4.index else np.nan
                for s in EVAL_SHAPES]
        v4_s = [agg4.loc[(k4, s), "std"]  if (k4, s) in agg4.index else 0
                for s in EVAL_SHAPES]

        x, w = np.arange(4), 0.35
        ax.bar(x - w/2, v3_m, w, yerr=v3_s, capsize=4,
               label="fixed penalties (v3)", color="#4C72B0",
               alpha=0.85, edgecolor="black")
        ax.bar(x + w/2, v4_m, w, yerr=v4_s, capsize=4,
               label="annealed penalties (v4)", color="#DD8452",
               alpha=0.85, edgecolor="black")

        ax.set_title(f"Train: {label}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(EVAL_SHAPES, rotation=20, ha="right", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 2.5)
        if ax is axes[0]:
            ax.set_ylabel("Mean tracking error (m)", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Effect of penalty-annealing curriculum on every training condition\n"
        "(lower = better · error bars = std over 3 seeds)",
        fontsize=13,
    )
    plt.tight_layout()
    p = OUT / "v3_vs_v4_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {p}")


# ── Step 4: timeseries per training condition ────────────────────────────────
def plot_timeseries():
    for cond, label in zip(TRAIN_CONDITIONS,
                           ["Hover3D (position ctrl)", "Circle",
                            "Lemniscate", "Lissajous"]):
        fig, ax = plt.subplots(figsize=(12, 6))
        any_data = False

        for eval_shape, color in zip(EVAL_SHAPES, COLORS):
            t, mean, std = get_ts_mean_std(cond, eval_shape)
            if t is None:
                print(f"  [!] Missing timeseries: {cond} → {eval_shape}")
                continue
            any_data = True
            ax.plot(t, mean, color=color, linewidth=2,
                    label=f"Eval: {eval_shape}")
            if std.max() > 0:
                ax.fill_between(t, mean - std, mean + std,
                                color=color, alpha=0.15)

        if not any_data:
            plt.close()
            continue

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Mean tracking error (m)", fontsize=12)
        ax.set_title(
            f"Policy trained on {label} with penalty annealing —"
            f" tracking error over time\n(mean ± std over 3 seeds)",
            fontsize=13,
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        p = OUT / f"timeseries_v4_{cond}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[✓] {p}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Building eval_matrix_v4.csv ─────────────────────")
    df_v4 = build_csv()

    print("\n── Heatmap (4×4, annealB) ──────────────────────────")
    plot_heatmap(df_v4)

    print("\n── v3 vs v4 comparison ─────────────────────────────")
    plot_v3_vs_v4(df_v4)

    print("\n── Timeseries plots ────────────────────────────────")
    plot_timeseries()

    print(f"\nDone. Plots in: {OUT}")