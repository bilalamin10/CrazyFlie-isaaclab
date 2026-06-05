"""
Generate thesis plots:
  1. Tracking error heatmap (6x5: 5 original shapes + Hover3D row)
  2. One timeseries plot per training shape (6 plots),
     each showing all 5 eval shapes, mean +/- std over 3 seeds.

Usage:
    python scripts/3D_results_comparison.py

Expects:
    logs/eval_matrix_v2.csv
    logs/rsl_rl/{Shape}_v2_seed{0,1,2}/*/eval-on-{Shape}_timeseries.npy
    logs/rsl_rl/Hover3D_seed{0,1,2}/*/eval-on-{Shape}_timeseries.npy
    logs/rsl_rl/Hover3D_seed{0,1,2}/*/eval-on-{Shape}.json
    logs/hover3d_on_{shape}_timeseries.npy  (fallback for seed 0)
    logs/hover3d_on_{shape}.json            (fallback for seed 0)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SHAPES_5   = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]
TRAIN_ROWS = ["Hover", "Hover3D", "Static", "Circle", "Lemniscate", "Lissajous"]
SEEDS      = [0, 1, 2]
LOGS       = Path("logs/rsl_rl")
CSV_PATH   = Path("logs/eval_matrix_v2.csv")
OUT        = Path("logs/plots")
DT         = 1.0 / 60.0
COLORS     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
OUT.mkdir(parents=True, exist_ok=True)


# ── Helpers: v2 shapes ────────────────────────────────────────────────────────
def load_v2_timeseries(train_shape: str, eval_shape: str, seed: int):
    pattern = f"{train_shape}_v2_seed{seed}/*/eval-on-{eval_shape}_timeseries.npy"
    matches = sorted(LOGS.glob(pattern))
    return np.load(matches[-1]) if matches else None


def get_mean_std_v2(train_shape: str, eval_shape: str):
    series = [s for seed in SEEDS
              for s in [load_v2_timeseries(train_shape, eval_shape, seed)]
              if s is not None and len(s) > 0]
    if not series:
        return None, None, None
    min_len = min(len(s) for s in series)
    stacked = np.stack([s[:min_len] for s in series])
    return np.arange(min_len) * DT, stacked.mean(0), stacked.std(0)


# ── Helpers: Hover3D ──────────────────────────────────────────────────────────
def load_hover3d_timeseries(eval_shape: str, seed: int):
    """Load Hover3D timeseries — checks direct folder and timestamped subfolder."""
    seed_dir = LOGS / f"Hover3D_seed{seed}"

    # Case 1: saved directly in seed folder (eval run with metrics_out to seed dir)
    p = seed_dir / f"eval-on-{eval_shape.lower()}_timeseries.npy"
    if p.exists():
        return np.load(p)

    # Case 2: saved inside timestamped subfolder
    matches = sorted(seed_dir.glob(f"*/eval-on-{eval_shape}_timeseries.npy"))
    if matches:
        return np.load(matches[-1])

    # Case 3: fallback for seed 0 old location
    if seed == 0:
        p = Path(f"logs/hover3d_on_{eval_shape.lower()}_timeseries.npy")
        if p.exists():
            return np.load(p)

    return None


def get_hover3d_mean_std(eval_shape: str):
    """Mean ± std timeseries across available Hover3D seeds."""
    series = [s for seed in SEEDS
              for s in [load_hover3d_timeseries(eval_shape, seed)]
              if s is not None and len(s) > 0]
    if not series:
        return None, None, None
    min_len = min(len(s) for s in series)
    stacked = np.stack([s[:min_len] for s in series])
    n_seeds = stacked.shape[0]
    t = np.arange(min_len) * DT
    mean = stacked.mean(0)
    # std only meaningful with >1 seed
    std = stacked.std(0) if n_seeds > 1 else np.zeros_like(mean)
    return t, mean, std


def get_hover3d_mean_error(eval_shape: str) -> float:
    vals = []
    for seed in SEEDS:
        seed_dir = LOGS / f"Hover3D_seed{seed}"

        # Case 1: direct in seed folder (lowercase)
        p = seed_dir / f"eval-on-{eval_shape.lower()}.json"
        if p.exists():
            d = json.loads(p.read_text())
            v = d.get("Metrics/tracking_err_mean")
            if v is not None:
                vals.append(v)
            continue

        # Case 2: timestamped subfolder
        matches = sorted(seed_dir.glob(f"*/eval-on-{eval_shape}.json"))
        if matches:
            d = json.loads(matches[-1].read_text())
            v = d.get("Metrics/tracking_err_mean")
            if v is not None:
                vals.append(v)
            continue

        # Case 3: fallback seed 0
        if seed == 0:
            fb = Path(f"logs/hover3d_on_{eval_shape.lower()}.json")
            if fb.exists():
                d = json.loads(fb.read_text())
                v = d.get("Metrics/tracking_err_mean")
                if v is not None:
                    vals.append(v)

    return float(np.mean(vals)) if vals else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Extended heatmap (6×5)
# ══════════════════════════════════════════════════════════════════════════════
def plot_heatmap():
    df = pd.read_csv(CSV_PATH)

    # Compute mean AND std per cell across 3 seeds
    agg = (df.groupby(["train_shape", "eval_shape"])["tracking_err_mean"]
             .agg(["mean", "std"])
             .unstack())
    mean5 = agg["mean"].reindex(index=SHAPES_5, columns=SHAPES_5)
    std5  = agg["std"].reindex(index=SHAPES_5, columns=SHAPES_5)

    # Build 6×5 arrays (mean and std)
    data_mean = np.full((6, 5), np.nan)
    data_std  = np.full((6, 5), np.nan)

    for i, train in enumerate(TRAIN_ROWS):
        for j, eval_s in enumerate(SHAPES_5):
            if train == "Hover3D":
                data_mean[i, j] = get_hover3d_mean_error(eval_s)
                # compute std across Hover3D seeds
                vals = []
                for seed in SEEDS:
                    seed_dir = LOGS / f"Hover3D_seed{seed}"
                    p = seed_dir / f"eval-on-{eval_s.lower()}.json"
                    if p.exists():
                        d = json.loads(p.read_text())
                        v = d.get("Metrics/tracking_err_mean")
                        if v is not None:
                            vals.append(v)
                    else:
                        if seed == 0:
                            fb = Path(f"logs/hover3d_on_{eval_s.lower()}.json")
                            if fb.exists():
                                d = json.loads(fb.read_text())
                                v = d.get("Metrics/tracking_err_mean")
                                if v is not None:
                                    vals.append(v)
                data_std[i, j] = float(np.std(vals)) if len(vals) > 1 else 0.0
            elif train in mean5.index:
                data_mean[i, j] = mean5.loc[train, eval_s]
                data_std[i, j]  = std5.loc[train, eval_s] \
                                   if not np.isnan(std5.loc[train, eval_s]) else 0.0

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data_mean, cmap="RdYlGn_r", aspect="auto", vmin=0.0, vmax=2.5)

    ax.set_xticks(range(5))
    ax.set_xticklabels(SHAPES_5, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(6))
    ax.set_yticklabels(TRAIN_ROWS, fontsize=11)
    ax.set_xlabel("Eval trajectory", fontsize=12)
    ax.set_ylabel("Train trajectory", fontsize=12)
    ax.set_title(
        "Mean ± std tracking error (m) — cross-trajectory generalisation\n"
        "(lower = better  ·  blue border = diagonal  ·  3 seeds per cell)",
        fontsize=13,
    )

    # Annotate with mean ± std
    for i in range(6):
        for j in range(5):
            mean_val = data_mean[i, j]
            std_val  = data_std[i, j]
            if not np.isnan(mean_val):
                txt_color = "white" if mean_val > 1.8 else "black"
                if std_val > 0:
                    label = f"{mean_val:.2f}\n±{std_val:.2f}"
                else:
                    label = f"{mean_val:.2f}\n(1 seed)"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=8.5, color=txt_color, fontweight="bold",
                        linespacing=1.4)

    # Highlight diagonal (skip Hover3D row at index 1)
    diag_row = {s: i for i, s in enumerate(TRAIN_ROWS)}
    for k, shape in enumerate(SHAPES_5):
        row_idx = diag_row.get(shape, -1)
        if row_idx >= 0 and row_idx != 1:
            rect = plt.Rectangle((k - 0.5, row_idx - 0.5), 1, 1,
                                  linewidth=2.5, edgecolor="blue",
                                  facecolor="none")
            ax.add_patch(rect)

    # Dashed lines bracketing Hover3D row
    ax.axhline(0.5, color="white", linewidth=2, linestyle="--")
    ax.axhline(1.5, color="white", linewidth=2, linestyle="--")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Mean tracking error (m)")
    plt.tight_layout()
    p = OUT / "heatmap_extended_6x5.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — 2D vs 3D Hover comparison bar chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_hover_comparison():
    df = pd.read_csv(CSV_PATH)
    hover2d = (df[df.train_shape == "Hover"]
               .groupby("eval_shape")["tracking_err_mean"]
               .mean().reindex(SHAPES_5))
    hover3d = pd.Series({s: get_hover3d_mean_error(s) for s in SHAPES_5})

    x, w = np.arange(5), 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w/2, hover2d.values, w, label="Hover 2D (fixed origin)",
                color="#4C72B0", alpha=0.85, edgecolor="black")
    b2 = ax.bar(x + w/2, hover3d.values, w, label="Hover 3D (cuboid)",
                color="#DD8452", alpha=0.85, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(SHAPES_5, fontsize=11)
    ax.set_ylabel("Mean tracking error (m)", fontsize=12)
    ax.set_title("2D vs 3D position controller — tracking error on all eval shapes\n"
                 "(lower = better)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 2.4)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.04,
                    f"{h:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    p = OUT / "hover_2d_vs_3d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Timeseries: one figure per training shape (6 total)
# ══════════════════════════════════════════════════════════════════════════════
def plot_one_timeseries(train_shape: str, get_fn, suffix=""):
    fig, ax = plt.subplots(figsize=(12, 6))
    any_data = False

    for eval_shape, color in zip(SHAPES_5, COLORS):
        t, mean, std = get_fn(eval_shape)
        if t is None:
            print(f"  [!] Missing timeseries: {train_shape} → {eval_shape}")
            continue
        any_data = True
        ax.plot(t, mean, color=color, linewidth=2, label=f"Eval: {eval_shape}")
        if std is not None and std.max() > 0:
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)

    if not any_data:
        plt.close()
        return

    n_seeds_label = "3 seeds" if suffix == "" else "mean ± std over 3 seeds" if suffix == "_3s" else "single seed"
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Mean tracking error (m)", fontsize=12)
    ax.set_title(
        f"Policy trained on {train_shape} — tracking error over time\n"
        f"(mean ± std over 3 seeds)",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.tight_layout()
    p = OUT / f"timeseries_train_{train_shape}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {p}")


def plot_timeseries():
    # v2 shapes — 3 seeds each
    for train_shape in SHAPES_5:
        plot_one_timeseries(
            train_shape,
            lambda es, ts=train_shape: get_mean_std_v2(ts, es),
        )

    # Hover3D — up to 3 seeds (with fallback to seed 0 if others not ready)
    plot_one_timeseries(
        "Hover3D",
        lambda es: get_hover3d_mean_std(es),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("── Heatmap ─────────────────────────────────────────")
    plot_heatmap()

    print("\n── 2D vs 3D Hover comparison ───────────────────────")
    plot_hover_comparison()

    print("\n── Timeseries (one plot per training shape) ────────")
    plot_timeseries()

    print(f"\nDone. All plots in: {OUT}")