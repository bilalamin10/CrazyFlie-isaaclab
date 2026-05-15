"""Plot tracking error over time for all training policies on a given eval shape.

Usage:
    python scripts/plot_timeseries.py --eval Circle
    python scripts/plot_timeseries.py --eval Lemniscate --seed 0
    python scripts/plot_timeseries.py --eval all   # one plot per eval shape, 5 figs total
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SHAPES = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]
LOGS = Path("logs/rsl_rl")
DT = 1.0 / 60.0  # env step_dt


def find_timeseries(train_shape: str, eval_shape: str, seed: int) -> Path | None:
    """Find the _timeseries.npy file produced by eval."""
    pattern = f"{train_shape}_seed{seed}/*/eval-on-{eval_shape}_timeseries.npy"
    matches = sorted(LOGS.glob(pattern))
    return matches[-1] if matches else None


def plot_one(eval_shape: str, seed: int, ax=None, average_seeds: bool = False):
    """Plot one panel: tracking error over time for all training policies on eval_shape."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for train_shape in SHAPES:
        if average_seeds:
            # Average across all seeds
            series_list = []
            for s in [0, 1, 2]:
                p = find_timeseries(train_shape, eval_shape, s)
                if p and p.exists():
                    series_list.append(np.load(p))
            if not series_list:
                continue
            min_len = min(len(s) for s in series_list)
            stacked = np.stack([s[:min_len] for s in series_list])
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            t = np.arange(min_len) * DT
            ax.plot(t, mean, label=f"Trained on {train_shape}", linewidth=2)
            ax.fill_between(t, mean - std, mean + std, alpha=0.15)
        else:
            p = find_timeseries(train_shape, eval_shape, seed)
            if not p:
                print(f"  Missing: {train_shape} seed {seed}")
                continue
            data = np.load(p)
            t = np.arange(len(data)) * DT
            ax.plot(t, data, label=f"Trained on {train_shape}", linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean tracking error (m)")
    title = f"Evaluated on {eval_shape}"
    if average_seeds:
        title += " (mean ± std over 3 seeds)"
    else:
        title += f" (seed {seed})"
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", default="all", help="Eval shape, or 'all' for 5-panel figure")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--avg", action="store_true", help="Average over seeds with std shading")
    args = p.parse_args()

    out_dir = Path("logs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval == "all":
        fig, axes = plt.subplots(2, 3, figsize=(20, 11))
        for ax, eval_shape in zip(axes.flat, SHAPES):
            plot_one(eval_shape, args.seed, ax=ax, average_seeds=args.avg)
        axes.flat[5].axis("off")  # hide unused panel
        plt.tight_layout()
        suffix = "_avg" if args.avg else f"_seed{args.seed}"
        out = out_dir / f"timeseries_all{suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_one(args.eval, args.seed, ax=ax, average_seeds=args.avg)
        plt.tight_layout()
        suffix = "_avg" if args.avg else f"_seed{args.seed}"
        out = out_dir / f"timeseries_on_{args.eval}{suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")

    plt.show()


if __name__ == "__main__":
    main()