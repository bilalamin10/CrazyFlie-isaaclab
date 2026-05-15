"""Plot tracking error over time for ONE training policy evaluated on all eval shapes.

Usage:
    python scripts/plot_baseline_one.py --train Hover
    python scripts/plot_baseline_one.py --train Lemniscate --avg
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SHAPES = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]
LOGS = Path("logs/rsl_rl")
DT = 1.0 / 60.0


def find_timeseries(train_shape: str, eval_shape: str, seed: int):
    pattern = f"{train_shape}_seed{seed}/*/eval-on-{eval_shape}_timeseries.npy"
    matches = sorted(LOGS.glob(pattern))
    return matches[-1] if matches else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, choices=SHAPES,
                   help="Training shape (the one policy whose behavior we want to inspect)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--avg", action="store_true",
                   help="Average over seeds with std shading")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(12, 7))

    for eval_shape in SHAPES:
        if args.avg:
            series_list = []
            for s in [0, 1, 2]:
                p = find_timeseries(args.train, eval_shape, s)
                if p and p.exists():
                    series_list.append(np.load(p))
            if not series_list:
                continue
            min_len = min(len(s) for s in series_list)
            stacked = np.stack([s[:min_len] for s in series_list])
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            t = np.arange(min_len) * DT
            ax.plot(t, mean, label=f"Eval on {eval_shape}", linewidth=2)
            ax.fill_between(t, mean - std, mean + std, alpha=0.15)
        else:
            p = find_timeseries(args.train, eval_shape, args.seed)
            if not p:
                print(f"  Missing: {args.train} → {eval_shape} seed {args.seed}")
                continue
            data = np.load(p)
            t = np.arange(len(data)) * DT
            ax.plot(t, data, label=f"Eval on {eval_shape}", linewidth=2)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Mean tracking error (m)", fontsize=12)
    title = f"Policy trained on {args.train} — tracking error on all trajectories"
    if args.avg:
        title += "\n(mean ± std over 3 seeds)"
    else:
        title += f"\n(seed {args.seed})"
    ax.set_title(title, fontsize=13)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = Path("logs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_avg" if args.avg else f"_seed{args.seed}"
    out = out_dir / f"baseline_{args.train}{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()