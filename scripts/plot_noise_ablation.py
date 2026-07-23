"""Create thesis-ready plots for the clean-versus-noisy observation ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Plotting requires matplotlib and pandas. Install them in the Python "
        "environment used to run this script."
    ) from exc


SHAPE_ORDER = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]
CONDITION_ORDER = [
    ("off", "off"),
    ("on", "off"),
    ("off", "on"),
    ("on", "on"),
]
CONDITION_LABELS = [
    "Clean train / clean test",
    "Noisy train / clean test",
    "Clean train / noisy test",
    "Noisy train / noisy test",
]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("logs/noise_ablation/noise_ablation_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/noise_ablation/plots"),
    )
    return parser.parse_args()


def validate(df: pd.DataFrame) -> None:
    required = {
        "shape", "seed", "train_noise", "eval_noise",
        "tracking_err_mean", "tracking_err_p95", "success_rate",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["shape", "train_noise", "eval_noise"], as_index=False)
        .agg(
            tracking_err_mean=("tracking_err_mean", "mean"),
            tracking_err_std=("tracking_err_mean", "std"),
            tracking_err_p95=("tracking_err_p95", "mean"),
            success_rate=("success_rate", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    grouped["tracking_err_std"] = grouped["tracking_err_std"].fillna(0.0)
    return grouped


def condition_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    shapes = [shape for shape in SHAPE_ORDER if shape in set(summary["shape"])]
    x = np.arange(len(shapes))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 6.5))

    for index, ((train_noise, eval_noise), label, color) in enumerate(
        zip(CONDITION_ORDER, CONDITION_LABELS, COLORS)
    ):
        subset = summary[
            (summary.train_noise == train_noise)
            & (summary.eval_noise == eval_noise)
        ].set_index("shape")
        means = np.array([
            subset.loc[shape, "tracking_err_mean"] if shape in subset.index else np.nan
            for shape in shapes
        ])
        stds = np.array([
            subset.loc[shape, "tracking_err_std"] if shape in subset.index else 0.0
            for shape in shapes
        ])
        offset = (index - 1.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xticks(x, shapes)
    ax.set_ylabel("Mean tracking error (m)")
    ax.set_title("Effect of observation noise on trajectory tracking")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "noise_ablation_tracking_error.png", dpi=300)
    plt.close(fig)


def clean_baseline_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    clean = summary[summary.eval_noise == "off"]
    shapes = [shape for shape in SHAPE_ORDER if shape in set(clean["shape"])]
    x = np.arange(len(shapes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.8))

    for index, (train_noise, label, color) in enumerate(
        [("off", "Noise-free baseline", COLORS[0]), ("on", "Trained with noise", COLORS[1])]
    ):
        subset = clean[clean.train_noise == train_noise].set_index("shape")
        means = [subset.loc[s, "tracking_err_mean"] if s in subset.index else np.nan for s in shapes]
        stds = [subset.loc[s, "tracking_err_std"] if s in subset.index else 0.0 for s in shapes]
        ax.bar(
            x + (index - 0.5) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_xticks(x, shapes)
    ax.set_ylabel("Mean tracking error (m)")
    ax.set_title("Initial baseline and effect of noise during training\n(clean evaluation; curriculum disabled)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "initial_baseline_clean_evaluation.png", dpi=300)
    plt.close(fig)


def success_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    shapes = [shape for shape in SHAPE_ORDER if shape in set(summary["shape"])]
    x = np.arange(len(shapes))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 6.5))

    for index, ((train_noise, eval_noise), label, color) in enumerate(
        zip(CONDITION_ORDER, CONDITION_LABELS, COLORS)
    ):
        subset = summary[
            (summary.train_noise == train_noise)
            & (summary.eval_noise == eval_noise)
        ].set_index("shape")
        values = [subset.loc[s, "success_rate"] if s in subset.index else np.nan for s in shapes]
        ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xticks(x, shapes)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success rate (tracking error < 0.30 m)")
    ax.set_title("Noise robustness by training and evaluation condition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "noise_ablation_success_rate.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    validate(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = aggregate(df)
    summary.to_csv(args.output_dir.parent / "noise_ablation_summary.csv", index=False)
    clean_baseline_bars(summary, args.output_dir)
    condition_bars(summary, args.output_dir)
    success_bars(summary, args.output_dir)
    print(f"Wrote summary and plots under {args.output_dir.parent}")


if __name__ == "__main__":
    main()
