"""Run the clean-versus-noisy observation ablation used in the thesis.

The design is a 2 x 2 experiment:

* train_noise: off/on
* eval_noise: off/on

Curriculum is disabled in every training run. Each policy is evaluated on the
same trajectory on which it was trained. Use at least three seeds for reported
results; the defaults are 0, 1, and 2.

Examples:
    python scripts/run_noise_ablation.py --dry-run
    python scripts/run_noise_ablation.py
    python scripts/run_noise_ablation.py --shapes Hover Static --seeds 0
    python scripts/run_noise_ablation.py --eval-only
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ALL_SHAPES = ("Hover", "Static", "Circle", "Lemniscate", "Lissajous")
TRAIN_NOISE_LEVELS = ("off", "on")
EVAL_NOISE_LEVELS = ("off", "on")
LOG_ROOT = Path("logs/rsl_rl")
OUTPUT_CSV = Path("logs/noise_ablation/noise_ablation_results.csv")
TRAIN_SCRIPT = Path("scripts/rsl_rl/train.py")
PLAY_SCRIPT = Path("scripts/rsl_rl/play.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", choices=ALL_SHAPES, default=list(ALL_SHAPES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--train-num-envs", type=int, default=None)
    parser.add_argument("--eval-num-envs", type=int, default=64)
    parser.add_argument("--eval-num-steps", type=int, default=2400)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Repeat completed train/eval jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running Isaac Sim.")
    args = parser.parse_args()
    if args.train_only and args.eval_only:
        parser.error("--train-only and --eval-only are mutually exclusive")
    return args


def experiment_name(shape: str, train_noise: str, seed: int) -> str:
    return f"NoiseAblation_train-{train_noise}_{shape}_seed{seed}"


def task_name(shape: str) -> str:
    return f"Isaac-Quadcopter-{shape}-Direct-v0"


def checkpoint_for(experiment: str, final_iteration: int) -> Path | None:
    matches = sorted((LOG_ROOT / experiment).glob(f"*/model_{final_iteration}.pt"))
    return matches[-1] if matches else None


def run_command(command: list[str], dry_run: bool) -> bool:
    print("  $ " + " ".join(command), flush=True)
    if dry_run:
        return True
    result = subprocess.run(command, check=False)
    return result.returncode == 0


def write_results(rows: list[dict]) -> None:
    """Merge results by experiment cell so partial runs do not erase prior data."""
    key_fields = ("shape", "seed", "train_noise", "eval_noise")
    merged: dict[tuple[str, ...], dict] = {}
    if OUTPUT_CSV.exists():
        with OUTPUT_CSV.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = tuple(str(row[field]) for field in key_fields)
                merged[key] = row
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        merged[key] = row

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(
        merged.values(),
        key=lambda row: (
            ALL_SHAPES.index(row["shape"]),
            int(row["seed"]),
            row["train_noise"],
            row["eval_noise"],
        ),
    )
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_rows[0].keys())
        writer.writeheader()
        writer.writerows(ordered_rows)


def train(shape: str, train_noise: str, seed: int, args: argparse.Namespace) -> Path | None:
    experiment = experiment_name(shape, train_noise, seed)
    final_iteration = args.max_iterations - 1
    existing = checkpoint_for(experiment, final_iteration)
    if existing is not None and not args.force:
        print(f"[skip train] {existing}")
        return existing

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--task",
        task_name(shape),
        "--headless",
        "--seed",
        str(seed),
        "--max_iterations",
        str(args.max_iterations),
        "--experiment_name",
        experiment,
        "--run_name",
        f"train-noise-{train_noise}_no-curriculum",
        "--observation_noise",
        train_noise,
        "--disable_curriculum",
    ]
    if args.train_num_envs is not None:
        command.extend(("--num_envs", str(args.train_num_envs)))

    print(f"[train] shape={shape}, train_noise={train_noise}, seed={seed}")
    if not run_command(command, args.dry_run):
        print("  [error] training failed")
        return None
    return checkpoint_for(experiment, final_iteration)


def evaluate(
    checkpoint: Path,
    shape: str,
    train_noise: str,
    eval_noise: str,
    seed: int,
    args: argparse.Namespace,
) -> dict | None:
    output_dir = OUTPUT_CSV.parent / experiment_name(shape, train_noise, seed)
    output_json = output_dir / f"eval-noise-{eval_noise}.json"
    if output_json.exists() and not args.force:
        return json.loads(output_json.read_text())

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PLAY_SCRIPT),
        "--task",
        task_name(shape),
        "--headless",
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        str(args.eval_num_envs),
        "--num_steps",
        str(args.eval_num_steps),
        "--metrics_out",
        str(output_json),
        "--observation_noise",
        eval_noise,
    ]
    print(
        f"[eval] shape={shape}, train_noise={train_noise}, "
        f"eval_noise={eval_noise}, seed={seed}"
    )
    if not run_command(command, args.dry_run):
        print("  [error] evaluation failed")
        return None
    if args.dry_run:
        return None
    return json.loads(output_json.read_text())


def main() -> None:
    args = parse_args()
    rows: list[dict] = []

    for shape in args.shapes:
        for train_noise in TRAIN_NOISE_LEVELS:
            for seed in args.seeds:
                experiment = experiment_name(shape, train_noise, seed)
                checkpoint = checkpoint_for(experiment, args.max_iterations - 1)

                if not args.eval_only:
                    checkpoint = train(shape, train_noise, seed, args)
                if args.train_only or args.dry_run:
                    continue
                if checkpoint is None:
                    print(f"[skip eval] no checkpoint for {experiment}")
                    continue

                for eval_noise in EVAL_NOISE_LEVELS:
                    metrics = evaluate(
                        checkpoint, shape, train_noise, eval_noise, seed, args
                    )
                    if metrics is None:
                        continue
                    rows.append(
                        {
                            "shape": shape,
                            "seed": seed,
                            "train_noise": train_noise,
                            "eval_noise": eval_noise,
                            "tracking_err_mean": metrics.get("Metrics/tracking_err_mean"),
                            "tracking_err_p95": metrics.get("Metrics/tracking_err_p95"),
                            "success_rate": metrics.get("Metrics/success_rate"),
                            "num_steps": metrics.get("num_steps_total"),
                            "checkpoint": str(checkpoint),
                        }
                    )

    if rows:
        write_results(rows)
        print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
        print("Create plots with: python scripts/plot_noise_ablation.py")
    elif not args.dry_run and not args.train_only:
        print("No evaluation rows were produced.")


if __name__ == "__main__":
    main()
