# scripts/run_baselines.py
"""
Usage:
    python scripts/run_baselines.py --shapes Hover Circle Lemniscate \
        --train_iters 1500 --seeds 0 1 2 --eval_episodes 50

Trains one policy per (shape, seed), then evaluates every checkpoint on every shape.
Writes a CSV with columns: train_shape, eval_shape, seed, mean_tracking_err, success_rate.
"""
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "baselines"
RUNS.mkdir(parents=True, exist_ok=True)

def train(shape, seed, iters, headless=True):
    run_dir = RUNS / f"train-{shape}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "scripts/rsl_rl/train.py",
        "--task", f"Isaac-Quadcopter-{shape}-Direct-v0",
        "--num_envs", "4096",
        "--max_iterations", str(iters),
        "--seed", str(seed),
        "--logdir", str(run_dir),
    ]
    if headless:
        cmd.append("--headless")
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # find checkpoint (rsl_rl convention: model_<iter>.pt)
    ckpts = sorted(run_dir.rglob("model_*.pt"))
    return ckpts[-1] if ckpts else None

def evaluate(checkpoint, eval_shape, seed, episodes):
    out_json = checkpoint.parent / f"eval-on-{eval_shape}-seed{seed}.json"
    cmd = [
        "python", "scripts/rsl_rl/play.py",
        "--task", f"Isaac-Quadcopter-{eval_shape}-Direct-v0",
        "--num_envs", "256",
        "--checkpoint", str(checkpoint),
        "--seed", str(seed),
        "--num_episodes", str(episodes),
        "--metrics_out", str(out_json),
        "--headless",
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    with open(out_json) as f:
        return json.load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+",
                   default=["Hover", "Static", "Circle", "Lemniscate", "Lissajous"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--train_iters", type=int, default=1500)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--skip_train", action="store_true",
                   help="Reuse existing checkpoints, only re-evaluate.")
    args = p.parse_args()

    rows = []
    for train_shape, seed in product(args.shapes, args.seeds):
        ckpt_dir = RUNS / f"train-{train_shape}" / f"seed{seed}"
        existing = sorted(ckpt_dir.rglob("model_*.pt"))
        if args.skip_train and existing:
            ckpt = existing[-1]
        else:
            ckpt = train(train_shape, seed, args.train_iters)
        if ckpt is None:
            print(f"[WARN] no checkpoint for {train_shape}/seed{seed}, skipping")
            continue

        for eval_shape in args.shapes:
            metrics = evaluate(ckpt, eval_shape, seed, args.eval_episodes)
            rows.append({
                "train_shape": train_shape,
                "eval_shape": eval_shape,
                "seed": seed,
                **metrics,
            })

    out_csv = RUNS / "baseline_matrix.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()