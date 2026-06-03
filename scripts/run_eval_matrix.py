"""
Eval matrix orchestrator.

For each trained checkpoint (15 of them = 5 shapes × 3 seeds), evaluate the
policy on all 5 shapes. Produces logs/eval_matrix.csv with one row per
(train_shape, eval_shape, seed) combination.

Usage:
    python scripts/run_eval_matrix.py
"""

import csv
import json
import subprocess
from itertools import product
from pathlib import Path

# --- Config ---
SHAPES = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]
SEEDS = [0, 1, 2]
NUM_EPISODES = 30
NUM_ENVS = 64
LOGS_DIR = Path("logs/rsl_rl")
OUTPUT_CSV = Path("logs/eval_matrix_v2.csv")
PLAY_SCRIPT = "scripts/rsl_rl/play.py"
TRAINING_ITER = 399  # which checkpoint to evaluate (model_399.pt)


def find_checkpoint(train_shape: str, seed: int) -> Path:
    """Find the model_399.pt for a given (train_shape, seed)."""
    pattern = f"{train_shape}_v2_seed{seed}/*/model_{TRAINING_ITER}.pt"
    matches = sorted(LOGS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for {train_shape} seed {seed}")
    # If multiple timestamped subfolders, take the most recent
    return matches[-1]


def evaluate(train_shape: str, eval_shape: str, seed: int) -> dict:
    """Run play.py on a checkpoint with the given eval shape, return metrics."""
    ckpt = find_checkpoint(train_shape, seed)
    out_json = ckpt.parent / f"eval-on-{eval_shape}.json"

    cmd = [
        "python", PLAY_SCRIPT,
        "--task", f"Isaac-Quadcopter-{eval_shape}-Direct-v0",
        "--num_envs", str(NUM_ENVS),
        "--checkpoint", str(ckpt),
        "--num_steps", "2400",        # ← replaces --num_episodes
        "--metrics_out", str(out_json),
        "--headless",
    ]
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] play.py failed for train={train_shape} eval={eval_shape} seed={seed}")
        print(result.stderr[-2000:])  # last bit of stderr
        return None

    with open(out_json) as f:
        return json.load(f)


def main():
    rows = []
    combos = list(product(SHAPES, SHAPES, SEEDS))
    total = len(combos)

    for i, (train_shape, eval_shape, seed) in enumerate(combos, 1):
        print(f"\n[{i}/{total}] train={train_shape} eval={eval_shape} seed={seed}")
        metrics = evaluate(train_shape, eval_shape, seed)
        if metrics is None:
            continue
        rows.append({
            "train_shape": train_shape,
            "eval_shape":  eval_shape,
            "seed":        seed,
            "tracking_err_mean": metrics.get("Metrics/tracking_err_mean"),
            "tracking_err_p95":  metrics.get("Metrics/tracking_err_p95"),
            "success_rate":      metrics.get("Metrics/success_rate"),
            "num_steps_total":   metrics.get("num_steps_total"),   # ← updated
        })

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()