"""Lookahead sweep on Static-trained policies.

For each (seed, eval_shape, lookahead) combination, evaluate the Static policy.
Writes a CSV with results.
"""

import csv
import json
import subprocess
from itertools import product
from pathlib import Path

SEEDS = [0, 1, 2]
EVAL_SHAPES = ["Circle", "Lemniscate", "Lissajous"]  # skip Hover (stationary)
LOOKAHEADS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
NUM_EPISODES = 30
NUM_ENVS = 64

LOGS = Path("logs/rsl_rl")
OUTPUT_CSV = Path("logs/lookahead_sweep.csv")
PLAY_SCRIPT = "scripts/rsl_rl/play.py"


def find_checkpoint(seed):
    matches = sorted(LOGS.glob(f"Static_seed{seed}/*/model_399.pt"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint for Static_seed{seed}")
    return matches[-1]


def evaluate(seed, eval_shape, lookahead):
    ckpt = find_checkpoint(seed)
    out_json = LOGS / f"lookahead_sweep/static_seed{seed}_on_{eval_shape}_la{lookahead}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", PLAY_SCRIPT,
        "--task", f"Isaac-Quadcopter-{eval_shape}-Direct-v0",
        "--num_envs", str(NUM_ENVS),
        "--checkpoint", str(ckpt),
        "--num_episodes", str(NUM_EPISODES),
        "--metrics_out", str(out_json),
        "--trajectory_lookahead", str(lookahead),
        "--headless",
    ]
    print(f"  → seed={seed} eval={eval_shape} lookahead={lookahead}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Failed: {result.stderr[-500:]}")
        return None
    with open(out_json) as f:
        return json.load(f)


def main():
    rows = []
    combos = list(product(SEEDS, EVAL_SHAPES, LOOKAHEADS))
    total = len(combos)

    for i, (seed, eval_shape, lookahead) in enumerate(combos, 1):
        print(f"\n[{i}/{total}] seed={seed} eval={eval_shape} lookahead={lookahead}")
        metrics = evaluate(seed, eval_shape, lookahead)
        if metrics is None:
            continue
        rows.append({
            "seed": seed,
            "eval_shape": eval_shape,
            "lookahead": lookahead,
            "tracking_err_mean": metrics.get("Metrics/tracking_err_mean"),
            "tracking_err_p95": metrics.get("Metrics/tracking_err_p95"),
            "success_rate": metrics.get("Metrics/success_rate"),
            "num_episodes": metrics.get("num_episodes"),
        })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()