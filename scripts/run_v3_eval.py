"""
v3 eval matrix:
  Train: Hover3D, Circle_v3, Lemniscate_v3, Lissajous_v3  (3 seeds each)
  Eval:  Hover, Static, Circle, Lemniscate, Lissajous       (5 shapes)
  Total: 4 × 3 × 5 = 60 eval runs
"""
import json
import subprocess
from itertools import product
from pathlib import Path

LOGS = Path("logs/rsl_rl")
PLAY = "scripts/rsl_rl/play.py"
NUM_ENVS = 64
SEEDS = [0, 1, 2]
EVAL_SHAPES = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]

# Training conditions: (experiment_name, task_to_load_with)
TRAIN_CONDITIONS = [
    ("Hover3D",        "Isaac-Quadcopter-Hover-Direct-v0"),
    ("Circle_v3",      "Isaac-Quadcopter-Circle-Direct-v0"),
    ("Lemniscate_v3",  "Isaac-Quadcopter-Lemniscate-Direct-v0"),
    ("Lissajous_v3",   "Isaac-Quadcopter-Lissajous-Direct-v0"),
]

EVAL_SHAPE_LOWER = {
    "Hover": "hover", "Static": "static", "Circle": "circle",
    "Lemniscate": "lemniscate", "Lissajous": "lissajous",
}


def find_checkpoint(exp_name: str, seed: int) -> Path:
    pattern = f"{exp_name}_seed{seed}/*/model_399.pt"
    matches = sorted(LOGS.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint: {exp_name} seed {seed}")
    return matches[-1]


def run_eval(exp_name: str, task: str, seed: int, eval_shape: str):
    ckpt = find_checkpoint(exp_name, seed)
    out_json = ckpt.parent.parent / f"eval-on-{eval_shape}.json"

    cmd = [
        "python", PLAY,
        "--task", task,
        "--num_envs", str(NUM_ENVS),
        "--checkpoint", str(ckpt),
        "--metrics_out", str(out_json),
        "--eval_trajectory", EVAL_SHAPE_LOWER[eval_shape],
        "--headless",
    ]
    print(f"  [{exp_name} s{seed}] → {eval_shape}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr[-500:]}")
        return None
    with open(out_json) as f:
        return json.load(f)


def main():
    results = []
    combos = list(product(TRAIN_CONDITIONS, SEEDS, EVAL_SHAPES))
    total = len(combos)

    for i, ((exp_name, task), seed, eval_shape) in enumerate(combos, 1):
        print(f"\n[{i}/{total}] {exp_name} seed{seed} → {eval_shape}")
        m = run_eval(exp_name, task, seed, eval_shape)
        if m is None:
            continue
        results.append({
            "train": exp_name,
            "eval_shape": eval_shape,
            "seed": seed,
            "tracking_err_mean": m.get("Metrics/tracking_err_mean"),
            "tracking_err_p95":  m.get("Metrics/tracking_err_p95"),
            "success_rate":      m.get("Metrics/success_rate"),
        })

    # Save CSV
    import csv
    out_csv = Path("logs/eval_matrix_final.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Saved {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    main()