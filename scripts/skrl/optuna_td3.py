"""
Optuna hyperparameter search for TD3 on the Crazyflie Hover task.
Each trial:
  1. Optuna suggests TD3 hyperparameters + reward_scale
  2. Writes a temp agent yaml
  3. Launches scripts/skrl/train.py as a subprocess (short training)
  4. Reads tail-mean reward from the run's TensorBoard logs
  5. Returns it to Optuna (maximize)

Usage:
    python scripts/skrl/optuna_td3.py --n_trials 40 --trial_steps 50000
"""
import argparse
import copy
import glob
import os
import subprocess
import tempfile
import time
from pathlib import Path

import optuna
import yaml
from tensorboard.backend.event_processing import event_accumulator

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO       = Path(__file__).resolve().parents[2]
TRAIN      = REPO / "scripts" / "skrl" / "train.py"
BASE_YAML  = (REPO / "source/crazyflie/crazyflie/tasks/direct/quadcopter"
                   / "agents/skrl_td3_cfg.yaml")
LOG_ROOT   = REPO / "logs" / "skrl" / "quadcopter_td3"
TASK       = "Isaac-Quadcopter-HoverTD3-Direct-v0"

# ── CLI ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--n_trials",    type=int, default=40)
ap.add_argument("--trial_steps", type=int, default=50000)
ap.add_argument("--num_envs",    type=int, default=64)
ap.add_argument("--study_name",  type=str, default="td3_hover")
args = ap.parse_args()

with open(BASE_YAML) as f:
    BASE_CFG = yaml.safe_load(f)


# def read_tail_mean_reward(run_dir: Path, tail_frac: float = 0.2) -> float:
#     """Read 'Reward / Total reward (mean)' from TensorBoard, average the tail."""
#     # skrl writes events under run_dir
#     event_files = glob.glob(str(run_dir / "**" / "events.out.tfevents.*"), recursive=True)
#     if not event_files:
#         return -1e9
#     ea = event_accumulator.EventAccumulator(
#         os.path.dirname(event_files[-1]),
#         size_guidance={event_accumulator.SCALARS: 0},
#     )
#     ea.Reload()
#     # find the reward tag (skrl naming varies slightly)
#     tag = None
#     for candidate in ea.Tags().get("scalars", []):
#         if "Total reward" in candidate or "Instantaneous reward" in candidate:
#             tag = candidate
#             break
#     if tag is None:
#         return -1e9
#     vals = [e.value for e in ea.Scalars(tag)]
#     if not vals:
#         return -1e9
#     n_tail = max(1, int(len(vals) * tail_frac))
#     return float(sum(vals[-n_tail:]) / n_tail)

def read_metric(run_dir: Path, tail_frac: float = 0.2) -> float:
    """Mean episode length over the tail of training — scale-invariant learning signal."""
    event_files = glob.glob(str(run_dir / "**" / "events.out.tfevents.*"), recursive=True)
    if not event_files:
        return -1e9
    ea = event_accumulator.EventAccumulator(
        os.path.dirname(event_files[-1]),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    tag = "Episode / Total timesteps (mean)"
    if tag not in ea.Tags().get("scalars", []):
        return -1e9
    vals = [e.value for e in ea.Scalars(tag)]
    if not vals:
        return -1e9
    n_tail = max(1, int(len(vals) * tail_frac))
    return float(sum(vals[-n_tail:]) / n_tail)


def objective(trial: optuna.Trial) -> float:
    # ── Suggest hyperparameters ──
    lr            = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096])
    polyak        = trial.suggest_float("polyak", 1e-3, 2e-2, log=True)
    expl_std      = trial.suggest_float("exploration_std", 0.05, 0.3)
    policy_delay  = trial.suggest_categorical("policy_delay", [2, 3, 4])
    learn_starts  = trial.suggest_categorical("learning_starts", [1000, 5000, 10000])
    grad_steps    = trial.suggest_categorical("gradient_steps", [1, 2])
    reward_scale  = trial.suggest_float("reward_scale", 0.01, 1.0, log=True)

    # ── Build trial agent yaml ──
    cfg = copy.deepcopy(BASE_CFG)
    cfg["agent"]["learning_rate"]    = lr
    cfg["agent"]["batch_size"]       = batch_size
    cfg["agent"]["polyak"]           = polyak
    cfg["agent"]["policy_delay"]     = policy_delay
    cfg["agent"]["learning_starts"]  = learn_starts
    cfg["agent"]["random_timesteps"] = learn_starts
    cfg["agent"]["gradient_steps"]   = grad_steps
    cfg["agent"]["exploration_noise_kwargs"]["std"] = expl_std
    cfg["trainer"]["timesteps"]      = args.trial_steps
    cfg["agent"]["experiment"]["experiment_name"] = f"optuna_t{trial.number}"

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir="/tmp"
    )
    yaml.safe_dump(cfg, tmp)
    tmp.close()

    # ── Launch training subprocess ──
    cmd = [
        "python", str(TRAIN),
        "--task", TASK,
        "--algorithm", "TD3",
        "--num_envs", str(args.num_envs),
        "--headless",
        "--reward_scale", str(reward_scale),
        "--agent_cfg_file", tmp.name,
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(tmp.name)

    if result.returncode != 0:
        print(f"[Trial {trial.number}] FAILED:\n{result.stderr[-1500:]}")
        return -1e9

    # ── Find this trial's run dir (most recent containing the experiment name) ──
    candidates = sorted(LOG_ROOT.glob(f"*optuna_t{trial.number}*"),
                        key=os.path.getmtime)
    if not candidates:
        # fall back to most recent run dir
        candidates = sorted(LOG_ROOT.glob("*_td3_torch*"), key=os.path.getmtime)
    run_dir = candidates[-1]

    score = read_metric(run_dir)
    print(f"[Trial {trial.number}] ep_len={score:.1f} "
          f"({time.time()-t0:.0f}s)  lr={lr:.1e} rs={reward_scale:.3f} "
          f"bs={batch_size} polyak={polyak:.4f}")
    return score


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"  reward: {study.best_value:.3f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save best params
    out = REPO / "logs" / "optuna_td3_best.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(study.best_params, f)
    print(f"\nSaved best params to {out}")