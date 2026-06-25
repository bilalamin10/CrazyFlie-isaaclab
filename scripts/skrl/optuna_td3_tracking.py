"""Optuna search for TD3 minimizing MEAN TRACKING ERROR (meters).
Each trial: train (annealed task) -> eval_td3.py -> read tracking_err_mean -> minimize."""
import argparse, copy, glob, json, os, subprocess, tempfile, time
from pathlib import Path
import optuna, yaml

REPO      = Path(__file__).resolve().parents[2]
TRAIN     = REPO / "scripts" / "skrl" / "train.py"
EVAL      = REPO / "scripts" / "skrl" / "eval_td3.py"
BASE_YAML = REPO / "source/crazyflie/crazyflie/tasks/direct/quadcopter/agents/skrl_td3_cfg.yaml"
LOG_ROOT  = REPO / "logs" / "skrl" / "quadcopter_td3"
TASK      = "Isaac-Quadcopter-HoverTD3-Anneal-Direct-v0"

ap = argparse.ArgumentParser()
ap.add_argument("--n_trials",    type=int, default=20)
ap.add_argument("--trial_steps", type=int, default=50000)
ap.add_argument("--num_envs",    type=int, default=64)
ap.add_argument("--eval_steps",  type=int, default=2400)
ap.add_argument("--transient",   type=int, default=300)
ap.add_argument("--study_name",  type=str, default="td3_tracking")
args = ap.parse_args()

with open(BASE_YAML) as f:
    BASE_CFG = yaml.safe_load(f)


def objective(trial):
    lr           = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    polyak       = trial.suggest_float("polyak", 1e-3, 2e-2, log=True)
    expl_std     = trial.suggest_float("exploration_std", 0.05, 0.2)
    policy_delay = trial.suggest_categorical("policy_delay", [2, 3])
    learn_starts = trial.suggest_categorical("learning_starts", [1000, 5000])
    reward_scale = trial.suggest_float("reward_scale", 0.01, 0.3, log=True)

    cfg = copy.deepcopy(BASE_CFG)
    cfg["agent"]["learning_rate"]    = lr
    cfg["agent"]["batch_size"]       = batch_size
    cfg["agent"]["polyak"]           = polyak
    cfg["agent"]["policy_delay"]     = policy_delay
    cfg["agent"]["learning_starts"]  = learn_starts
    cfg["agent"]["random_timesteps"] = learn_starts
    cfg["agent"]["exploration_noise_kwargs"]["std"] = expl_std
    cfg["trainer"]["timesteps"]      = args.trial_steps
    cfg["agent"]["experiment"]["experiment_name"] = f"opt_track_t{trial.number}"

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir="/tmp")
    yaml.safe_dump(cfg, tmp); tmp.close()

    # --- train ---
    train_cmd = [
        "python", str(TRAIN), "--task", TASK, "--algorithm", "TD3",
        "--num_envs", str(args.num_envs), "--headless",
        "--reward_scale", str(reward_scale), "--agent_cfg_file", tmp.name,
    ]
    t0 = time.time()
    r = subprocess.run(train_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[Trial {trial.number}] TRAIN FAILED:\n{r.stderr[-1200:]}")
        os.unlink(tmp.name)
        return 1e9

    # --- find checkpoint ---
    runs = sorted(LOG_ROOT.glob(f"*opt_track_t{trial.number}*"), key=os.path.getmtime)
    if not runs:
        runs = sorted(LOG_ROOT.glob("*_td3_torch*"), key=os.path.getmtime)
    ckpts = sorted((runs[-1] / "checkpoints").glob("agent_*.pt"), key=os.path.getmtime)
    if not ckpts:
        os.unlink(tmp.name); return 1e9
    ckpt = ckpts[-1]

    # --- eval ---
    metrics_out = f"/tmp/opt_track_t{trial.number}.json"
    eval_cmd = [
        "python", str(EVAL), "--task", TASK, "--algorithm", "TD3",
        "--num_envs", str(args.num_envs), "--headless",
        "--checkpoint", str(ckpt), "--metrics_out", metrics_out,
        "--num_steps", str(args.eval_steps), "--transient_steps", str(args.transient),
    ]
    re = subprocess.run(eval_cmd, capture_output=True, text=True)
    os.unlink(tmp.name)
    if re.returncode != 0 or not os.path.exists(metrics_out):
        print(f"[Trial {trial.number}] EVAL FAILED:\n{re.stderr[-1200:]}")
        return 1e9

    with open(metrics_out) as f:
        m = json.load(f)
    err = m.get("Metrics/tracking_err_mean", 1e9)
    sr  = m.get("Metrics/success_rate", 0.0)
    print(f"[Trial {trial.number}] err={err:.3f}m SR={sr:.2f} "
          f"({time.time()-t0:.0f}s) lr={lr:.1e} rs={reward_scale:.3f} bs={batch_size}",
          flush=True)
    return err


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name=args.study_name,
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials)
    print("\n" + "="*60 + "\nBEST TRIAL\n" + "="*60)
    print(f"  tracking_err: {study.best_value:.4f} m")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    out = REPO / "logs" / "optuna_td3_tracking_best.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(study.best_params, f)
    print(f"\nSaved to {out}")