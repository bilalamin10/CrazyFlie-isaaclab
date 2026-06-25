# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate a trained skrl TD3 checkpoint with the SAME metric pipeline as
rsl_rl play.py — deterministic, no exploration noise, eval_mode env,
mean L2 tracking error over a fixed window. Produces a JSON directly
comparable to the PPO matrix."""

import argparse
from isaaclab.app import AppLauncher
#import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate skrl TD3 with tracking metric.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-HoverTD3-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--algorithm", type=str, default="TD3")
parser.add_argument("--ml_framework", type=str, default="torch")
parser.add_argument("--agent", type=str, default=None)
parser.add_argument("--metrics_out", type=str, default=None)
parser.add_argument("--eval_trajectory", type=str, default=None)
parser.add_argument("--num_steps", type=int, default=2400)
parser.add_argument("--transient_steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import json
import os
import torch
import numpy as np

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner
import isaaclab_tasks  # noqa: F401
import crazyflie.tasks  # noqa: F401


def main():
    # ---- env config with eval_mode ON (same as rsl_rl play.py) ----
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.eval_mode = True
    if args_cli.eval_trajectory is not None:
        env_cfg.trajectory_type = args_cli.eval_trajectory

    # ---- agent config from the task entry point ----
    algorithm = args_cli.algorithm.lower()
    entry = args_cli.agent or f"skrl_{algorithm}_cfg_entry_point"
    from isaaclab_tasks.utils import load_cfg_from_registry
    experiment_cfg = load_cfg_from_registry(args_cli.task, entry)

    # ---- build env ----
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # ---- build runner + load checkpoint ----
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    # ---- eval loop (deterministic, no noise) ----
    metric_keys = ["Metrics/tracking_err_mean", "Metrics/tracking_err_p95", "Metrics/success_rate"]
    metric_sums = {k: 0.0 for k in metric_keys}
    metric_counts = {k: 0 for k in metric_keys}
    per_step_error = []
    step_count = 0

    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            norm_obs = runner.agent._observation_preprocessor(obs)
            actions, _ = runner.agent.policy.act({"observations": norm_obs}, role="policy")
            obs, _, _, _, extras = env.step(actions)

        step_count += 1
        log_dict = extras.get("log", {}) if isinstance(extras, dict) else {}
        if "Metrics/tracking_err_mean" in log_dict:
            per_step_error.append(float(log_dict["Metrics/tracking_err_mean"]))
        if step_count > args_cli.transient_steps:
            for k in metric_keys:
                if k in log_dict:
                    metric_sums[k] += float(log_dict[k])
                    metric_counts[k] += 1
        if step_count >= args_cli.num_steps:
            print(f"[INFO] Reached {step_count} steps, stopping.")
            break

    # ---- write JSON ----
    means = {k: (metric_sums[k] / metric_counts[k]) if metric_counts[k] > 0 else float("nan")
             for k in metric_keys}
    means["num_steps_total"] = step_count
    means["num_steps_aggregated"] = max(step_count - args_cli.transient_steps, 0)
    means["num_envs"] = args_cli.num_envs
    print(json.dumps(means, indent=2))

    if args_cli.metrics_out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.metrics_out)), exist_ok=True)
        with open(args_cli.metrics_out, "w") as f:
            json.dump(means, f, indent=2)
        np.save(args_cli.metrics_out.replace(".json", "_timeseries.npy"), np.array(per_step_error))
        print(f"[INFO] Saved metrics to {args_cli.metrics_out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()