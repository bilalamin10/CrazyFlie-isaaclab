# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

parser.add_argument("--num_episodes", type=int, default=None,
                    help="Number of complete episodes to run before exiting. If None, runs forever.")
parser.add_argument("--metrics_out", type=str, default=None,
                    help="Path to save evaluation metrics as JSON.")

parser.add_argument("--trajectory_lookahead", type=float, default=None,
                    help="Setpoint lookahead in seconds (overrides cfg).")

parser.add_argument("--num_steps", type=int, default=2400,
                    help="Total simulation steps to run during eval (default 2400 = ~40 sec).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
import crazyflie.tasks  # noqa: F401 — triggers gym.register


# ==================== DEBUG: CHECK WHICH ENV FILE IS USED ====================
import crazyflie.tasks.direct.quadcopter.quadcopter_env as custom_env
print("="*80)
print("DEBUG: Using CUSTOM quadcopter_env.py from:")
print(custom_env.__file__)
print("="*80)

import isaaclab_tasks
print("Isaac Lab tasks path:", isaaclab_tasks.__file__)
# ============================================================================

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Force eval_mode for evaluation runs — disables training noise/tilt
    env_cfg.eval_mode = True
    #env_cfg.trajectory_lookahead = 0.2  # temporary for this experiment
    if args_cli.trajectory_lookahead is not None:
        env_cfg.trajectory_lookahead = args_cli.trajectory_lookahead

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # Shim for IsaacLab 2.1 / newer rsl_rl compatibility
    if not hasattr(agent_cfg, "obs_groups") or agent_cfg.obs_groups is None:
        agent_cfg.obs_groups = {"policy": ["obs"]}
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    per_step_error = []
    step_count = 0

    # --- Eval metrics accumulators ---
    TRANSIENT_STEPS = 120  # skip first 2 seconds (60 Hz simulation)

    # In the accumulator block:
    if step_count > TRANSIENT_STEPS:
        log_dict = extras.get("log", {}) if isinstance(extras, dict) else {}
        for k in metric_keys:
            if k in log_dict:
                v = log_dict[k]
                metric_sums[k] += float(v)
                metric_counts[k] += 1
    metric_keys = [
        "Metrics/tracking_err_mean",
        "Metrics/tracking_err_p95",
        "Metrics/success_rate",
    ]
    metric_sums = {k: 0.0 for k in metric_keys}
    metric_counts = {k: 0 for k in metric_keys}
    episodes_completed = 0
    num_envs = env.unwrapped.num_envs
    prev_dones = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        # --- Accumulate metrics from env.extras (logged by _log_eval_metrics) ---
        log_dict = extras.get("log", {}) if isinstance(extras, dict) else {}
        for k in metric_keys:
            if k in log_dict:
                v = log_dict[k]
                metric_sums[k] += float(v)
                metric_counts[k] += 1

        # --- Per-step time series capture ---
        if "Metrics/tracking_err_mean" in log_dict:
            per_step_error.append(float(log_dict["Metrics/tracking_err_mean"]))

        # --- Step-based termination (run for fixed simulation time) ---
        step_count += 1
        if step_count >= args_cli.num_steps:
            print(f"[INFO] Reached {step_count} simulation steps "
                f"({step_count/60:.1f}s), stopping eval.")
            break

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # --- Write metrics JSON ---
    if args_cli.metrics_out is not None:
        import json
        means = {
            k: (metric_sums[k] / metric_counts[k]) if metric_counts[k] > 0 else float("nan")
            for k in metric_keys
        }
        means["num_episodes"] = episodes_completed
        means["num_envs"] = num_envs
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.metrics_out)), exist_ok=True)
        with open(args_cli.metrics_out, "w") as f:
            json.dump(means, f, indent=2)
        print(f"[INFO] Saved metrics to {args_cli.metrics_out}")
        print(json.dumps(means, indent=2))

    # --- Save per-step time series as .npy ---
    if args_cli.metrics_out is not None:
        import numpy as np
        npy_path = args_cli.metrics_out.replace(".json", "_timeseries.npy")
        np.save(npy_path, np.array(per_step_error))
        print(f"[INFO] Saved time series to {npy_path}")

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
