# play_all_traj.py
"""Evaluate policy on ALL 4 trajectories and save .npy logs."""

import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_steps", type=int, default=2000)

args_cli = parser.parse_args()

AppLauncher.add_app_launcher_args(parser)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Safe imports
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Import your configs directly
from crazyflie.tasks.direct.quadcopter.quadcopter_env import QuadcopterEnvCfg
from crazyflie.tasks.direct.quadcopter.rsl_rl_cfg import QuadcopterPPORunnerCfg
def main():
    print(f"[INFO] Using checkpoint: {args_cli.checkpoint}")

    traj_types = ["static", "circle", "lemniscate", "lissajous"]
    log_root = "logs/eval"
    os.makedirs(log_root, exist_ok=True)

    env_cfg = QuadcopterEnvCfg()
    agent_cfg = QuadcopterPPORunnerCfg()

    for traj in traj_types:
        print(f"\n{'='*80}")
        print(f"[INFO] Evaluating → {traj.upper()}")
        print(f"{'='*80}")

        env_cfg.trajectory_type = traj
        env_cfg.scene.num_envs = args_cli.num_envs

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env)

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=env.device)
        runner.load(args_cli.checkpoint)

        # === FIXED observation handling ===
        obs_dict = env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        for step in range(args_cli.num_steps):
            actions = runner.alg.act(obs)
            step_output = env.step(actions)

            # step_output can be (obs, rew, terminated, truncated, info) or dict
            if isinstance(step_output, dict):
                obs = step_output["policy"] if "policy" in step_output else step_output
            else:
                obs = step_output[0]   # obs is first element
                if isinstance(obs, dict):
                    obs = obs["policy"]

            if (step + 1) % 500 == 0:
                print(f"   Step {step+1}/{args_cli.num_steps} done")

        # Save log
        save_data = {
            "trajectory_type": traj,
            "desired_pos_w": env.unwrapped._desired_pos_w.cpu().numpy(),
            "root_pos_w": env.unwrapped._robot.data.root_pos_w.cpu().numpy(),
            "final_distance_to_goal": float(env.unwrapped.extras.get("Metrics/final_distance_to_goal", 0.0)),
            "mean_distance_reward": float(env.unwrapped.extras.get("Episode_Reward/distance_to_goal", 0.0)),
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }

        save_dir = os.path.join(log_root, traj)
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"eval_{traj}_{save_data['timestamp']}.npy")

        np.save(filename, save_data)
        print(f"[SUCCESS] Saved → {filename}")

        env.close()

    print("\n🎉 All 4 trajectories finished! Logs saved in logs/eval/")
    simulation_app.close()


if __name__ == "__main__":
    main()