# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# Correct imports for Isaac Lab v2.2.0
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg


@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for Crazyflie Quadcopter."""

    class_name: str = "OnPolicyRunner"

    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    save_videos: bool = True
    video_interval: int = 50
    experiment_name = "quadcopter_direct"

    # Policy Network Configuration
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

    # PPO Algorithm Configuration
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,          # small entropy for better exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )