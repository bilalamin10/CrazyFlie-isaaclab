# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip

from .quadcopter_env import QuadcopterEnvWindow


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 2
    action_space = 4
    observation_space = 16
    #observation_space = 16 + 32 * 4   # base obs + action_history_length * action_space
    state_space = 0
    debug_vis = True

    # Setpoint shifting lookahead — seconds into the future the policy "sees" the target
    trajectory_lookahead: float = 0.0

    # Action history length
    action_history_length: int = 0

    # initial tilt in rads
    initial_rotation_range: tuple[float, float] = (-3.14, 3.14)

    # noise parameters
    noise_lin_vel: float = 0.05
    noise_ang_vel: float = 0.02
    noise_pos: float = 0.1
    noise_quat: float = 0.1

    # curriculum for static goals
    static_goal_curriculum: bool = True
    static_goal_min_dist: float = 0.5
    static_goal_max_dist: float = 4.0
    static_goal_curriculum_resets: int = 25_000

    # trajectory parameters
    trajectory_type: str = "static"   # base default; subclasses override
    trajectory_radius: float = 2.0
    trajectory_speed: float = 0.5
    trajectory_z_height: float = 1.5

    # 3D cuboid bounds for Hover/Static tasks
    cuboid_x_min: float = -3.0
    cuboid_x_max: float =  3.0
    cuboid_y_min: float = -3.0
    cuboid_y_max: float =  3.0
    cuboid_z_min: float =  0.5   # minimum hover height
    cuboid_z_max: float =  3.0   # maximum hover height

    # eval-specific switches
    eval_mode: bool = False
    eval_initial_rotation_range: tuple[float, float] = (-0.05, 0.05)

    # reward scale parameter (for Optuna tuning)
    reward_scale: float = 1.0

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True,
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 3.0
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.05
    distance_to_goal_reward_scale: float = 35.0
    tilt_penalty_scale: float = -0.5

    # Tanh scale parameter "a" in reward: 1 - tanh(distance / a)
    # Small a = narrow reward (only near target), large a = wide reward (signal from far)
    tanh_scale: float = 0.8   # current baseline

    # ── Scheme A: anneal tanh scale (Eschmann-style reward tightening) ──
    anneal_tanh: bool = False          # off by default — baseline unaffected
    tanh_scale_init: float = 2.5       # lenient/wide at start of training
    tanh_scale_target: float = 0.5     # tight/precise at end
    tanh_anneal_steps: int = 6000      # env.step() calls to reach target
                                    # (400 iters × 24 steps/iter = 9600 total,
                                    #  so 6000 ≈ first ~250 iterations)


    # ── Scheme B: anneal velocity/action penalty weights ──
    anneal_penalties: bool = False
    lin_vel_scale_init:  float = 0.0     # no velocity penalty at start
    lin_vel_scale_target: float = -0.20   # your current value at end
    ang_vel_scale_init:  float = 0.0
    ang_vel_scale_target: float = -0.1
    penalty_anneal_steps: int = 6000     # same schedule length as Scheme A

    action_rate_reward_scale: float = -0.01      # fixed value if not annealing
    action_rate_scale_init: float = 0.0          # annealing start (lenient)
    action_rate_scale_target: float = -0.02      # annealing end (strict) — tune

@configclass
class QuadcopterCircleCfg(QuadcopterEnvCfg):
    trajectory_type: str = "circle"
    trajectory_speed: float = 0.5  # mean linear speed = 1.0 m/s at radius 2.0

@configclass
class QuadcopterLemniscateCfg(QuadcopterEnvCfg):
    trajectory_type: str = "lemniscate"
    trajectory_speed: float = 0.526  # mean linear speed = 1.0 m/s at radius 2.0

@configclass
class QuadcopterLissajousCfg(QuadcopterEnvCfg):
    trajectory_type: str = "lissajous"
    trajectory_speed: float = 0.25  # mean linear speed = 1.0 m/s at radius 2.0, 3:2 ratio

@configclass
class QuadcopterHoverCfg(QuadcopterEnvCfg):
    trajectory_type: str = "hover"
    # trajectory_speed irrelevant — target is stationary

@configclass
class QuadcopterStaticCfg(QuadcopterEnvCfg):
    trajectory_type: str = "static"
    # trajectory_speed irrelevant — target is stationary
    static_goal_curriculum: bool = False

@configclass
class QuadcopterStaticAHCfg(QuadcopterStaticCfg):
    """Static task WITH action history (Eschmann-style)."""
    action_history_length: int = 32
    observation_space: int = 16 + 32 * 4   # 144 total obs = base obs + action_history_length * action_space

@configclass
class QuadcopterLandCfg(QuadcopterEnvCfg):
    trajectory_type: str = "land"
    static_goal_curriculum: bool = False
    # Target is at z=0 — reward needs to handle ground proximity
    # Spawn drone at random height, it must descend

@configclass
class QuadcopterHoverTD3Cfg(QuadcopterHoverCfg):
    """3D Hover for TD3 — Eschmann recipe: annealed penalties + action-rate."""
    anneal_penalties: bool = True
    # ramp velocity penalty from 0 → stronger-than-PPO to force settling
    lin_vel_scale_init: float = 0.0
    lin_vel_scale_target: float = -0.2     # stronger than the -0.05 default
    ang_vel_scale_init: float = 0.0
    ang_vel_scale_target: float = -0.1
    action_rate_scale_init: float = 0.0
    action_rate_scale_target: float = -0.02
    penalty_anneal_steps: int = 30000      # ~40% of 160k-step training