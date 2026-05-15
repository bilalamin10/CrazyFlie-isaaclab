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
    state_space = 0
    debug_vis = True

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

    # eval-specific switches
    eval_mode: bool = False
    eval_initial_rotation_range: tuple[float, float] = (-0.05, 0.05)

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
    lin_vel_reward_scale: float = -0.2
    ang_vel_reward_scale: float = -0.05
    distance_to_goal_reward_scale: float = 35.0
    tilt_penalty_scale: float = -0.5


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