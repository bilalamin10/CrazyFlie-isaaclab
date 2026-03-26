# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils import math as math_utils

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


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
    initial_rotation_range: tuple[float, float] = (-3.14, 3.14) # approx +/- 180 degrees
    # noise parameters
    noise_lin_vel: float = 0.05  # e.g., +/- 0.05 m/s (Optical Flow/GPS noise)
    noise_ang_vel: float = 0.02  # e.g., +/- 0.02 rad/s (Gyroscope noise)
    noise_pos: float = 0.1      # e.g., +/- 1 cm (Mocap/GPS noise)
    noise_quat: float = 0.2     # Orientation noise (IMU filter error)

    # New: curriculum for static goals
    static_goal_curriculum: bool = True
    static_goal_min_dist: float = 0.5     # start easy
    static_goal_max_dist: float = 4.0     # target difficulty
    #static_goal_curriculum_timesteps: int = 20_000_000  # when to reach max

    # Trajectory Settings
    # Options: "lemniscate" (Figure-8), "circle", "lissajous" (Random knots)
    #trajectory_type: str = "lemniscate"
    #trajectory_type: str = "static"

    # --- Trajectory Parameters ---
    trajectory_type: str = "static" # Options: "static", "circle"
    trajectory_radius: float = 2  # Radius of the circle in meters
    trajectory_speed: float = 0.5   # Speed in radians/second (approx 0.75 m/s)
    trajectory_z_height: float = 1.5 # Height to fly at

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
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 2.0
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.2     
    ang_vel_reward_scale = -0.05
    distance_to_goal_reward_scale = 25.0
    
    # Penalizes the drone for being tilted (not horizontal).
    # More negative = stronger penalty.
    tilt_penalty_scale: float = -1.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._traj_phase_offset = torch.zeros(self.num_envs, device=self.device)
        # --- New: Add local up vector for tilt calculation ---
        #self._local_up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        #Goal position offset
        self.fixed_goal_offset_x = torch.zeros(self.num_envs, device=self.device)
        self.fixed_goal_offset_y = torch.zeros(self.num_envs, device=self.device)

        self.current_max_goal_dist = torch.full((self.num_envs,), self.cfg.static_goal_min_dist, device=self.device)
        #self.global_progress = 0.0  # we'll update this from runner or approximate
        
        #self.curriculum_iteration = 0
        self.curriculum_counter = 0
        #self.max_curriculum_steps = 15_000_000  # ← tune this! (e.g. when you want full difficulty)
        #self.curriculum_update_freq = 50               # Update every N physics steps (to reduce overhead)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "tilt_penalty",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    #   Circle
    # def _pre_physics_step(self, actions: torch.Tensor):
    #     # --- UPDATE TRAJECTORY TARGET ---
    #     # Calculate current time (t) for each env
    #     # episode_length_buf gives steps. Multiply by dt to get seconds.
    #     current_time = self.episode_length_buf * self.step_dt
        
    #     # Calculate the angle: theta = speed * time + offset
    #     speed = self.cfg.trajectory_speed
    #     radius = self.cfg.trajectory_radius
        
    #     # Vectorized calculation for all 4096 envs
    #     theta = (current_time * speed) + self._traj_phase_offset
        
    #     # 1. Calculate Circle Offsets
    #     target_x = radius * torch.cos(theta)
    #     target_y = radius * torch.sin(theta)
    #     target_z = self.cfg.trajectory_z_height

    #     # 2. Apply to Global Coordinates (Add Environment Origins!)
    #     # Note: We update ALL indices every step.
    #     self._desired_pos_w[:, 0] = target_x + self._terrain.env_origins[:, 0]
    #     self._desired_pos_w[:, 1] = target_y + self._terrain.env_origins[:, 1]
    #     self._desired_pos_w[:, 2] = target_z


    #     self._actions = actions.clone().clamp(-1.0, 1.0)
    #     self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
    #     self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]


    # Lemniscate
    # def _pre_physics_step(self, actions: torch.Tensor):
    #     # --- 1. UPDATE TRAJECTORY TARGET (Lemniscate) ---
    #     # Calculate current time (t) for each env
    #     # We use the episode length buffer to get the time elapsed in simulation
    #     current_time = self.episode_length_buf * self.step_dt
        
    #     speed = self.cfg.trajectory_speed
    #     radius = self.cfg.trajectory_radius
        
    #     # Calculate the phase angle theta based on time and random offset
    #     # theta shape: (num_envs,)
    #     theta = (current_time * speed) + self._traj_phase_offset
        
    #     # --- Lemniscate Math ---
    #     # parametric equations:
    #     # x = (r * cos(theta)) / (1 + sin^2(theta))
    #     # y = (r * sin(theta) * cos(theta)) / (1 + sin^2(theta))
        
    #     sin_theta = torch.sin(theta)
    #     cos_theta = torch.cos(theta)
    #     # Denominator: 1 + sin^2(theta)
    #     denominator = 1 + torch.square(sin_theta)
        
    #     # Calculate Offsets
    #     target_x = (radius * cos_theta) / denominator
    #     target_y = (radius * sin_theta * cos_theta) / denominator
    #     target_z = self.cfg.trajectory_z_height

    #     # Apply to Global Coordinates (Add Environment Origins)
    #     self._desired_pos_w[:, 0] = target_x + self._terrain.env_origins[:, 0]
    #     self._desired_pos_w[:, 1] = target_y + self._terrain.env_origins[:, 1]
    #     self._desired_pos_w[:, 2] = target_z

    #     # --- 2. ACTIONS & PHYSICS ---
    #     # Clamp actions to [-1, 1]
    #     self._actions = actions.clone().clamp(-1.0, 1.0)
        
    #     # Convert Network Output to Physics Forces
    #     # Map [-1, 1] action to [0, Max Thrust]
    #     # Note: We use the randomized thrust scale if you implemented DR there, 
    #     # otherwise use self.cfg.thrust_to_weight * self._robot_weight
        
    #     # Standard (Non-DR) version:
    #     total_thrust = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        
    #     # Apply to thrust buffer (Z-axis only for simple model)
    #     self._thrust[:, 0, 2] = total_thrust
        
    #     # Map action to Torques
    #     self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _pre_physics_step(self, actions: torch.Tensor):
        # --- UPDATE TRAJECTORY TARGET ---
        current_time = self.episode_length_buf * self.step_dt
        speed = self.cfg.trajectory_speed
        radius = self.cfg.trajectory_radius
        
        # Calculate phase angle
        theta = (current_time * speed) + self._traj_phase_offset
        
        # Initialize target variables
        target_x = torch.zeros_like(theta)
        target_y = torch.zeros_like(theta)

        # --- SHAPE LOGIC SWITCH ---
        if self.cfg.trajectory_type == "circle":
            # Simple Circle
            target_x = radius * torch.cos(theta)
            target_y = radius * torch.sin(theta)

        elif self.cfg.trajectory_type == "lemniscate":
            # Figure-8 (Your current training shape)
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            denom = 1 + torch.square(sin_theta)
            target_x = (radius * cos_theta) / denom
            target_y = (radius * sin_theta * cos_theta) / denom

        elif self.cfg.trajectory_type == "lissajous":
            # Complex Knot / Random-looking path
            # x = A sin(a*t + d), y = B sin(b*t)
            target_x = radius * torch.sin(3 * theta)
            target_y = radius * torch.sin(2 * theta)

        elif self.cfg.trajectory_type == "static":
            target_x = self.fixed_goal_offset_x
            target_y = self.fixed_goal_offset_y

        # --- Apply to Global Coordinates ---
        self._desired_pos_w[:, 0] = target_x + self._terrain.env_origins[:, 0]
        self._desired_pos_w[:, 1] = target_y + self._terrain.env_origins[:, 1]
        self._desired_pos_w[:, 2] = self.cfg.trajectory_z_height

        # --- 2. ACTIONS & PHYSICS ---
        # Clamp actions to [-1, 1]
        self._actions = actions.clone().clamp(-1.0, 1.0)
        #print(self._actions[0]) # to print the actions values [1,2,3,4]
        
        # Convert Network Output to Physics Forces
        # Map [-1, 1] action to [0, Max Thrust]
        # Note: We use the randomized thrust scale if you implemented DR there, 
        # otherwise use self.cfg.thrust_to_weight * self._robot_weight
        
        # Standard (Non-DR) version:
        total_thrust = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        
        # Apply to thrust buffer (Z-axis only for simple model)
        self._thrust[:, 0, 2] = total_thrust
        
        # Map action to Torques
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        self._maybe_update_curriculum()



    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # def _get_observations(self) -> dict:
    #     desired_pos_b, _ = subtract_frame_transforms(
    #         self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
    #     )
    #     obs = torch.cat(
    #         [
    #             self._robot.data.root_lin_vel_b,        # 3-dim
    #             self._robot.data.root_ang_vel_b,        # 3-dim
    #             self._robot.data.root_quat_w,           # 4-dim
    #             self._robot.data.projected_gravity_b,   #3-dim
    #             desired_pos_b,                          # 3-dim
    #         ],
    #         dim=-1,
    #     )
    #     observations = {"policy": obs}
    #     return observations

    def _get_observations(self) -> dict:
        # --- 1. GET GROUND TRUTH (Use _b suffix for clarity) ---
        lin_vel_b = self._robot.data.root_lin_vel_b.clone()
        ang_vel_b = self._robot.data.root_ang_vel_b.clone()
        pos_w = self._robot.data.root_pos_w.clone()
        quat_w = self._robot.data.root_quat_w.clone()

        # --- 2. APPLY NOISE ---
        if self.cfg.noise_lin_vel > 0.0:
            lin_vel_b += torch.randn_like(lin_vel_b) * self.cfg.noise_lin_vel
            
        if self.cfg.noise_ang_vel > 0.0:
            ang_vel_b += torch.randn_like(ang_vel_b) * self.cfg.noise_ang_vel

        if self.cfg.noise_quat > 0.0:
            quat_w += torch.randn_like(quat_w) * self.cfg.noise_quat
            quat_w = torch.nn.functional.normalize(quat_w, p=2, dim=-1)

        if self.cfg.noise_pos > 0.0:
            pos_w += torch.randn_like(pos_w) * self.cfg.noise_pos

        # --- 3. DERIVED STATES ---
        gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        proj_grav_b = math_utils.quat_apply_inverse(quat_w, gravity_vec_w) # Result is in Body Frame

        desired_pos_b, _ = subtract_frame_transforms(
            pos_w, quat_w, self._desired_pos_w
        ) # Result is in Body Frame

        # --- 4. CONCATENATE ---
        obs = torch.cat(
            [
                lin_vel_b,     # Body Frame Linear Vel
                ang_vel_b,     # Body Frame Angular Vel
                quat_w,        # World Frame Orientation
                proj_grav_b,   # Body Frame Gravity
                desired_pos_b, # Body Frame Relative Pos
            ],
            dim=-1,
        )

        #print range
        #print(f"lin_vel: {lin_vel_b.min()}, {lin_vel_b.max()}")
        # print(f"ang_vel_b: {ang_vel_b.min()}, {ang_vel_b.max()}")
        # print(f"quat_w: {quat_w.min()}, {quat_w.max()}")
        # print(f"proj_grav_b: {proj_grav_b.min()}, {proj_grav_b.max()}")
        # #print(f"ang_vel_b: {ang_vel_b.min()}, {ang_vel_b.max()}")

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        local_up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        # --- Calculate New Tilt Penalty ---
        # Get the robot's "up" vector in the world frame
        robot_up_vec_w = math_utils.quat_apply(self._robot.data.root_quat_w, local_up_vec)
        
        # Calculate the dot product with the world's "up" vector (Z-axis)
        # 1.0 = perfectly upright, 0.0 = 90-degree tilt
        tilt_dot_product = robot_up_vec_w[:, 2] 

        # The error is the deviation from 1.0
        # 0.0 = no error, 1.0 = 90-degree tilt error
        tilt_error = 1.0 - tilt_dot_product

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "tilt_penalty": tilt_error * self.cfg.tilt_penalty_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.1, 
            self._robot.data.root_pos_w[:, 2] > 6.0  # <--- THIS VALUE increased because of toss and 2 meter hight
        )
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # --- 1. LOAD STATE DATA (Do this FIRST) ---
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]

        # --- 2. APPLY POSITION MODIFIERS ---
        # Apply Terrain Origin
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # FIX 1: Increase Spawn Height (Add 2.0m to Z)
        default_root_state[:, 2] += 2.0

        # --- 3. APPLY ROTATION MODIFIERS ---
        min_rad, max_rad = self.cfg.initial_rotation_range
        
        # Sample random roll, pitch, yaw
        roll = torch.rand(len(env_ids), device=self.device) * (max_rad - min_rad) + min_rad
        pitch = torch.rand(len(env_ids), device=self.device) * (max_rad - min_rad) + min_rad
        yaw = torch.zeros(len(env_ids), device=self.device)
        
        # Convert to quaternion
        random_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        
        # Apply rotation to default orientation
        default_quat = default_root_state[:, 3:7]
        new_quat = math_utils.quat_mul(random_quat, default_quat)
        default_root_state[:, 3:7] = new_quat

        # # --- 4. APPLY VELOCITY MODIFIERS (The Toss) ---
        # # FIX 2: Add Upward Velocity
        # toss_vel = 1.0 + torch.rand(len(env_ids), device=self.device) * 2.0
        
        # # Clear default linear velocities (indices 7,8,9)
        # default_root_state[:, 7:10] = 0.0 
        # # Set upward velocity (index 9 is Z)
        # default_root_state[:, 9] = toss_vel 

        # --- 4. APPLY VELOCITY RANDOMIZATION (The "Throw") ---
        
        # Randomize X and Y (Horizontal Velocity)
        # Range: -1.0 to +1.0 m/s (Simulates a slightly messy throw)
        default_root_state[:, 7] = torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0 
        default_root_state[:, 8] = torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0 
        
        # Randomize Z (Vertical Velocity)
        # Range: +2.0 to +4.0 m/s (Strong upward toss to fight gravity)
        # CRITICAL: This must be positive and strong to allow time for the 180-degree flip.
        default_root_state[:, 9] = 2.0 + torch.rand(len(env_ids), device=self.device) * 2.0

        # Randomize Angular Velocity (Optional but recommended)
        # Range: -1.0 to +1.0 rad/s (Simulates a slight spin on release)
        default_root_state[:, 10:13] = torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0
        # --- 5. WRITE TO SIMULATOR (Do this ONCE at the end) ---
        # Write Root State (Pose + Velocity)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        
        # Write Joint State
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # # --- 6. SAMPLE NEW COMMANDS ---
        # self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        # self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        # self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # --- 6. SAMPLE TRAJECTORY PARAMETERS ---
        # Create a random "Phase" (starting angle) for the circle (0 to 2pi)
        # We store this in a new buffer if you want persistence, 
        # but for now we can calculate position directly based on episode time.
        
        # For a moving target, we don't set a fixed position here.
        # We just initialize the logic. The position will be updated in _pre_physics_step.
        
        # However, to prevent a huge jump at step 0, let's calculate the t=0 position now.
        
        # Define a random starting phase offset for each environment
        # (Store this as a class variable in __init__ first: self._traj_phase_offset)
        self._traj_phase_offset[env_ids] = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi

        # Calculate initial target position
        radius = self.cfg.trajectory_radius
        theta = self._traj_phase_offset[env_ids] # t=0
        
        # Initialize target variables
        target_x = torch.zeros_like(theta)
        target_y = torch.zeros_like(theta)
        
        # --- SHAPE LOGIC SWITCH (Consistency with Pre-Physics) ---
        if self.cfg.trajectory_type == "circle":
            target_x = radius * torch.cos(theta)
            target_y = radius * torch.sin(theta)
            
        elif self.cfg.trajectory_type == "lemniscate":
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            denom = 1 + torch.square(sin_theta)
            target_x = (radius * cos_theta) / denom
            target_y = (radius * sin_theta * cos_theta) / denom
            
        elif self.cfg.trajectory_type == "lissajous":
            target_x = radius * torch.sin(3 * theta)
            target_y = radius * torch.sin(2 * theta)
            
        elif self.cfg.trajectory_type == "static":
            # # Static target at center (or random offset if you prefer)
            # target_x[:] = 0.0
            # target_y[:] = 0.0
            # Random goal in XY (e.g., disk of radius 4m)

            # angle = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
            # dist = torch.rand(len(env_ids), device=self.device) * 2.0
            # target_x = dist * torch.cos(angle)
            # target_y = dist * torch.sin(angle)
            # self.fixed_goal_offset_x[env_ids] = target_x
            # self.fixed_goal_offset_y[env_ids] = target_y

            # Use current curriculum value per env
            max_d = self.current_max_goal_dist[env_ids]

            angle = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
            dist = torch.rand(len(env_ids), device=self.device) * max_d   # 0 to current_max

            target_x = dist * torch.cos(angle)
            target_y = dist * torch.sin(angle)

            self.fixed_goal_offset_x[env_ids] = target_x
            self.fixed_goal_offset_y[env_ids] = target_y

        # --- Apply to Global Coordinates ---
        self._desired_pos_w[env_ids, 0] = target_x + self._terrain.env_origins[env_ids, 0]
        self._desired_pos_w[env_ids, 1] = target_y + self._terrain.env_origins[env_ids, 1]
        self._desired_pos_w[env_ids, 2] = self.cfg.trajectory_z_height
        
        # # X = Radius * cos(angle) + Env_Origin_X
        # self._desired_pos_w[env_ids, 0] = radius * torch.cos(angle) + self._terrain.env_origins[env_ids, 0]
        # # Y = Radius * sin(angle) + Env_Origin_Y
        # self._desired_pos_w[env_ids, 1] = radius * torch.sin(angle) + self._terrain.env_origins[env_ids, 1]
        # # Z = Constant Height
        # self._desired_pos_w[env_ids, 2] = self.cfg.trajectory_z_height

    # def update_curriculum(self, progress: float):
    #     """progress: 0..1 (total timesteps / total expected timesteps)"""
    #     if not self.cfg.static_goal_curriculum:
    #         return
    #     factor = min(1.0, progress)
    #     target_dist = self.cfg.static_goal_min_dist + factor * (
    #         self.cfg.static_goal_max_dist - self.cfg.static_goal_min_dist
    #     )
    #     self.current_max_goal_dist[:] = target_dist

    # # New method
    # def update_curriculum_from_iteration(self, current_iteration: int):
    #     if not self.cfg.static_goal_curriculum:
    #         return
    #     progress = min(1.0, current_iteration / self.max_curriculum_iterations)
    #     target_dist = (
    #         self.cfg.static_goal_min_dist +
    #         progress * (self.cfg.static_goal_max_dist - self.cfg.static_goal_min_dist)
    #     )
    #     self.current_max_goal_dist[:] = target_dist

    #     # Optional: log current value so you see it in console/TensorBoard extras
    #     self.extras["Metrics/curr_max_goal_dist"] = target_dist

    def _maybe_update_curriculum(self):
    #"""Fast curriculum ramp - tuned for short test runs and real training."""
        self.curriculum_counter += 1

    # Update every 64 steps (faster response)
        if self.curriculum_counter % 64 != 0:
            return

        if not self.cfg.static_goal_curriculum:
            return

        # Rough estimation: ~120-150 steps per iteration
        # So 500 iterations ≈ 60,000 - 75,000 steps
        estimated_iter = self.curriculum_counter // 128

        # === MAIN CHANGE: Reach 4.0m around iteration 480-500 ===
        progress = min(1.0, estimated_iter / 200.0)

        target_dist = (
            self.cfg.static_goal_min_dist +
            progress * (self.cfg.static_goal_max_dist - self.cfg.static_goal_min_dist)
        )

        self.current_max_goal_dist[:] = target_dist

        # Print progress more frequently so you can see it clearly
        if estimated_iter % 50 == 0 and estimated_iter > 0:
            print(
                f"[Curriculum] Iter ~{estimated_iter:3d} / 500 | "
                f"progress={progress:.3f} | "
                f"max_goal_dist = {self.current_max_goal_dist.mean().item():.3f} m"
            )

        # Log to TensorBoard
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["Metrics/curr_max_goal_dist"] = target_dist

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
