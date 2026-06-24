# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .quadcopter_env_cfg import QuadcopterEnvCfg
from asyncio import log
import gymnasium as gym
import torch
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils import math as math_utils
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
import isaaclab.sim as sim_utils

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


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # TD3 (and other off-policy algos) require finite action-space bounds.
        # PPO tolerates the default unbounded Box; off-policy clamps to low/high.
        import numpy as np
        from gymnasium import spaces
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=np.float32
        )

        from .trajectories import TRAJECTORY_REGISTRY
        from .curriculum.goal_distance import GoalDistanceCurriculum

        self._trajectory = TRAJECTORY_REGISTRY[cfg.trajectory_type](
            cfg, self.num_envs, self.device
        )
        self._curriculum = GoalDistanceCurriculum(cfg, self.num_envs, self.device)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # --- New: Add local up vector for tilt calculation ---
        #self._local_up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Action history buffer (for observation)
        if self.cfg.action_history_length > 0:
            self._action_history = torch.zeros(
                self.num_envs, self.cfg.action_history_length, 4,
                device=self.device,
            )

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

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Shift action history: drop oldest, append newest
        if self.cfg.action_history_length > 0:
            self._action_history = torch.roll(self._action_history, shifts=-1, dims=1)
            self._action_history[:, -1, :] = self._actions

        # --- SHAPE LOGIC SWITCH ---
        t = self.episode_length_buf * self.step_dt + self.cfg.trajectory_lookahead
        tx, ty = self._trajectory.get_target_xy(t)
        self._desired_pos_w[:, 0] = tx + self._terrain.env_origins[:, 0]
        self._desired_pos_w[:, 1] = ty + self._terrain.env_origins[:, 1]
        self._desired_pos_w[:, 2] = self._trajectory.get_target_z()

        # --- 2. ACTIONS & PHYSICS ---
        # Clamp actions to [-1, 1]
        #print(self._actions[0]) # to print the actions values [1,2,3,4]
           
        # Standard (Non-DR) version:
        total_thrust = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        # Apply to thrust buffer (Z-axis only for simple model)
        self._thrust[:, 0, 2] = total_thrust
        # Map action to Torques
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        # --- 1. GET GROUND TRUTH (Use _b suffix for clarity) ---
        lin_vel_b = self._robot.data.root_lin_vel_b.clone()
        ang_vel_b = self._robot.data.root_ang_vel_b.clone()
        pos_w = self._robot.data.root_pos_w.clone()
        quat_w = self._robot.data.root_quat_w.clone()

        # --- 2. APPLY NOISE ---
        if self.cfg.noise_lin_vel > 0.0 and not self.cfg.eval_mode:
            lin_vel_b += torch.randn_like(lin_vel_b) * self.cfg.noise_lin_vel
            
        if self.cfg.noise_ang_vel > 0.0 and not self.cfg.eval_mode:
            ang_vel_b += torch.randn_like(ang_vel_b) * self.cfg.noise_ang_vel

        if self.cfg.noise_quat > 0.0 and not self.cfg.eval_mode:
            quat_w += torch.randn_like(quat_w) * self.cfg.noise_quat
            quat_w = torch.nn.functional.normalize(quat_w, p=2, dim=-1)

        if self.cfg.noise_pos > 0.0 and not self.cfg.eval_mode:
            pos_w += torch.randn_like(pos_w) * self.cfg.noise_pos

        # --- 3. DERIVED STATES ---
        gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        proj_grav_b = math_utils.quat_rotate_inverse(quat_w, gravity_vec_w)
        #proj_grav_b = math_utils.quat_apply_inverse(quat_w, gravity_vec_w) # Result is in Body Frame

        desired_pos_b, _ = subtract_frame_transforms(
            pos_w, quat_w, self._desired_pos_w
        ) # Result is in Body Frame

        #action_history_flat = self._action_history.reshape(self.num_envs, -1)
        
        # --- 4. CONCATENATE ---
        if self.cfg.action_history_length > 0 and hasattr(self, '_action_history'):
            action_history_flat = self._action_history.reshape(self.num_envs, -1)
            obs = torch.cat([
                desired_pos_b,
                lin_vel_b,
                ang_vel_b,
                proj_grav_b,
                action_history_flat,
                quat_w,
            ], dim=-1)
        else:
            obs = torch.cat([
                desired_pos_b,
                lin_vel_b,
                ang_vel_b,
                proj_grav_b,
                quat_w,  # 
            ], dim=-1)

        #print range
        #print(f"lin_vel: {lin_vel_b.min()}, {lin_vel_b.max()}")
        # print(f"ang_vel_b: {ang_vel_b.min()}, {ang_vel_b.max()}")
        # print(f"quat_w: {quat_w.min()}, {quat_w.max()}")
        # print(f"proj_grav_b: {proj_grav_b.min()}, {proj_grav_b.max()}")
        # #print(f"ang_vel_b: {ang_vel_b.min()}, {ang_vel_b.max()}")

        return {"policy": obs}

    #  enable this _get_rewards without fuzzy logic
    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        #distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        #distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / self.cfg.tanh_scale)
        # ── Scheme A: annealed tanh scale ──
        if self.cfg.anneal_tanh and not self.cfg.eval_mode:
            # Geometric interpolation from init → target over tanh_anneal_steps.
            # common_step_counter is IsaacLab's built-in env-step counter.
            progress = min(self.common_step_counter / self.cfg.tanh_anneal_steps, 1.0)
            current_a = (self.cfg.tanh_scale_init
                        * (self.cfg.tanh_scale_target / self.cfg.tanh_scale_init) ** progress)
        else:
            current_a = (self.cfg.tanh_scale_target if self.cfg.anneal_tanh
                        else self.cfg.tanh_scale)

        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / current_a)

        # Log the schedule so it's visible in TensorBoard
        self.extras.setdefault("log", {})
        self.extras["log"]["Curriculum/tanh_scale"] = float(current_a)

        # Simple fixed tilt penalty (no fuzzy for now)
        local_up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        robot_up_vec_w = math_utils.quat_apply(self._robot.data.root_quat_w, local_up_vec)
        tilt_error = 1.0 - robot_up_vec_w[:, 2]

        tilt_penalty = tilt_error * self.cfg.tilt_penalty_scale

        # ── Scheme B: annealed penalty weights ──
        if self.cfg.anneal_penalties and not self.cfg.eval_mode:
            p = min(self.common_step_counter / self.cfg.penalty_anneal_steps, 1.0)
            lin_vel_scale = (self.cfg.lin_vel_scale_init
                            + (self.cfg.lin_vel_scale_target - self.cfg.lin_vel_scale_init) * p)
            ang_vel_scale = (self.cfg.ang_vel_scale_init
                            + (self.cfg.ang_vel_scale_target - self.cfg.ang_vel_scale_init) * p)
            self.extras.setdefault("log", {})
            self.extras["log"]["Curriculum/lin_vel_scale"] = float(lin_vel_scale)
        else:
            lin_vel_scale = self.cfg.lin_vel_reward_scale
            ang_vel_scale = self.cfg.ang_vel_reward_scale

        rewards = {
            "lin_vel": lin_vel * lin_vel_scale * self.step_dt,
            "ang_vel": ang_vel * ang_vel_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "tilt_penalty": tilt_penalty * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # === CRITICAL SAFETY ===
        reward = torch.clamp(reward, -10.0, 30.0)
        reward[self.reset_terminated] -= 15.0   # strong death penalty

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._log_eval_metrics()
        
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

        # --- 1. LOAD STATE DATA ---
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]

        # --- 2. APPLY POSITION MODIFIERS ---
        # Apply Terrain Origin
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        #  Increase Spawn Height (Add 2.0m to Z)
        # Apply spawn height
        if self.cfg.eval_mode:
            target_z = self._trajectory.get_target_z()
            if isinstance(target_z, torch.Tensor):
                default_root_state[:, 2] += target_z[env_ids]
            else:
                default_root_state[:, 2] += target_z

        # --- 3. APPLY ROTATION MODIFIERS ---
        if self.cfg.eval_mode:
            min_rad, max_rad = self.cfg.eval_initial_rotation_range
        else:
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

        # --- 4. APPLY VELOCITY RANDOMIZATION (The "Throw") ---
        if self.cfg.eval_mode:
            default_root_state[:, 7:13] = 0.0  # zero linear and angular velocity at spawn
        else:
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

        # --- 6. CURRICULUM (update max dist before sampling new targets) ---
        if not self.cfg.eval_mode:
            log = self._curriculum.step(self._trajectory)
        else:
            # In eval: fix target at 2m radius (representative but not too far)
            if hasattr(self._trajectory, 'current_max_dist'):
                self._trajectory.current_max_dist[:] = 2.0

        # --- 7. RESET TRAJECTORY AND COMPUTE INITIAL TARGET ---
        self._trajectory.reset(env_ids)
        t_zero = torch.zeros(self.num_envs, device=self.device)
        tx, ty = self._trajectory.get_target_xy(t_zero)
        self._desired_pos_w[env_ids, 0] = tx[env_ids] + self._terrain.env_origins[env_ids, 0]
        self._desired_pos_w[env_ids, 1] = ty[env_ids] + self._terrain.env_origins[env_ids, 1]
        target_z = self._trajectory.get_target_z()
        if isinstance(target_z, torch.Tensor):
            self._desired_pos_w[env_ids, 2] = target_z[env_ids]
        else:
            self._desired_pos_w[env_ids, 2] = target_z

        # Reset action history for newly-reset envs
        if self.cfg.action_history_length > 0:
            self._action_history[env_ids] = 0.0
        
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

    def _save_fuzzy_data(self):
        """Save collected fuzzy data to .npy file"""
        import os
        import numpy as np
        from datetime import datetime

        save_dir = "logs/fuzzy_data"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        data = {
            "tilt_error": np.array(self.tilt_history),
            "vel_error": np.array(self.vel_history),
            "adaptive_factor": np.array(self.fuzzy_history),
        }

        filename = os.path.join(save_dir, f"fuzzy_data_{timestamp}.npy")
        np.save(filename, data)

        print(f"[Fuzzy Data] Saved {len(self.tilt_history)} samples → {filename}")


    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    def _log_eval_metrics(self):
        """Per-step tracking metrics logged to extras for eval scraping."""
        dist = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )
        self.extras.setdefault("log", {})
        self.extras["log"]["Metrics/tracking_err_mean"] = dist.mean().item()
        self.extras["log"]["Metrics/tracking_err_p95"] = torch.quantile(dist, 0.95).item()
        self.extras["log"]["Metrics/success_rate"] = (dist < 0.3).float().mean().item()
