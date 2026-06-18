import torch
from abc import ABC, abstractmethod

class TrajectoryBase(ABC):
    """All trajectory generators must implement this interface."""

    def __init__(self, cfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        # Per-env phase offset so envs are decorrelated
        self.phase_offset = torch.zeros(num_envs, device=device)

    @abstractmethod
    def reset(self, env_ids: torch.Tensor) -> None:
        """Called from _reset_idx. Resamples per-episode params (phase, fixed offsets, etc.)."""

    @abstractmethod
    def get_target_xy(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (target_x, target_y) in env-local frame, shape (num_envs,).
        t is current episode time in seconds, shape (num_envs,)."""

    def get_target_z(self) -> float:
        return self.cfg.trajectory_z_height
    
    def get_target_velocity_xy(self, t: torch.Tensor):
        """Returns (vx, vy) of the target in env-local frame, shape (num_envs,).
        Default: stationary target → zero velocity. Moving trajectories override."""
        zeros = torch.zeros(self.num_envs, device=self.device)
        return zeros, zeros