import torch
from .base import TrajectoryBase

class HoverTrajectory(TrajectoryBase):
    """Fixed point at env origin. The simplest baseline."""
    def __init__(self, cfg, num_envs, device):
        super().__init__(cfg, num_envs, device)
        self.target = torch.zeros(num_envs, 2, device=device)

    def reset(self, env_ids):
        self.target[env_ids] = 0.0  # always (0, 0) at trajectory_z_height

    def get_target_xy(self, t):
        return self.target[:, 0], self.target[:, 1]