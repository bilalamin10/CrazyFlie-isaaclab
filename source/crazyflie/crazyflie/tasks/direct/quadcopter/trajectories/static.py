import torch
from .base import TrajectoryBase

class StaticTrajectory(TrajectoryBase):
    """Random fixed target per episode. Supports a per-env max-distance curriculum."""
    def __init__(self, cfg, num_envs, device):
        super().__init__(cfg, num_envs, device)
        self.target = torch.zeros(num_envs, 2, device=device)
        self.current_max_dist = torch.full((num_envs,), cfg.static_goal_min_dist, device=device)

    def reset(self, env_ids):
        max_d = self.current_max_dist[env_ids]
        angle = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        dist = torch.rand(len(env_ids), device=self.device) * max_d
        self.target[env_ids, 0] = dist * torch.cos(angle)
        self.target[env_ids, 1] = dist * torch.sin(angle)

    def get_target_xy(self, t):
        return self.target[:, 0], self.target[:, 1]