import torch
from .base import TrajectoryBase

class LemniscateTrajectory(TrajectoryBase):
    def reset(self, env_ids):
        self.phase_offset[env_ids] = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi

    def get_target_xy(self, t):
        theta = t * self.cfg.trajectory_speed + self.phase_offset
        s, c = torch.sin(theta), torch.cos(theta)
        denom = 1 + s.square()
        r = self.cfg.trajectory_radius
        return (r * c) / denom, (r * s * c) / denom