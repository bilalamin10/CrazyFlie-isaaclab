import torch
from .base import TrajectoryBase

class CircleTrajectory(TrajectoryBase):
    def reset(self, env_ids):
        self.phase_offset[env_ids] = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi

    def get_target_xy(self, t):
        theta = t * self.cfg.trajectory_speed + self.phase_offset
        r = self.cfg.trajectory_radius
        return r * torch.cos(theta), r * torch.sin(theta)