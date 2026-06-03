import torch
from .base import TrajectoryBase

class LandTrajectory(TrajectoryBase):
    """
    Landing task: drone must navigate to a random XY point and land (z=0).
    Target is on the ground: (random_x, random_y, 0).
    """
    def __init__(self, cfg, num_envs, device):
        super().__init__(cfg, num_envs, device)
        self.target_x = torch.zeros(num_envs, device=device)
        self.target_y = torch.zeros(num_envs, device=device)

    def reset(self, env_ids):
        n = len(env_ids)
        self.target_x[env_ids] = (
            torch.rand(n, device=self.device)
            * (self.cfg.cuboid_x_max - self.cfg.cuboid_x_min)
            + self.cfg.cuboid_x_min
        )
        self.target_y[env_ids] = (
            torch.rand(n, device=self.device)
            * (self.cfg.cuboid_y_max - self.cfg.cuboid_y_min)
            + self.cfg.cuboid_y_min
        )

    def get_target_xy(self, t):
        return self.target_x, self.target_y

    def get_target_z(self):
        # Ground level — drone must descend to z=0
        return torch.zeros(
            self.target_x.shape[0], device=self.target_x.device
        )