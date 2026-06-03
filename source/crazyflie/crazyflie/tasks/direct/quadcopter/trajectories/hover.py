import torch
from .base import TrajectoryBase

# class HoverTrajectory(TrajectoryBase):
#     """Fixed point at env origin. The simplest baseline."""
#     def __init__(self, cfg, num_envs, device):
#         super().__init__(cfg, num_envs, device)
#         self.target = torch.zeros(num_envs, 2, device=device)

#     def reset(self, env_ids):
#         self.target[env_ids] = 0.0  # always (0, 0) at trajectory_z_height

#     def get_target_xy(self, t):
#         return self.target[:, 0], self.target[:, 1]
    
class HoverTrajectory(TrajectoryBase):
    def __init__(self, cfg, num_envs, device):
        super().__init__(cfg, num_envs, device)
        self.target_x = torch.zeros(num_envs, device=device)
        self.target_y = torch.zeros(num_envs, device=device)
        self.target_z = torch.zeros(num_envs, device=device)

    def reset(self, env_ids):
        n = len(env_ids)
        # Sample random 3D point in cuboid
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
        self.target_z[env_ids] = (
            torch.rand(n, device=self.device)
            * (self.cfg.cuboid_z_max - self.cfg.cuboid_z_min)
            + self.cfg.cuboid_z_min
        )

    def get_target_xy(self, t):
        return self.target_x, self.target_y

    def get_target_z(self):
        return self.target_z   # per-env tensor, not scalar