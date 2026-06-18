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
    
    def get_target_velocity_xy(self, t):
        theta = t * self.cfg.trajectory_speed + self.phase_offset
        s, c = torch.sin(theta), torch.cos(theta)
        omega = self.cfg.trajectory_speed
        r = self.cfg.trajectory_radius
        denom = 1 + s.square()

        # x = r*c/denom  →  dx/dθ = r*(-s*denom - c*2*s*c) / denom^2
        dx_dtheta = r * (-s * denom - c * (2 * s * c)) / denom.square()
        # y = r*s*c/denom = r*0.5*sin(2θ)/denom
        #   d/dθ[s*c] = c² - s²;   d/dθ[1/denom] = -2*s*c/denom²
        dy_dtheta = r * ((c.square() - s.square()) * denom - (s * c) * (2 * s * c)) / denom.square()

        return dx_dtheta * omega, dy_dtheta * omega