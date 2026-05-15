import torch
from .base import TrajectoryBase


class LissajousTrajectory(TrajectoryBase):
    """Lissajous curve: x = A*sin(a*t + delta), y = B*sin(b*t).

    The ratio a:b determines the shape. Defaults to 3:2 with a quarter-phase
    offset, which produces the classic 3-lobe-by-2-lobe Lissajous figure.
    """

    def reset(self, env_ids):
        # Random phase per episode so envs don't all start at the same point
        self.phase_offset[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        )

    def get_target_xy(self, t):
        # Frequency ratio a:b — pull from cfg if present, else default to 3:2
        a = getattr(self.cfg, "lissajous_freq_a", 3.0)
        b = getattr(self.cfg, "lissajous_freq_b", 2.0)
        delta = getattr(self.cfg, "lissajous_phase_delta", torch.pi / 2)

        omega = t * self.cfg.trajectory_speed + self.phase_offset
        r = self.cfg.trajectory_radius
        x = r * torch.sin(a * omega + delta)
        y = r * torch.sin(b * omega)
        return x, y