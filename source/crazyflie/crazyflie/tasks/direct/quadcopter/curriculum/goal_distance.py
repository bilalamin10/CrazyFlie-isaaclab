# curriculum/goal_distance.py
import torch


class GoalDistanceCurriculum:
    """Linear curriculum on max target distance.

    Advances by a fixed step every time it's called (which should be once per
    reset call from _reset_idx, NOT per physics step). The new max is written
    to the trajectory immediately, so freshly-reset envs see it on the same tick.
    """

    def __init__(self, cfg, num_envs, device):
        self.cfg = cfg
        self.device = device
        self.current_target = float(cfg.static_goal_min_dist)
        self.reset_counter = 0

    def step(self, trajectory) -> dict:
        """Call this from _reset_idx, BEFORE trajectory.reset(env_ids)."""
        if not self.cfg.static_goal_curriculum:
            return {}

        self.reset_counter += 1

        # Ramp linearly over `static_goal_curriculum_resets` resets.
        n_ramp = getattr(self.cfg, "static_goal_curriculum_resets", 25_000)
        progress = min(1.0, self.reset_counter / n_ramp)
        self.current_target = self.cfg.static_goal_min_dist + progress * (
            self.cfg.static_goal_max_dist - self.cfg.static_goal_min_dist
        )

        # Write into the trajectory so reset() sees the new max on this same tick.
        if hasattr(trajectory, "current_max_dist"):
            trajectory.current_max_dist[:] = self.current_target

        return {"Metrics/curr_max_goal_dist": self.current_target}