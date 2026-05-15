# fuzzy class implementation logic 1

import torch

class SimpleFuzzyTiltPenalty:
    """Simple fuzzy logic for adaptive tilt penalty."""

    def __init__(self, device):
        self.device = device
        self.debug_counter = 0
        self.strength = 0.3     # Start very soft (0.0 = no fuzzy, 1.0 = full fuzzy)

    def _tri_mf(self, x, a, b, c):
        """Triangle membership function."""
        return torch.clamp(torch.min((x - a) / (b - a), (c - x) / (c - b)), 0.0, 1.0)

    def compute(self, tilt_error: torch.Tensor, vel_error: torch.Tensor) -> torch.Tensor:
        """Return adaptive multiplier for tilt penalty (0.3 = lenient, 2.0 = strict)."""
        # Tilt error memberships
        tilt_low = self._tri_mf(tilt_error, 0.0, 0.0, 0.4)
        tilt_med = self._tri_mf(tilt_error, 0.2, 0.5, 0.8)
        tilt_high = self._tri_mf(tilt_error, 0.6, 1.0, 1.0)

        # Velocity error memberships
        vel_low = self._tri_mf(vel_error, 0.0, 0.0, 0.6)
        vel_med = self._tri_mf(vel_error, 0.3, 0.8, 1.3)
        vel_high = self._tri_mf(vel_error, 1.0, 1.5, 2.0)

        # Fuzzy rules (Mamdani)
        rule1 = torch.min(tilt_low, vel_low)      # Very lenient
        rule2 = torch.min(tilt_med, vel_med)
        rule3 = torch.min(tilt_high, vel_high)    # Strict

        # # Output fuzzy sets
        # out_low = rule1 * 0.5
        # out_med = rule2 * 0.9
        # out_high = rule3 * 1.4

        # # Defuzzify (centroid approximation)
        # numerator = out_low * 0.5 + out_med * 0.9 + out_high * 1.4
        # denominator = out_low + out_med + out_high + 1e-8
        # adaptive_factor = numerator / denominator

        numerator = rule1 * 0.4 + rule2 * 0.8 + rule3 * 1.3
        denominator = rule1 + rule2 + rule3 + 1e-8
        adaptive_factor = numerator / denominator

        # Apply strength (makes fuzzy very gentle at the beginning)
        adaptive_factor = 1.0 + self.strength * (adaptive_factor - 1.0)
        #adaptive_factor = torch.clamp(adaptive_factor, 0.4, 1.5)

        # === DEBUG PRINTS (every 200 steps) ===
        self.debug_counter += 1
        if self.debug_counter % 200 == 0:
            print(f"[Fuzzy Debug] tilt_error={tilt_error.mean().item():.3f} | "
                  f"vel_error={vel_error.mean().item():.3f} | "
                  f"adaptive_factor={adaptive_factor.mean().item():.3f}")

        return torch.clamp(adaptive_factor, 0.4, 1.5)


# Fuzzy Logic 2
class SimpleFuzzyTiltPenalty:
    """Fuzzy logic that prioritizes stabilization first (upright the drone)."""

    def __init__(self, device):
        self.device = device
        self.debug_counter = 0

    def _tri_mf(self, x, a, b, c):
        """Triangle membership function."""
        return torch.clamp(torch.min((x - a) / (b - a), (c - x) / (c - b)), 0.0, 1.0)

    def compute(self, tilt_error: torch.Tensor, vel_error: torch.Tensor) -> torch.Tensor:
        """
        Returns adaptive multiplier for tilt penalty.
        High tilt → strong penalty (stabilize first).
        Low tilt → normal penalty (allow tracking).
        """
        # Tilt error memberships
        tilt_very_high = self._tri_mf(tilt_error, 0.6, 0.9, 1.0)   # > 60° tilt
        tilt_high = self._tri_mf(tilt_error, 0.3, 0.6, 0.9)
        tilt_low = self._tri_mf(tilt_error, 0.0, 0.2, 0.4)

        # Velocity error (used only when tilt is low)
        vel_low = self._tri_mf(vel_error, 0.0, 0.0, 0.6)

        # Rules (Stabilization-first)
        rule_stabilize = tilt_very_high                     # Very strong penalty when highly tilted
        rule_moderate = torch.min(tilt_high, vel_low)       # Moderate when somewhat tilted
        rule_track = tilt_low                               # Normal when almost upright

        # Defuzzify (weighted)
        numerator = rule_stabilize * 2.2 + rule_moderate * 1.2 + rule_track * 0.6
        denominator = rule_stabilize + rule_moderate + rule_track + 1e-8

        adaptive_factor = numerator / denominator

        # Debug print every 200 steps
        self.debug_counter += 1
        if self.debug_counter % 200 == 0:
            print(f"[Fuzzy Debug] tilt={tilt_error.mean().item():.3f} | "
                  f"vel={vel_error.mean().item():.3f} | "
                  f"factor={adaptive_factor.mean().item():.3f} (higher = stronger penalty)")

        return torch.clamp(adaptive_factor, 0.6, 2.5)   # 0.6 = lenient, 2.5 = very strong penalty
    