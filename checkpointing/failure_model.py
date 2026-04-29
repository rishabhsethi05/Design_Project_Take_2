import math
import random
from abc import ABC, abstractmethod


class FailureModel(ABC):
    @abstractmethod
    def should_fail(self, work_units: float) -> bool:
        pass


# ==========================================================
# POISSON FAILURE (STABLE ACCUMULATION)
# ==========================================================

class PoissonFailureModel(FailureModel):

    def __init__(self, failure_rate: float, seed: int = 42):
        self.lambda_rate = failure_rate
        self.rng = random.Random(seed)

    def should_fail(self, work_units: float) -> bool:
        """
        Calculates failure probability over a work interval.
        P(T < t) = 1 - e^(-lambda * t)
        """
        if self.lambda_rate <= 0:
            return False

        # --- CALIBRATION FIX ---
        # We removed the 5.0x multiplier.
        # This ensures the 'Real' failure rate matches the 'Analytical'
        # and 'ML' expectations exactly.
        prob = 1.0 - math.exp(-self.lambda_rate * work_units)

        return self.rng.random() < prob


# ==========================================================
# BURST FAILURE (STABLE VERSION)
# ==========================================================

class BurstFailureModel(FailureModel):

    def __init__(self, base_rate: float, burst_multiplier: float = 5.0, seed: int = 42):
        self.base_rate = base_rate
        self.burst_multiplier = burst_multiplier
        self.rng = random.Random(seed)
        self.in_burst = False

    def should_fail(self, work_units: float) -> bool:
        # State transition: 3% chance to toggle burst mode
        if self.rng.random() < 0.03:
            self.in_burst = not self.in_burst

        current_rate = self.base_rate * (self.burst_multiplier if self.in_burst else 1.0)

        # --- CALIBRATION FIX ---
        prob = 1.0 - math.exp(-current_rate * work_units)

        return self.rng.random() < prob