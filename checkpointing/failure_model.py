import math
import random
from abc import ABC, abstractmethod


class FailureModel(ABC):
    """
    Abstract base class for hardware failure simulations.
    Allows for different 'Intermittency Profiles'.
    """

    @abstractmethod
    def should_fail(self, work_units: float) -> bool:
        pass


class PoissonFailureModel(FailureModel):
    """
    Standard Reliability Engineering model.
    The probability of failure increases exponentially with time/work.
    P(fail) = 1 - e^(-λ * t)
    """

    def __init__(self, failure_rate: float, seed: int = 42):
        self.lambda_rate = failure_rate
        self.rng = random.Random(seed)

    def should_fail(self, work_units: float) -> bool:
        if self.lambda_rate <= 0:
            return False

        # The math: 1 - exp(-lambda * time)
        prob = 1.0 - math.exp(-self.lambda_rate * work_units)
        return self.rng.random() < prob


class BurstFailureModel(FailureModel):
    """
    Simulates 'Dirty Power' or unstable solar harvesting.
    Failures happen in clusters (bursts).
    """

    def __init__(self, base_rate: float, burst_multiplier: float = 5.0):
        self.base_rate = base_rate
        self.burst_multiplier = burst_multiplier
        self.in_burst = False

    def should_fail(self, work_units: float) -> bool:
        # Toggle burst mode randomly
        if random.random() < 0.05: self.in_burst = not self.in_burst

        current_rate = self.base_rate * (self.burst_multiplier if self.in_burst else 1.0)
        prob = 1.0 - math.exp(-current_rate * work_units)
        return random.random() < prob