import random

class StochasticEnvironment:
    """
    Simulates a messy, real-world environment where the failure rate
    isn't constant. This 'breaks' the math models but lets ML shine.
    """
    def __init__(self, base_lambda=5.0):
        self.base_lambda = base_lambda

    def get_noisy_lambda(self):
        # Introduce a 20% swing in failure probability
        # This simulates a system under fluctuating stress.
        noise = random.uniform(0.8, 1.2)
        return self.base_lambda * noise

    def get_stale_lambda(self):
        # Simulates 'Stale Data' - the model thinks lambda is 5,
        # but it's actually spiked to 8.
        return self.base_lambda + 3.0