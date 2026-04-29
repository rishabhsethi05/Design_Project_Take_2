import numpy as np

class FailureRegressionModel:
    """
    Regression-based checkpoint decision model.

    Predicts expected efficiency based on runtime state.
    Decision:
    - If predicted efficiency is low → checkpoint
    """

    def __init__(self):
        # Learned from dataset
        self.intercept = 36.16198227575802

        self.coeff_failure_rate = 0.004703650648479176
        self.coeff_checkpoint_cost = 639.1672867333219
        self.coeff_checkpoint_count = -0.9332017395116281

        # loop_count and cyclomatic_complexity were 0 → ignored

        # Decision threshold (you can tune this)
        self.threshold = 25.0

    def predict_efficiency(self, failure_rate, checkpoint_cost, checkpoint_count):
        prediction = (
            self.intercept
            + self.coeff_failure_rate * failure_rate
            + self.coeff_checkpoint_cost * checkpoint_cost
            + self.coeff_checkpoint_count * checkpoint_count
        )
        return prediction

    def should_checkpoint(self, failure_rate, checkpoint_cost, checkpoint_count):
        predicted_eff = self.predict_efficiency(
            failure_rate,
            checkpoint_cost,
            checkpoint_count
        )

        # If efficiency drops → checkpoint
        return predicted_eff < self.threshold