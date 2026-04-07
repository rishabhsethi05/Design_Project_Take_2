class CheckpointDecisionModel:
    def __init__(self):
        self.weights = {
            "work_since_last_checkpoint": 0.55, # Increased: progress is key
            "failure_rate": 65.0,               # Increased: lambda is a massive risk
            "avg_block_cost": 0.15,
            "execution_variance": 0.1,
            "loop_density": 8.5,                # High: Loops are recompute traps
            "branch_density": 4.0,
            "cyclomatic_complexity": 0.05
        }
        self.bias = -15.0 # Higher bias requires more "proof" to save energy

    def predict_score(self, features):
        score = self.bias
        for key, value in features.items():
            if key in self.weights:
                score += self.weights[key] * value
        return score

    def should_checkpoint(self, features):
        return self.predict_score(features) > 0