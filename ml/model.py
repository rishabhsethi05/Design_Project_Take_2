import math
from ml.model import CheckpointRiskModel  # Import your hardcoded weights


class DecisionEngine:
    def __init__(self, structural_metrics: dict):
        self.metrics = structural_metrics
        # Move the weights into the RiskModel structure
        self.bias = -3.5  # Adjusted for the new scaling

    def evaluate(self, work_since_last_checkpoint: float, failure_rate: float, current_line_cost: float):
        # 1. Get structural context
        loop_depth = self.metrics.get('loop_count', 0)
        complexity = self.metrics.get('cyclomatic_complexity', 1)

        # 2. Use your Model.py math
        # We pass a generic cp_cost estimate of 0.01 if not provided
        structural_risk = CheckpointRiskModel.calculate_risk(
            loop_depth=loop_depth,
            cyclomatic_complexity=complexity,
            failure_rate=failure_rate,
            cp_cost=0.01
        )

        # 3. Time Risk (The "Accumulated Pain")
        # We multiply structural risk by work done so we don't save 
        # at the very start of a loop.
        time_factor = work_since_last_checkpoint * 850.0

        # 4. Final Logistic Score
        score = self.bias + time_factor + structural_risk

        capped_score = max(min(score, 20), -20)
        probability = 1 / (1 + math.exp(-capped_score))

        return score > 0, probability