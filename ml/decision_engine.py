import math


class DecisionEngine:
    """
    ML-Driven Checkpoint Decision Engine.
    Uses a Logistic Regression inspired model to predict the 
    optimality of a checkpoint at the current instruction line.
    """

    def __init__(self, structural_metrics: dict):
        self.metrics = structural_metrics

        # Optimized Weights (Learned parameters)
        self.weights = {
            "work_time_weight": 45.0,  # Sensitivity to accumulated time
            "failure_rate_weight": 120.0,  # Sensitivity to λ
            "complexity_weight": 8.5,  # Sensitivity to Cyclomatic Complexity
            "loop_weight": 15.0  # Penalty for being inside a loop
        }

        # Bias: This is the 'Threshold'. 
        # A higher negative bias means the model is more conservative (saves less).
        self.bias = -5.0

    def evaluate(self, work_since_last_checkpoint: float, failure_rate: float, current_line_cost: float):
        """
        Decision Function: Score = Bias + (W1*Time) + (W2*Lambda) + (W3*Complexity)
        Returns: (Boolean Decision, Probability Score)
        """

        # Feature 1: Accumulated risk over time
        time_risk = work_since_last_checkpoint * self.weights["work_time_weight"]

        # Feature 2: Environmental risk
        env_risk = failure_rate * self.weights["failure_rate_weight"]

        # Feature 3: Structural risk (from Static Analysis)
        # We use Cyclomatic Complexity and Loop Count as multipliers
        struct_risk = (self.metrics.get('cyclomatic_complexity', 1) * self.weights["complexity_weight"]) + \
                      (self.metrics.get('loop_count', 0) * self.weights["loop_weight"])

        # Normalize structural risk so it doesn't overwhelm the time factor
        normalized_struct = (struct_risk / 100.0)

        # Final Logit Calculation
        score = self.bias + time_risk + env_risk + (normalized_struct * time_risk)

        # Logistic Sigmoid Function to get a probability 0.0 to 1.0
        probability = 1 / (1 + math.exp(-max(min(score, 20), -20)))

        # Decision: If probability > 0.5 (or score > 0), checkpoint!
        return score > 0, probability