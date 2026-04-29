import math


class DecisionEngine:
    def __init__(self, structural_metrics: dict):
        self.metrics = structural_metrics

        # RECALIBRATION:
        # We drastically lower work_time_weight so time alone doesn't trigger a CP.
        # We set complexity_weight so only the loops can overcome the bias.
        self.weights = {
            "work_time_weight": 50.0,  # Dropped from 450 (Mutes the clock)
            "failure_rate_weight": 2.0,  # Dropped from 5
            "complexity_weight": 15.0,  # Increased significantly
            "loop_weight": 25.0  # Increased significantly
        }

        # High negative bias means the structural risk MUST be present to say 'Yes'
        self.bias = -20.0

    def evaluate(self, work_since_last_checkpoint: float, failure_rate: float, current_line_cost: float):
        # 1. Base Risks (Now very small)
        time_risk = work_since_last_checkpoint * self.weights["work_time_weight"]
        env_risk = failure_rate * self.weights["failure_rate_weight"]

        # 2. Structural Hotspots
        complexity = self.metrics.get('cyclomatic_complexity', 1)
        loops = self.metrics.get('loop_count', 0)

        # 3. Targeted Logic:
        # Only provide structural risk if we are in a Loop.
        # This is what differentiates ML from a standard timer.
        struct_risk = 0
        if loops > 0:
            struct_risk = (complexity * self.weights["complexity_weight"]) + \
                          (loops * self.weights["loop_weight"])

        # 4. Score Calculation
        score = self.bias + time_risk + env_risk + struct_risk

        capped_score = max(min(score, 20), -20)
        probability = 1 / (1 + math.exp(-capped_score))

        # We require a higher hurdle to stand out
        decision = score > 5.0

        return decision, probability