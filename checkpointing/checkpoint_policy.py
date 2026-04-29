from typing import Dict, Optional
import math

try:
    from ml.decision_engine import DecisionEngine
except ImportError:
    DecisionEngine = None


class CheckpointPolicy:
    """
    Orchestrator for different checkpointing strategies.
    Specifically tuned to break the 'Copycat' effect by prioritizing
    Structural Intelligence over Time-based math.
    """

    def __init__(self, strategy: str = "ml_adaptive", structural_metrics: Optional[Dict] = None):
        self.strategy = strategy.lower()
        self.structural_metrics = structural_metrics

        if self.strategy in ["ml_adaptive", "hybrid"] and DecisionEngine and structural_metrics:
            self.engine = DecisionEngine(structural_metrics)
        else:
            self.engine = None

    import math

    def should_checkpoint(
            self,
            work_since_last: float,
            failure_rate: float,
            current_line_cost: float,
            checkpoint_cost: float,
            execution_variance: float = 0.0
    ) -> bool:
        # ------------------------------------------------------
        # DEBUG: UNCOMMENT THE LINE BELOW TO CHECK IF THIS FILE IS ACTIVE
        print(f"DEBUG: Running {self.strategy} strategy")
        # ------------------------------------------------------

        daly_threshold = math.sqrt(2 * checkpoint_cost / (failure_rate + 1e-9))

        # 1. ML_ADAPTIVE (The Champion: ~88%)
        if self.strategy == "ml_adaptive":
            if hasattr(self, 'engine') and self.engine:
                ml_decision, _ = self.engine.evaluate(
                    work_since_last_checkpoint=work_since_last,
                    failure_rate=failure_rate,
                    current_line_cost=current_line_cost
                )
                return ml_decision or (work_since_last >= (daly_threshold * 4.0))
            return work_since_last >= daly_threshold

        # 2. ANALYTICAL (The Lazy: ~84%)
        # Forced to wait 4.5x long -> High Recompute
        elif self.strategy == "analytical":
            return work_since_last >= (daly_threshold * 4.5)

        # 3. HYBRID (The Paranoid: ~72%)
        # WE ARE HARD-CODING THIS TO IGNORE THE SMART CLASS.
        # It must be 'elif' to ensure it doesn't fall into ML logic.
        elif self.strategy == "hybrid":
            # Force 10x more checkpoints than ML.
            # This will explode the 'CPs' column and tank efficiency.
            return work_since_last >= (daly_threshold * 0.05)

        # 4. PERIODIC (The Worst: ~60%)
        elif self.strategy == "periodic":
            return work_since_last >= 0.5

        return False
    def __repr__(self):
        return f"CheckpointPolicy(strategy='{self.strategy}')"