from typing import Dict, Optional
import math

try:
    from ml.decision_engine import DecisionEngine
except ImportError:
    DecisionEngine = None


class CheckpointPolicy:
    """
    Orchestrator for different checkpointing strategies.
    Now updated to support line-level micro-decisions.
    """

    def __init__(self, strategy: str = "ml_adaptive", structural_metrics: Optional[Dict] = None):
        """
        Strategies:
        - 'none': No checkpoints.
        - 'periodic': Checkpoint every X seconds of execution time.
        - 'analytical': Young/Daly model (Optimal for steady-state failure).
        - 'ml_adaptive': Instruction-aware ML model (The Research Core).
        """
        self.strategy = strategy
        self.structural_metrics = structural_metrics

        # Initialize the ML Brain if the strategy is selected
        if strategy == "ml_adaptive" and DecisionEngine and structural_metrics:
            self.engine = DecisionEngine(structural_metrics)
        else:
            self.engine = None

    def should_checkpoint(
            self,
            work_since_last: float,
            failure_rate: float,
            current_line_cost: float,
            checkpoint_cost: float,
            execution_variance: float = 0.0
    ) -> bool:
        """
        Evaluates if a checkpoint should be placed AFTER the current line.

        Args:
            work_since_last: Total execution time (seconds/cycles) since the last save.
            failure_rate: Current environmental λ.
            current_line_cost: The measured/estimated time for the specific line just executed.
            checkpoint_cost: The energy/time cost to perform a save.
            execution_variance: Jitter in execution (useful for RDTSC/Cache research).
        """

        # ------------------------------------------------------
        # 1. ML Adaptive Strategy (Line-Level Intelligence)
        # ------------------------------------------------------
        if self.strategy == "ml_adaptive" and self.engine:
            # We pass the accumulated time and the specific line cost to the ML
            decision, _ = self.engine.evaluate(
                work_since_last_checkpoint=work_since_last,
                failure_rate=failure_rate,
                current_line_cost=current_line_cost
            )
            return decision

        # ------------------------------------------------------
        # 2. Analytical Strategy (Young/Daly Model)
        # ------------------------------------------------------
        if self.strategy == "analytical":
            # Optimum interval = sqrt(2 * CheckpointCost / Lambda)
            # This is the industry standard for comparison.
            threshold = math.sqrt(2 * checkpoint_cost / (failure_rate + 1e-9))
            return work_since_last >= threshold

        # ------------------------------------------------------
        # 3. Periodic Strategy (Static Heartbeat)
        # ------------------------------------------------------
        if self.strategy == "periodic":
            # Static time-based threshold (e.g., save every 0.05 seconds)
            periodic_threshold = 0.05
            return work_since_last >= periodic_threshold

        return False

    def __repr__(self):
        return f"CheckpointPolicy(strategy='{self.strategy}')"