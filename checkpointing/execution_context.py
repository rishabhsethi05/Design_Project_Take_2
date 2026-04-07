import random
import math
from dataclasses import dataclass
from typing import Optional, Dict

# Integrated Imports
try:
    from checkpointing.failure_model import PoissonFailureModel
except ImportError:
    PoissonFailureModel = None

try:
    from checkpointing.checkpoint_policy import CheckpointPolicy
except ImportError:
    CheckpointPolicy = None


@dataclass
class ExecutionMetrics:
    useful_work_time: float = 0.0
    recompute_time: float = 0.0
    checkpoint_time: float = 0.0
    checkpoint_count: int = 0
    failure_count: int = 0


class ExecutionContext:
    """
    Main Research Orchestrator for Intermittent Execution.
    - Tracks Time/Energy metrics at Instruction Granularity.
    - Manages Stochastic Failures via Poisson Distribution.
    - Bridges the Engine to the ML Policy.
    """

    def __init__(
            self,
            failure_rate: float,
            checkpoint_cost: float,
            state_size_cost_factor: float = 0.01,
            structural_metrics: Optional[dict] = None,
            strategy: str = "ml_adaptive",
            seed: Optional[int] = None
    ):
        # 1. Stochastic Setup (Critical for generating Mean +/- Std Dev)
        if seed is not None:
            random.seed(seed)

        self.failure_rate = failure_rate
        self.base_checkpoint_cost = checkpoint_cost
        self.state_size_cost_factor = state_size_cost_factor
        self.metrics = ExecutionMetrics()

        # 2. State Tracking
        self.last_checkpoint_progress = 0.0
        self.current_progress = 0.0
        self.current_state_size = 0.0
        self.checkpoint_log = []

        # 3. Runtime Profiling Stats
        self.total_blocks_executed = 0
        self.total_block_work = 0.0

        # 4. Failure Physics Plugin
        if PoissonFailureModel:
            self.failure_model = PoissonFailureModel(failure_rate, seed=seed)
        else:
            self.failure_model = None

        # 5. Decision Policy Plugin (ML vs Analytical vs Periodic)
        if CheckpointPolicy:
            self.policy = CheckpointPolicy(
                strategy=strategy,
                structural_metrics=structural_metrics
            )
        else:
            self.policy = None

        self.simulation_active = True

    # ==========================================================
    # WORK EXECUTION & FAILURE LOGIC
    # ==========================================================

    def add_work(self, work_units: float):
        """Executes work (time/cycles) and simulates power stability."""
        if not self.simulation_active:
            return

        # Note: 'work_units' is now actual time (seconds) from the Engine
        self.total_block_work += work_units

        # Determine if power fails during this specific instruction/line
        if self.failure_model:
            failed = self.failure_model.should_fail(work_units)
        else:
            # P(Failure) = 1 - e^(-lambda * t)
            prob = 1 - math.exp(-self.failure_rate * work_units)
            failed = random.random() < prob

        if failed:
            self._handle_failure()
        else:
            # If successful, commit time to volatile RAM
            self.metrics.useful_work_time += work_units
            self.current_progress += work_units

    def _handle_failure(self):
        """Simulates power loss: Wipes RAM and rolls back to NVRAM."""
        self.metrics.failure_count += 1

        # Penalty = everything executed since the last successful save
        lost_progress = self.current_progress - self.last_checkpoint_progress
        self.metrics.recompute_time += lost_progress

        # Roll back to the last stable state (NVRAM)
        self.current_progress = self.last_checkpoint_progress

    # ==========================================================
    # CHECKPOINT EVALUATION (The ML Integration Point)
    # ==========================================================

    def evaluate_checkpoint(self, event_type: str, state_size: float, current_line_cost: float):
        """
        Calls the ML Policy to decide if we should save state
        after the line just executed.
        """
        self.current_state_size = state_size

        # Time accumulated in RAM since the last NVRAM write
        work_since_last = self.current_progress - self.last_checkpoint_progress

        if work_since_last <= 0:
            return

        # Delegate decision to the Policy (ML/Analytical)
        should_save = False
        if self.policy:
            # We now pass the 'current_line_cost' as required by the professor
            should_save = self.policy.should_checkpoint(
                work_since_last=work_since_last,
                failure_rate=self.failure_rate,
                current_line_cost=current_line_cost,
                checkpoint_cost=self.base_checkpoint_cost
            )
        else:
            # Simple threshold fallback if policy is missing
            should_save = work_since_last >= 0.05

        if should_save:
            self._create_checkpoint(event_type)

    def _create_checkpoint(self, event_type: str):
        """Calculates energy/time cost and 'burns' the state to NVRAM."""
        cost = (
                self.base_checkpoint_cost +
                (self.state_size_cost_factor * self.current_state_size)
        )

        self.metrics.checkpoint_count += 1
        self.metrics.checkpoint_time += cost

        # Log the specific line number (event_type) for the professor's 'Ultimate Goal'
        self.checkpoint_log.append({
            "event_type": event_type,
            "progress": round(self.current_progress, 4),
            "cost": round(cost, 4)
        })

        # Progress is now committed to Non-Volatile Memory
        self.last_checkpoint_progress = self.current_progress

    # ==========================================================
    # RESULTS EXPORT
    # ==========================================================

    def get_metrics(self) -> Dict:
        """Generates the final execution report for data analysis."""
        total_time = (
                self.metrics.useful_work_time +
                self.metrics.recompute_time +
                self.metrics.checkpoint_time
        )
        baseline = self.metrics.useful_work_time

        return {
            "useful_work_time": round(baseline, 6),
            "recompute_time": round(self.metrics.recompute_time, 6),
            "checkpoint_time": round(self.metrics.checkpoint_time, 6),
            "total_execution_time": round(total_time, 6),
            "overhead_ratio": round((total_time - baseline) / baseline, 6) if baseline > 0 else 0,
            "checkpoint_count": self.metrics.checkpoint_count,
            "failure_count": self.metrics.failure_count,
            "checkpoint_log": self.checkpoint_log
        }