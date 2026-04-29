import random
from dataclasses import dataclass
from typing import Optional, Dict
import math

from checkpointing.failure_model import PoissonFailureModel
# Import the policy we fixed earlier
from checkpointing.checkpoint_policy import CheckpointPolicy


# ==========================================================
# METRICS
# ========================== ================================

@dataclass
class ExecutionMetrics:
    useful_work_time: float = 0.0
    recompute_time: float = 0.0
    checkpoint_time: float = 0.0
    checkpoint_count: int = 0
    failure_count: int = 0


# ==========================================================
# CONTEXT
# ==========================================================

class ExecutionContext:

    def __init__(
            self,
            failure_rate: float,
            checkpoint_cost: float,
            state_size_cost_factor: float = 0.0002,
            structural_metrics: Optional[dict] = None,
            strategy: str = "ml_adaptive",
            seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)

        self.failure_rate = failure_rate
        self.checkpoint_cost = checkpoint_cost
        self.state_size_cost_factor = state_size_cost_factor
        self.strategy = strategy
        self.structural_metrics = structural_metrics or {}
        self.metrics = ExecutionMetrics()

        self.last_checkpoint_progress = 0.0
        self.current_progress = 0.0
        self.current_state_size = 0.0
        self.checkpoint_log = []

        self.total_reads = 0
        self.total_writes = 0

        self.failure_model = PoissonFailureModel(
            failure_rate=failure_rate,
            seed=seed or 42
        )

        # --- CRITICAL CHANGE ---
        # Initialize the actual Policy object here
        self.policy = CheckpointPolicy(
            strategy=self.strategy,
            structural_metrics=self.structural_metrics
        )

        self.min_checkpoint_gap = 0.005  # Lowered to allow ML more precision
        self.last_checkpoint_time = 0.0
        self.checkpoint_log = []

    def record_checkpoint(self, line_number):
        """Call this in your engine whenever a checkpoint is triggered."""
        self.checkpoint_log.append(line_number)

    def get_checkpoint_log(self):
        return self.checkpoint_log

    def add_memory_access(self, reads: int, writes: int):
        self.total_reads += reads
        self.total_writes += writes

    def _memory_intensity(self):
        total = self.total_reads + self.total_writes
        if total == 0:
            return 0.0
        return min(1.0, total / (1 + self.current_progress))

    def add_work(self, work_units: float):
        self.metrics.useful_work_time += work_units
        self.current_progress += work_units
        if self.failure_model.should_fail(work_units):
            self._handle_failure()

    def _handle_failure(self):
        self.metrics.failure_count += 1
        lost = self.current_progress - self.last_checkpoint_progress
        self.metrics.recompute_time += max(0.0, lost)
        self.current_progress = self.last_checkpoint_progress

    # ==========================================================
    # LINKED DECISION LOGIC
    # ==========================================================

    def evaluate_checkpoint(
            self,
            event_type: str,
            state_size: float,
            current_line_cost: float,
            verbose: bool = False,
            stall_hint: bool = False
    ) -> bool:
        self.current_state_size = state_size
        work_since_last = self.current_progress - self.last_checkpoint_progress

        # 1. Base Guard (Preventing 'Zero-Cost' thrashing)
        if work_since_last < self.min_checkpoint_gap:
            return False

        # 2. Use the External Policy (The one we fixed)
        # This replaces the hardcoded "Sniper" score with the ML Engine
        decision = self.policy.should_checkpoint(
            work_since_last=work_since_last,
            failure_rate=self.failure_rate,
            current_line_cost=current_line_cost,
            checkpoint_cost=self.checkpoint_cost
        )

        if decision:
            self._create_checkpoint(event_type, verbose)
            return True

        return False


    def _create_checkpoint(self, event_type: str, verbose: bool):
        """
        Finalized Checkpoint Creation with Hybrid Fallback logic.
        """
        # 1. Calculate the work interval since the last checkpoint
        # In this context, current_progress/current_time represent the same axis
        work_since_last = self.current_progress - self.last_checkpoint_time

        # 2. Calculate dynamic cost based on current state size
        cost = self.checkpoint_cost + (
                self.state_size_cost_factor * self.current_state_size
        )

        # 3. Update primary metrics
        self.metrics.checkpoint_count += 1
        self.metrics.checkpoint_time += cost

        # 4. Memory simulation (I/O pressure)
        self.total_reads += int(self.current_state_size * 2)
        self.total_writes += int(self.current_state_size * 1.5)

        # 5. Log the event
        self.checkpoint_log.append({
            "event": event_type,
            "progress": self.current_progress,
            "cost": cost
        })

        # 6. Update tracking markers
        self.last_checkpoint_progress = self.current_progress
        self.last_checkpoint_time = self.current_progress

        # 7. Verbose Logging with Hybrid Fallback Detection
        if verbose:
            # Standard message
            msg = f"[CHECKPOINT] {event_type} @ {self.current_progress:.4f}"

            # Calculate the theoretical optimal interval (Young's Formula)
            # Using 1e-9 to prevent DivisionByZero
            optimal_interval = math.sqrt(2 * self.checkpoint_cost / (self.failure_rate + 1e-9))

            # If we are using a hybrid strategy and we've surpassed the optimal
            # interval, flag this as a 'Hybrid-Fallback' save.
            if "hybrid" in self.strategy and work_since_last >= optimal_interval:
                msg = f"[CHECKPOINT] [HYBRID-FALLBACK] {event_type} @ {self.current_progress:.4f}"

            print(msg)

    def get_metrics(self) -> Dict:
        total = (
                self.metrics.useful_work_time +
                self.metrics.recompute_time +
                self.metrics.checkpoint_time
        )
        base = self.metrics.useful_work_time
        overhead = (total - base) / (base + 1e-9)

        return {
            "useful_work_time": base,
            "recompute_time": self.metrics.recompute_time,
            "checkpoint_time": self.metrics.checkpoint_time,
            "total_execution_time": total,
            "overhead_ratio": round(overhead, 6),
            "checkpoint_count": self.metrics.checkpoint_count,
            "failure_count": self.metrics.failure_count,
            "memory_ratio": self._memory_intensity(),
            "total_reads": self.total_reads,
            "total_writes": self.total_writes
        }