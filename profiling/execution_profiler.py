import time
from collections import defaultdict
from typing import Optional, Dict


class ExecutionProfiler:
    """
    Enhanced runtime profiler for basic blocks.
    Optimized to reduce instrumentation overhead in tight loops.
    """

    def __init__(self):
        # performance stats: block_id -> metrics
        self.block_stats = defaultdict(lambda: {
            "count": 0,
            "mean": 0.0,
            "M2": 0.0,
            "last_start": 0.0  # Use 0.0 instead of None for faster float ops
        })

        # Transition tracking
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.prev_block_id: Optional[str] = None

    # ------------------------------------------------------
    # OPTIMIZED INSTRUMENTATION
    # ------------------------------------------------------

    def start_block(self, block_id: str):
        # Update transition map first
        if self.prev_block_id is not None:
            self.transitions[self.prev_block_id][block_id] += 1

        self.block_stats[block_id]["last_start"] = time.perf_counter()
        self.prev_block_id = block_id

    def end_block(self, block_id: str):
        end_t = time.perf_counter()
        stats = self.block_stats[block_id]

        # Avoid overhead if start wasn't recorded
        if stats["last_start"] == 0.0:
            return

        duration = end_t - stats["last_start"]

        # Welford's Algorithm (Online Mean/Variance)
        stats["count"] += 1
        count = stats["count"]
        delta = duration - stats["mean"]
        stats["mean"] += delta / count
        # Optimization: We only calculate M2 if we really need variance to save CPU
        stats["M2"] += delta * (duration - stats["mean"])

        stats["last_start"] = 0.0

    # ------------------------------------------------------
    # PREDICTIVE METRICS
    # ------------------------------------------------------

    def predict_next_state_cost(self, current_block: str, state_size_map: Dict[str, float]) -> float:
        """
        Calculates the expected state size cost of the next step.
        """
        successors = self.transitions.get(current_block)
        if not successors:
            return state_size_map.get(current_block, 0.0)

        total_exits = sum(successors.values())

        # NEW: Confidence Check.
        # If we've only seen this block once, predictions aren't reliable.
        if total_exits < 3:
            return state_size_map.get(current_block, 0.0)

        expected_cost = 0.0
        for succ_id, count in successors.items():
            prob = count / total_exits
            expected_cost += prob * state_size_map.get(succ_id, 0.0)

        return expected_cost

    # ------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------

    def get_block_variance(self, block_id: str) -> float:
        stats = self.block_stats[block_id]
        return stats["M2"] / (stats["count"] - 1) if stats["count"] > 1 else 0.0

    def reset_traversal(self):
        self.prev_block_id = None