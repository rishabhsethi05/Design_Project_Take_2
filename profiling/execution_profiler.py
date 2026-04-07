import time
from collections import defaultdict


class ExecutionProfiler:
    """
    Lightweight runtime profiler for basic blocks.

    Tracks:
    - execution count
    - mean execution time
    - variance (online, numerically stable)
    """

    def __init__(self):
        self.block_stats = defaultdict(lambda: {
            "count": 0,
            "mean": 0.0,
            "M2": 0.0,   # for variance
            "last_start": None
        })

    # ------------------------------------------------------
    # BLOCK TIMING
    # ------------------------------------------------------

    def start_block(self, block_id):
        self.block_stats[block_id]["last_start"] = time.perf_counter()

    def end_block(self, block_id):
        end_time = time.perf_counter()
        start_time = self.block_stats[block_id]["last_start"]

        if start_time is None:
            return

        duration = end_time - start_time

        stats = self.block_stats[block_id]

        stats["count"] += 1
        delta = duration - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = duration - stats["mean"]
        stats["M2"] += delta * delta2

        stats["last_start"] = None

    # ------------------------------------------------------
    # METRICS ACCESS
    # ------------------------------------------------------

    def get_block_mean(self, block_id):
        return self.block_stats[block_id]["mean"]

    def get_block_variance(self, block_id):
        stats = self.block_stats[block_id]
        if stats["count"] < 2:
            return 0.0
        return stats["M2"] / (stats["count"] - 1)

    def get_global_mean(self):
        total_time = 0.0
        total_count = 0

        for stats in self.block_stats.values():
            total_time += stats["mean"] * stats["count"]
            total_count += stats["count"]

        if total_count == 0:
            return 0.0

        return total_time / total_count

    def get_global_variance(self):
        variances = []
        for block_id in self.block_stats:
            variances.append(self.get_block_variance(block_id))

        if not variances:
            return 0.0

        return sum(variances) / len(variances)