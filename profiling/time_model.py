import statistics


class TimeModel:
    """
    Advanced Instruction-Level Time Model.

    Functions:
    1. Tracks execution time at the Basic Block level.
    2. Derived 'Average Time Per Line' for granular checkpointing.
    3. Handles RDTSC cycle counts (or high-res nanoseconds).
    """

    def __init__(self, alpha=0.7):
        """
        alpha: Smoothing factor for Moving Average (higher = trust new data more).
        """
        self.alpha = alpha

        # Mapping: block_id -> { 'avg_total_time': float, 'line_count': int }
        self.block_history = {}

        # Mapping: line_number -> average_execution_time (The "Goal" metric)
        self.line_timing_map = {}

    def update_block_metrics(self, block_id: int, measured_time: float, lines: list):
        """
        Updates the model with a fresh measurement from the profiler.
        Calculates the new 'Average Time Per Line' for that block.
        """
        line_count = len(lines)
        if line_count == 0:
            return

        # 1. Update the Block's mean time using an Exponential Moving Average
        if block_id not in self.block_history:
            self.block_history[block_id] = measured_time
        else:
            self.block_history[block_id] = (self.alpha * measured_time) + \
                                           ((1 - self.alpha) * self.block_history[block_id])

        # 2. Calculate Average Time Per Line for this specific block
        avg_line_time = self.block_history[block_id] / line_count

        # 3. Map this timing to every line within the block
        # This satisfies Point #3 & #6 from your professor's list
        for line_num, _ in lines:
            self.line_timing_map[line_num] = avg_line_time

    def get_line_cost(self, line_number: int) -> float:
        """
        Returns the estimated time/cycles for a specific line.
        Default return is a small epsilon if the line hasn't been profiled yet.
        """
        return self.line_timing_map.get(line_number, 0.000001)

    def predict_region_cost(self, line_numbers: list) -> float:
        """
        Predicts the total execution time for a sequence of lines.
        Used by the ML model to decide 'Risk of Recomputation'.
        """
        return sum(self.get_line_cost(ln) for ln in line_numbers)

    def get_total_execution_estimate(self) -> float:
        """
        Point #1: Sum of all known block timings to provide Total Execution Time.
        """
        return sum(self.block_history.values())