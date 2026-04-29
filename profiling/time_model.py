import statistics


class TimeModel:
    """
    Balanced Instruction-Level Time Model.
    Updated to provide non-uniform line costs to guide ML decisions.
    """

    def __init__(self, alpha=0.4):  # Slightly lower alpha for better stability
        self.alpha = alpha
        self.block_history = {}
        self.line_timing_map = {}
        self.MIN_LINE_COST = 0.0001

    def update_block_metrics(self, block_id: int, measured_time: float, lines: list):
        line_count = len(lines)
        if line_count == 0:
            return

        # Ensure we don't drop below a realistic threshold for the hardware
        measured_time = max(measured_time, 0.0005)

        # Exponential Moving Average (EMA) for block history
        if block_id not in self.block_history:
            self.block_history[block_id] = measured_time
        else:
            self.block_history[block_id] = (
                    self.alpha * measured_time +
                    (1 - self.alpha) * self.block_history[block_id]
            )

        # Distribute costs based on line "weight"
        # Logic: Lines with more characters or complex operators (*, ^, <<) cost more
        total_weight = 0
        weights = []
        for entry in lines:
            code = str(entry[1]).lower()
            # Heuristic: Assign weight based on complexity
            weight = 1.0
            if any(op in code for op in ['^', '<<', '>>', '*', '/']):
                weight = 1.5
            if any(op in code for op in ['[', '->', '.']):
                weight = 1.2  # Memory access penalty

            weights.append(weight)
            total_weight += weight

        # Apply the weighted distribution to the total block time
        for i, entry in enumerate(lines):
            line_num = entry[0]
            share = weights[i] / total_weight
            line_cost = (self.block_history[block_id] * share)
            self.line_timing_map[line_num] = max(line_cost, self.MIN_LINE_COST)

    def get_line_cost(self, line_number: int) -> float:
        return self.line_timing_map.get(line_number, self.MIN_LINE_COST)

    def predict_region_cost(self, line_numbers: list) -> float:
        return sum(self.get_line_cost(ln) for ln in line_numbers)

    def get_total_execution_estimate(self) -> float:
        return sum(self.block_history.values())