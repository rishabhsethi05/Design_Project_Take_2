import random
import time
from profiling.execution_profiler import ExecutionProfiler
from profiling.time_model import TimeModel


class CFGExecutionEngine:
    def __init__(self, blocks, context):
        self.blocks = blocks
        self.context = context
        self.current_block_id = 0
        self.profiler = ExecutionProfiler()
        self.time_model = TimeModel()
        self.context.profiler = self.profiler
        self.visited_counts = {block_id: 0 for block_id in blocks.keys()}
        self.simulated_stack_depth = 0

    def choose_successor(self, block):
        if not block.successors: return None
        return block.successors[0] if len(block.successors) == 1 else random.choice(block.successors)

    def compute_dynamic_state_size(self, block_id):
        unique_blocks_visited = sum(1 for c in self.visited_counts.values() if c > 0)
        return len(self.blocks) + unique_blocks_visited + self.visited_counts[block_id] + self.simulated_stack_depth

    def execute(self, max_steps=10000):
        verbose = (self.context.strategy in ["ml_adaptive", "hybrid"])
        steps = 0

        while steps < max_steps:
            block = self.blocks[self.current_block_id]
            self.profiler.start_block(str(self.current_block_id))
            self.visited_counts[self.current_block_id] += 1

            # 1. Simulate Block Execution Time
            start_time = time.perf_counter()
            time.sleep(0.02 * len(block.lines) * random.uniform(0.8, 1.2))
            measured_duration = time.perf_counter() - start_time

            self.time_model.update_block_metrics(self.current_block_id, measured_duration, block.lines)

            # 2. Line-by-Line Execution
            for entry in block.lines:
                line_num, line_code = entry[0], entry[1]
                reads = entry[2] if len(entry) == 4 else 0
                writes = entry[3] if len(entry) == 4 else 0

                self.context.add_memory_access(reads, writes)
                line_cost = self.time_model.get_line_cost(line_num)
                self.context.add_work(line_cost)

                # Unified Strategy Trigger
                # We calculate lookahead (stall_hint) and let the Policy decide
                state_map = {str(bid): self.compute_dynamic_state_size(bid) for bid in self.blocks}
                predicted_cost = self.profiler.predict_next_state_cost(str(self.current_block_id), state_map)

                # Stall hint: Is the next state significantly cheaper?
                dynamic_state_size = self.compute_dynamic_state_size(self.current_block_id)
                stall_hint = predicted_cost < (dynamic_state_size * 0.85)

                # THE KEY FIX: One call to rule them all.
                # This respects the strategy set in ExecutionContext.
                self.context.evaluate_checkpoint(
                    event_type=f"Line {line_num}",
                    state_size=dynamic_state_size,
                    current_line_cost=line_cost,
                    verbose=verbose,
                    stall_hint=stall_hint
                )

            # 3. Handle Transitions
            self.profiler.end_block(str(self.current_block_id))
            next_block_id = self.choose_successor(block)
            if next_block_id is None: break

            # Stack depth simulation for state size
            last_line = block.lines[-1][1].strip()
            if "return" in last_line and self.simulated_stack_depth > 0:
                self.simulated_stack_depth -= 1
            elif "(" in last_line and ";" not in last_line:
                self.simulated_stack_depth += 1

            self.current_block_id = next_block_id
            steps += 1

        if verbose: self._report_stats()

    def _report_stats(self):
        final_time = self.time_model.get_total_execution_estimate()
        print(f"\n[Engine] {self.context.strategy.upper()} Pass Complete.")
        print(f"[Engine] Estimated Hardware Time: {final_time:.6f}s")
        print(f"[Memory] Total Reads: {self.context.total_reads} | Writes: {self.context.total_writes}")
        ratio = self.context.total_reads / (self.context.total_writes or 1)
        print(f"[Memory] Read/Write Ratio: {ratio:.2f}")