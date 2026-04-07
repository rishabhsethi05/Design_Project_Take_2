import random
import time
from profiling.execution_profiler import ExecutionProfiler
from profiling.time_model import TimeModel


class CFGExecutionEngine:
    """
    The 'Heart' of the simulator.
    Steps through Basic Blocks and individual lines to simulate
    real-world hardware execution and checkpointing.
    """

    def __init__(self, blocks, context):
        """
        blocks: dictionary {block_id: BasicBlock}
        context: The ExecutionContext (manages failures and saves)
        """
        self.blocks = blocks
        self.context = context
        self.current_block_id = 0

        # Profiling + time modeling
        self.profiler = ExecutionProfiler()
        self.time_model = TimeModel()

        # Expose profiler to context
        self.context.profiler = self.profiler

        # Track visit counts for state size estimation and loop detection
        self.visited_counts = {block_id: 0 for block_id in blocks.keys()}
        self.simulated_stack_depth = 0

    # --------------------------------------------------
    # SUCCESSOR SELECTION
    # --------------------------------------------------

    def choose_successor(self, block):
        if not block.successors:
            return None

        num_successors = len(block.successors)
        if num_successors == 1:
            return block.successors[0]

        # Dynamic weighting for branching
        primary_weight = 0.7
        others_weight = 0.3 / (num_successors - 1)
        weights = [primary_weight] + [others_weight] * (num_successors - 1)

        return random.choices(block.successors, weights=weights, k=1)[0]

    # --------------------------------------------------
    # DYNAMIC STATE SIZE MODEL
    # --------------------------------------------------

    def compute_dynamic_state_size(self, block_id):
        unique_blocks_visited = sum(1 for count in self.visited_counts.values() if count > 0)
        loop_depth_factor = self.visited_counts[block_id]

        return (
                len(self.blocks)
                + unique_blocks_visited
                + loop_depth_factor
                + self.simulated_stack_depth
        )

    # --------------------------------------------------
    # EXECUTION LOOP (The ML-Integrated Version)
    # --------------------------------------------------

    def execute(self, max_steps=10000):
        """
        The main loop that steps through the program structure.
        """
        steps = 0

        while steps < max_steps:
            block = self.blocks[self.current_block_id]
            self.visited_counts[self.current_block_id] += 1

            # --- 1. PROFILING THE BLOCK (Point #1) ---
            # Measure actual CPU time elapsed for this block
            start_time = time.perf_counter()

            # Simulated hardware latency (Point #3: Avg time per line)
            # Increase this value to see more checkpoints in the log
            time.sleep(0.005)

            end_time = time.perf_counter()
            measured_duration = end_time - start_time

            # --- 2. UPDATE TIME MODEL ---
            # Calibrate line-level timing based on the block measurement
            self.time_model.update_block_metrics(
                block_id=self.current_block_id,
                measured_time=measured_duration,
                lines=block.lines
            )

            # --- 3. LINE-BY-LINE EXECUTION (The ML Core) ---
            # --- 3. LINE-BY-LINE EXECUTION (Point #6 Enhanced) ---
            for line_num, line_code in block.lines:
                # OPTIONAL: Uncomment the line below to see a live trace of lines
                # print(f"  [Executing] Line {line_num}: {line_code.strip()}")

                # Get instruction cost from the model
                line_cost = self.time_model.get_line_cost(line_num)

                # Commit work
                self.context.add_work(line_cost)

                # Evaluate Checkpoint
                dynamic_state_size = self.compute_dynamic_state_size(self.current_block_id)

                # We pass the line_num and the line_code to the log
                self.context.evaluate_checkpoint(
                    event_type=f"Line {line_num}: {line_code.strip()}", # Detailed description
                    state_size=dynamic_state_size,
                    current_line_cost=line_cost
                )

            # --- 4. TRANSITION TO NEXT BLOCK ---
            next_block_id = self.choose_successor(block)

            if next_block_id is None:
                break

            # Heuristic for stack depth tracking
            last_line = block.lines[-1][1].strip()
            if "return" in last_line and self.simulated_stack_depth > 0:
                self.simulated_stack_depth -= 1
            elif "(" in last_line and ";" not in last_line:
                self.simulated_stack_depth += 1

            self.current_block_id = next_block_id
            steps += 1

        final_time = self.time_model.get_total_execution_estimate()
        print(f"\n[Engine] Simulation Complete.")
        print(f"[Engine] Total Hardware Execution Time (Estimated): {final_time:.6f}s")