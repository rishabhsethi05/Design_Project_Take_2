from typing import List
from checkpointing.execution_context import ExecutionContext


class InstrumentedCRC:
    """
    Research-grade instrumented CRC computation.

    - Linear streaming workload
    - Work reported per byte processed
    - Structural checkpoint trigger after each block
    - Models remainder + processed length as state
    """

    def __init__(
        self,
        context: ExecutionContext,
        polynomial: int = 0xEDB88320,
        block_size: int = 64
    ):
        self.context = context
        self.polynomial = polynomial
        self.block_size = block_size

    # ==========================================================
    # PUBLIC ENTRY
    # ==========================================================

    def compute(self, data: List[int]) -> int:
        """
        Compute CRC over input byte stream.
        """
        remainder = 0xFFFFFFFF
        total_length = len(data)

        for block_start in range(0, total_length, self.block_size):
            block = data[block_start:block_start + self.block_size]

            for byte in block:
                remainder ^= byte
                self.context.add_work(1.0)

                for _ in range(8):  # 8 bits per byte
                    self.context.add_work(0.2)

                    if remainder & 1:
                        remainder = (remainder >> 1) ^ self.polynomial
                    else:
                        remainder >>= 1

            # Structural progress: block completed
            state_size = self._estimate_state_size(
                processed=block_start + len(block),
                total=total_length
            )

            self.context.notify_event(
                event_type="block_processed",
                state_size=state_size
            )

        return remainder ^ 0xFFFFFFFF

    # ==========================================================
    # STATE SIZE MODEL
    # ==========================================================

    def _estimate_state_size(self, processed: int, total: int) -> float:
        """
        Estimate checkpoint state size.

        Components:
        - Current remainder (constant small)
        - Processed index
        - Total data reference
        """

        remainder_cost = 4
        index_cost = 2
        data_reference_cost = total * 0.1

        return remainder_cost + index_cost + data_reference_cost