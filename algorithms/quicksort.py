from typing import List
from checkpointing.execution_context import ExecutionContext


class InstrumentedQuickSort:
    """
    Research-grade instrumented QuickSort implementation.

    - Reports computational work to ExecutionContext
    - Triggers checkpoint decision after structural partition completion
    - Models recursion stack size as part of state cost
    """

    def __init__(self, context: ExecutionContext):
        self.context = context
        self.recursion_depth = 0
        self.max_depth = 0

    # ==========================================================
    # PUBLIC ENTRY
    # ==========================================================

    def sort(self, arr: List[int]) -> List[int]:
        self._quicksort(arr, 0, len(arr) - 1)
        return arr

    # ==========================================================
    # INTERNAL QUICK SORT
    # ==========================================================

    def _quicksort(self, arr: List[int], low: int, high: int):
        if low < high:
            self.recursion_depth += 1
            self.max_depth = max(self.max_depth, self.recursion_depth)

            pivot_index = self._partition(arr, low, high)

            # Notify structural progress (partition complete)
            state_size = self._estimate_state_size(len(arr))
            self.context.notify_event(
                event_type="partition_complete",
                state_size=state_size
            )

            self._quicksort(arr, low, pivot_index - 1)
            self._quicksort(arr, pivot_index + 1, high)

            self.recursion_depth -= 1

    # ==========================================================
    # PARTITION FUNCTION
    # ==========================================================

    def _partition(self, arr: List[int], low: int, high: int) -> int:
        pivot = arr[high]
        i = low - 1

        for j in range(low, high):
            # Each comparison = 1 unit of work
            self.context.add_work(1.0)

            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

                # Swap cost = additional work
                self.context.add_work(0.5)

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        self.context.add_work(0.5)

        return i + 1

    # ==========================================================
    # STATE SIZE MODEL
    # ==========================================================

    def _estimate_state_size(self, n: int) -> float:
        """
        Estimate checkpoint state size.

        Components:
        - Array size
        - Recursion stack depth
        """

        array_state_cost = n
        stack_cost = self.recursion_depth * 5

        return array_state_cost + stack_cost