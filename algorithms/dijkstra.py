import heapq
from typing import Dict, List, Tuple
from checkpointing.execution_context import ExecutionContext


class InstrumentedDijkstra:
    """
    Research-grade instrumented Dijkstra implementation.

    - Reports work per relaxation
    - Triggers checkpoint after permanent node finalization
    - Models graph + distance array as checkpoint state
    """

    def __init__(self, context: ExecutionContext):
        self.context = context

    # ==========================================================
    # PUBLIC ENTRY
    # ==========================================================

    def shortest_path(
        self,
        graph: Dict[int, List[Tuple[int, float]]],
        source: int
    ) -> Dict[int, float]:

        distances = {node: float("inf") for node in graph}
        distances[source] = 0.0

        visited = set()
        priority_queue = [(0.0, source)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            # Heap pop cost
            self.context.add_work(1.0)

            if current_node in visited:
                continue

            visited.add(current_node)

            # Structural progress: node permanently finalized
            state_size = self._estimate_state_size(
                graph,
                distances,
                visited,
                priority_queue
            )

            self.context.notify_event(
                event_type="node_finalized",
                state_size=state_size
            )

            for neighbor, weight in graph[current_node]:
                self.context.add_work(1.0)  # edge relaxation attempt

                new_distance = current_distance + weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbor))

                    # heap push cost
                    self.context.add_work(0.5)

        return distances

    # ==========================================================
    # STATE SIZE MODEL
    # ==========================================================

    def _estimate_state_size(
        self,
        graph,
        distances,
        visited,
        priority_queue
    ) -> float:
        """
        Estimate checkpoint state size.

        Components:
        - Distance array
        - Visited set
        - Priority queue
        """

        distance_cost = len(distances)
        visited_cost = len(visited)
        pq_cost = len(priority_queue) * 2

        return distance_cost + visited_cost + pq_cost