class MetricsExtractor:
    """
    Extracts structural metrics from a CFG.
    These metrics are later used for adaptive checkpoint placement.
    """

    def __init__(self, blocks):
        """
        blocks: dictionary of block_id -> BasicBlock
        """
        self.blocks = blocks

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def extract(self):
        metrics = {}

        metrics["total_basic_blocks"] = self._count_blocks()
        metrics["total_edges"] = self._count_edges()
        metrics["average_block_size"] = self._average_block_size()
        metrics["max_block_size"] = self._max_block_size()
        metrics["branch_count"] = self._count_branches()
        metrics["loop_count"] = self._count_loops()
        metrics["cyclomatic_complexity"] = self._cyclomatic_complexity(
            metrics["total_edges"],
            metrics["total_basic_blocks"]
        )

        return metrics

    # --------------------------------------------------
    # METRIC COMPUTATIONS
    # --------------------------------------------------

    def _count_blocks(self):
        return len(self.blocks)

    def _count_edges(self):
        edge_count = 0
        for block in self.blocks.values():
            edge_count += len(block.successors)
        return edge_count

    def _average_block_size(self):
        total_lines = 0
        for block in self.blocks.values():
            total_lines += len(block.lines)

        if len(self.blocks) == 0:
            return 0

        return total_lines / len(self.blocks)

    def _max_block_size(self):
        max_size = 0
        for block in self.blocks.values():
            size = len(block.lines)
            if size > max_size:
                max_size = size
        return max_size

    def _count_branches(self):
        branch_count = 0
        for block in self.blocks.values():
            if len(block.successors) > 1:
                branch_count += 1
        return branch_count

    def _count_loops(self):
        """
        Simple heuristic:
        If a successor block ID is less than current block ID,
        we treat it as a back-edge → loop.
        """
        loop_count = 0

        for block_id, block in self.blocks.items():
            for succ in block.successors:
                if succ <= block_id:
                    loop_count += 1

        return loop_count

    def _cyclomatic_complexity(self, edges, nodes):
        """
        McCabe Cyclomatic Complexity:
        M = E - N + 2P
        Assuming single connected component → P = 1
        """
        if nodes == 0:
            return 0

        return edges - nodes + 2