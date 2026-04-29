class MetricsExtractor:
    """
    Extracts structural metrics from a CFG.
    These metrics serve as the primary features for the ML Decision Engine.
    """

    def __init__(self, blocks):
        self.blocks = blocks

    def extract(self):
        # Cache basic counts to avoid re-calculating
        n_nodes = len(self.blocks)
        n_edges = sum(len(b.successors) for b in self.blocks.values())

        metrics = {
            "total_basic_blocks": n_nodes,
            "total_edges": n_edges,
            "average_block_size": self._average_block_size(n_nodes),
            "max_block_size": self._max_block_size(),
            "branch_count": self._count_branches(),
            "loop_count": self._count_loops(),
            "cyclomatic_complexity": self._cyclomatic_complexity(n_edges, n_nodes)
        }
        return metrics

    def _average_block_size(self, n_nodes):
        if n_nodes == 0: return 0
        total_lines = sum(len(block.lines) for block in self.blocks.values())
        return total_lines / n_nodes

    def _max_block_size(self):
        if not self.blocks: return 0
        return max(len(block.lines) for block in self.blocks.values())

    def _count_branches(self):
        # A branch is any block with multiple exits (if-else, switch, loop headers)
        return sum(1 for block in self.blocks.values() if len(block.successors) > 1)

    def _count_loops(self):
        """
        Detects back-edges in the CFG.
        A back-edge exists if a successor can reach its own ancestor.
        """
        loop_count = 0
        for block_id, block in self.blocks.items():
            for succ_id in block.successors:
                # Optimized heuristic: In structured C, back-edges
                # almost always point to lower or equal block IDs.
                if succ_id <= block_id:
                    loop_count += 1
        return loop_count

    def _cyclomatic_complexity(self, edges, nodes):
        """
        McCabe's Complexity Metric: M = E - N + 2.
        Higher M indicates more paths, suggesting more frequent checkpoints.
        """
        if nodes == 0: return 1
        # Complexity is at minimum 1 (linear code)
        return max(1, edges - nodes + 2)