import re
from typing import Dict, List
from static_analysis.basic_block_builder import BasicBlock


class CFGBuilder:
    """
    Structured Control Flow Graph builder for C programs.
    Links BasicBlocks to enable path-finding and runtime profiling.
    """

    def __init__(self, blocks: Dict[int, BasicBlock]):
        self.blocks = blocks
        self.block_ids = sorted(blocks.keys())

    def build(self) -> Dict[int, BasicBlock]:
        self._connect_blocks()
        return self.blocks

    def _connect_blocks(self):
        for i, block_id in enumerate(self.block_ids):
            block = self.blocks[block_id]
            if not block.lines: continue

            last_line = block.lines[-1][1].strip()

            # 1. Terminal Node (Return)
            if re.search(r"\breturn\b", last_line):
                continue

            # 2. Branching Node (If)
            if re.search(r"\bif\s*\(", last_line):
                self._connect_if(i)
                continue

            # 3. Cyclic Node (Loop Header)
            if re.search(r"\b(for|while)\s*\(", last_line):
                self._connect_loop(i)
                continue

            # 4. Sequential Fall-through
            self._connect_fallthrough(i)

    def _connect_fallthrough(self, index):
        if index + 1 < len(self.block_ids):
            self._add_edge(self.block_ids[index], self.block_ids[index + 1])

    def _connect_if(self, index):
        curr = self.block_ids[index]
        # Branch taken (True)
        if index + 1 < len(self.block_ids):
            self._add_edge(curr, self.block_ids[index + 1])
        # Branch not taken (False/Else) - Simple heuristic for structured C
        if index + 2 < len(self.block_ids):
            self._add_edge(curr, self.block_ids[index + 2])

    def _connect_loop(self, index):
        curr = self.block_ids[index]
        # Body entry
        if index + 1 < len(self.block_ids):
            body = self.block_ids[index + 1]
            self._add_edge(curr, body)
            # CRITICAL: Back-edge to header for loop iteration
            self._add_edge(body, curr)

        # Loop exit path
        if index + 2 < len(self.block_ids):
            self._add_edge(curr, self.block_ids[index + 2])

    def _add_edge(self, from_id, to_id):
        if to_id not in self.blocks[from_id].successors:
            self.blocks[from_id].successors.append(to_id)
        if from_id not in self.blocks[to_id].predecessors:
            self.blocks[to_id].predecessors.append(from_id)