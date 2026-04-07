import re
from typing import Dict
from static_analysis.basic_block_builder import BasicBlock


class CFGBuilder:
    """
    Structured Control Flow Graph builder for C programs.

    Builds successor/predecessor relationships between basic blocks.
    Assumes structured C (no goto).
    """

    def __init__(self, blocks: Dict[int, BasicBlock]):
        self.blocks = blocks
        self.block_ids = sorted(blocks.keys())

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def build(self) -> Dict[int, BasicBlock]:
        self._connect_blocks()
        return self.blocks

    # ==========================================================
    # CFG Construction
    # ==========================================================

    def _connect_blocks(self):

        for i, block_id in enumerate(self.block_ids):
            block = self.blocks[block_id]

            last_line = block.lines[-1][1].strip()

            # --------------------------------------------------
            # RETURN → no successors
            # --------------------------------------------------
            if self._is_return(last_line):
                continue

            # --------------------------------------------------
            # IF statement
            # --------------------------------------------------
            if self._is_if(last_line):
                self._connect_if(i)
                continue

            # --------------------------------------------------
            # FOR / WHILE loop
            # --------------------------------------------------
            if self._is_loop(last_line):
                self._connect_loop(i)
                continue

            # --------------------------------------------------
            # Default fall-through
            # --------------------------------------------------
            self._connect_fallthrough(i)

    # ==========================================================
    # EDGE CONNECTION HELPERS
    # ==========================================================

    def _connect_fallthrough(self, index):
        if index + 1 < len(self.block_ids):
            current_id = self.block_ids[index]
            next_id = self.block_ids[index + 1]
            self._add_edge(current_id, next_id)

    def _connect_if(self, index):
        current_id = self.block_ids[index]

        # True branch → next block
        if index + 1 < len(self.block_ids):
            true_id = self.block_ids[index + 1]
            self._add_edge(current_id, true_id)

        # False branch → block after next
        if index + 2 < len(self.block_ids):
            false_id = self.block_ids[index + 2]
            self._add_edge(current_id, false_id)

    def _connect_loop(self, index):
        current_id = self.block_ids[index]

        # Loop body
        if index + 1 < len(self.block_ids):
            body_id = self.block_ids[index + 1]
            self._add_edge(current_id, body_id)

            # Back-edge: body → loop header
            self._add_edge(body_id, current_id)

        # Exit edge
        if index + 2 < len(self.block_ids):
            exit_id = self.block_ids[index + 2]
            self._add_edge(current_id, exit_id)

    def _add_edge(self, from_id, to_id):
        if to_id not in self.blocks[from_id].successors:
            self.blocks[from_id].successors.append(to_id)

        if from_id not in self.blocks[to_id].predecessors:
            self.blocks[to_id].predecessors.append(from_id)

    # ==========================================================
    # DETECTION HELPERS
    # ==========================================================

    def _is_if(self, line: str) -> bool:
        return re.search(r"\bif\s*\(", line) is not None

    def _is_loop(self, line: str) -> bool:
        return re.search(r"\bfor\s*\(|\bwhile\s*\(", line) is not None

    def _is_return(self, line: str) -> bool:
        return re.search(r"\breturn\b", line) is not None