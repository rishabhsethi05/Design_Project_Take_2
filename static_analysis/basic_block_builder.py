import re
from typing import List, Dict, Tuple


class BasicBlock:
    """
    Research-grade Basic Block representation.

    A Basic Block is a maximal sequence of consecutive statements
    with single entry and single exit.
    """

    def __init__(self, block_id: int, start_line: int):
        self.id = block_id
        self.start_line = start_line
        self.end_line = start_line
        self.lines: List[Tuple[int, str]] = []
        self.successors: List[int] = []
        self.predecessors: List[int] = []

    def add_line(self, line_number: int, line_text: str):
        self.lines.append((line_number, line_text))
        self.end_line = line_number

    def __repr__(self):
        return (
            f"BasicBlock(id={self.id}, "
            f"start={self.start_line}, "
            f"end={self.end_line}, "
            f"lines={len(self.lines)})"
        )


class BasicBlockBuilder:
    """
    Constructs basic blocks from parsed C source code.

    Leader identification rules:
    1. First line is a leader
    2. Target of a branch is a leader
    3. Statement immediately following a branch is a leader
    """

    BRANCH_KEYWORDS = [
        r"\bif\b",
        r"\belse\b",
        r"\bfor\b",
        r"\bwhile\b",
        r"\bdo\b",
        r"\bswitch\b",
        r"\bcase\b",
    ]

    TERMINATORS = [
        r"\breturn\b",
        r"\bbreak\b",
        r"\bcontinue\b",
        r"\bgoto\b",
    ]

    def __init__(self, parsed_program: Dict):
        self.lines = parsed_program["lines"]
        self.total_lines = len(self.lines)
        self.leaders = set()
        self.blocks: Dict[int, BasicBlock] = {}

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def build(self) -> Dict[int, BasicBlock]:
        """
        Main entry point.
        Returns dictionary of block_id -> BasicBlock
        """
        self._identify_leaders()
        self._construct_blocks()
        return self.blocks

    # -----------------------------------------------------
    # Leader Identification
    # -----------------------------------------------------

    def _identify_leaders(self):
        """
        Identify all leader lines based on control flow rules.
        """
        if self.total_lines == 0:
            return

        # Rule 1: First line is always a leader
        first_line_number = self.lines[0][0]
        self.leaders.add(first_line_number)

        for idx, (line_number, line_text) in enumerate(self.lines):
            stripped = line_text.strip()

            # Branch start → mark as leader
            if self._is_branch(stripped):
                self.leaders.add(line_number)

                # Next line becomes leader (fall-through)
                if idx + 1 < self.total_lines:
                    next_line_number = self.lines[idx + 1][0]
                    self.leaders.add(next_line_number)

            # Terminator → next line becomes leader
            if self._is_terminator(stripped):
                if idx + 1 < self.total_lines:
                    next_line_number = self.lines[idx + 1][0]
                    self.leaders.add(next_line_number)

    def _is_branch(self, line: str) -> bool:
        return any(re.search(pattern, line) for pattern in self.BRANCH_KEYWORDS)

    def _is_terminator(self, line: str) -> bool:
        return any(re.search(pattern, line) for pattern in self.TERMINATORS)

    # -----------------------------------------------------
    # Block Construction
    # -----------------------------------------------------

    def _construct_blocks(self):
        """
        Build BasicBlock objects from identified leaders.
        """
        sorted_leaders = sorted(self.leaders)
        leader_set = set(sorted_leaders)

        block_id = 0
        current_block = None

        for idx, (line_number, line_text) in enumerate(self.lines):

            # If this line is a leader, start new block
            if line_number in leader_set:
                if current_block is not None:
                    self.blocks[current_block.id] = current_block

                current_block = BasicBlock(block_id, line_number)
                block_id += 1

            if current_block is None:
                continue  # safety guard

            current_block.add_line(line_number, line_text)

        # Add final block
        if current_block is not None:
            self.blocks[current_block.id] = current_block

    # -----------------------------------------------------
    # Debug Utilities
    # -----------------------------------------------------

    def print_blocks(self):
        for block in self.blocks.values():
            print(block)
            for ln, text in block.lines:
                print(f"    {ln}: {text.strip()}")
            print()