import re
from typing import List, Dict, Tuple


class BasicBlock:
    """
    Research-grade Basic Block representation.
    Encapsulates a sequence of instructions with one entry and one exit.
    """

    def __init__(self, block_id: int, start_line: int):
        self.id = block_id
        self.start_line = start_line
        self.end_line = start_line
        self.lines: List[Tuple[int, str, int, int]] = []
        self.successors: List[int] = []
        self.predecessors: List[int] = []

    def add_line(self, line_number: int, line_text: str, reads: int, writes: int):
        self.lines.append((line_number, line_text, reads, writes))
        self.end_line = line_number

    def __repr__(self):
        return f"BasicBlock(id={self.id}, range=[{self.start_line}-{self.end_line}], lines={len(self.lines)})"


class BasicBlockBuilder:
    """
    Constructs basic blocks from parsed C source code.
    Integrates static memory access metrics into the CFG nodes.
    """

    # Keywords that trigger a block boundary
    BRANCH_KEYWORDS = [r"\bif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b", r"\bdo\b", r"\bswitch\b", r"\bcase\b"]
    TERMINATORS = [r"\breturn\b", r"\bbreak\b", r"\bcontinue\b", r"\bgoto\b"]

    def __init__(self, parsed_program: Dict):
        self.lines = parsed_program["lines"]
        self.memory_lines = parsed_program.get("memory_lines", [])
        self.total_lines = len(self.lines)
        self.leaders = set()
        self.blocks: Dict[int, BasicBlock] = {}
        self.memory_map = self._build_memory_map()

    def _build_memory_map(self):
        """Maps line numbers to (reads, writes) for O(1) lookup during construction."""
        memory_map = {}
        for entry in self.memory_lines:
            code = entry["code"]
            # Optimization: Match line numbers using a secondary lookup if necessary,
            # but usually, the parser provides line_num directly.
            for line_num, line_text in self.lines:
                if line_text.strip() == code.strip():
                    memory_map[line_num] = (entry["reads"], entry["writes"])
                    break
        return memory_map

    def build(self) -> Dict[int, BasicBlock]:
        self._identify_leaders()
        self._construct_blocks()
        return self.blocks

    def _identify_leaders(self):
        """
        Standard Compiler Theory: A leader is:
        1. The first statement.
        2. Any statement that is the target of a branch.
        3. Any statement that immediately follows a branch.
        """
        if self.total_lines == 0: return

        # Rule 1: First line is always a leader
        self.leaders.add(self.lines[0][0])

        for idx, (line_number, line_text) in enumerate(self.lines):
            stripped = line_text.strip()

            # Rule 2 & 3: Branching logic
            if self._is_branch(stripped) or self._is_terminator(stripped):
                # The branch itself starts a new context (optional, but better for ML)
                self.leaders.add(line_number)

                # The line AFTER a branch/terminator is always a leader
                if idx + 1 < self.total_lines:
                    self.leaders.add(self.lines[idx + 1][0])

    def _is_branch(self, line: str) -> bool:
        return any(re.search(pattern, line) for pattern in self.BRANCH_KEYWORDS)

    def _is_terminator(self, line: str) -> bool:
        return any(re.search(pattern, line) for pattern in self.TERMINATORS)

    def _construct_blocks(self):
        sorted_leaders = sorted(list(self.leaders))
        leader_set = set(sorted_leaders)

        block_id = 0
        current_block = None

        for line_number, line_text in self.lines:
            if line_number in leader_set:
                if current_block is not None:
                    self.blocks[current_block.id] = current_block

                current_block = BasicBlock(block_id, line_number)
                block_id += 1

            if current_block:
                reads, writes = self.memory_map.get(line_number, (0, 0))
                current_block.add_line(line_number, line_text, reads, writes)

        if current_block:
            self.blocks[current_block.id] = current_block

    def print_blocks(self):
        for block in self.blocks.values():
            print(f"\n[Block {block.id}]")
            for ln, text, r, w in block.lines:
                print(f"  L{ln}: {text.strip():<30} | Mem(R:{r}, W:{w})")