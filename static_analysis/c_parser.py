import re
from typing import Dict, List, Tuple


class CAlgorithmParser:
    """
    Hardware-Aware Parser for C algorithms.
    Calculates Clock Cycles and Energy Depletion based on MSP430FR6989 (16MHz) specs.
    """

    # MSP430 / Mapi-Pro Paper Constants
    CLOCK_FREQ_MHZ = 16
    T_CYCLE_SEC = 1 / (CLOCK_FREQ_MHZ * 10 ** 6)

    # Energy in Nanojoules (nJ) - Derived from paper Section 3.1
    E_SRAM_READ = 5.50  # nJ
    E_SRAM_WRITE = 5.60  # nJ
    E_LOGIC_OP = 2.10  # nJ (CPU Baseline per cycle)

    # Checkpointing Costs (FRAM)
    E_FRAM_WRITE_BYTE = 13.125  # nJ per byte

    LOOP_ITERATION_FACTOR = 10000

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.source_code = ""
        self.cleaned_lines: List[Tuple[int, str]] = []
        self.memory_lines: List[Dict] = []
        self.analysis: Dict = {}

    def load(self):
        with open(self.file_path, "r") as f:
            self.source_code = f.read()
        self._preprocess()
        self._attach_hardware_model()  # Changed from memory_model to hardware_model

    def _preprocess(self):
        clean_code = re.sub(r"/\*.*?\*/", "", self.source_code, flags=re.DOTALL)
        clean_code = re.sub(r"//.*", "", clean_code)
        self.cleaned_lines = []
        lines = clean_code.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped and any(c in stripped for c in ['#', '{', '}', ';', '=', '(', ')']):
                self.cleaned_lines.append((i, stripped))

    def _attach_hardware_model(self):
        """
        Calculates cycles and energy depletion for every line.
        Satisfies professor's requirement for 'Energy depletion per clock cycle'.
        """
        loop_depth = 0
        self.memory_lines = []

        for line_num, line in self.cleaned_lines:
            is_loop_head = bool(re.search(r"\b(for|while)\b", line))
            if is_loop_head:
                loop_depth += 1

            # Get hardware metrics
            reads, writes, cycles = self._calculate_line_metrics(line)

            # Calculate energy based on Professor's formula: E = P * t
            # Or specifically: E = (Reads * E_read) + (Writes * E_write) + (LogicCycles * E_logic)
            line_energy = (reads * self.E_SRAM_READ) + (writes * self.E_SRAM_WRITE) + (cycles * self.E_LOGIC_OP)

            scale = self.LOOP_ITERATION_FACTOR if loop_depth > 0 else 1

            self.memory_lines.append({
                "line_no": line_num,
                "code": line,
                "reads": reads * scale,
                "writes": writes * scale,
                "cycles": cycles * scale,
                "energy_nJ": line_energy * scale,
                "in_loop": loop_depth > 0
            })

            if (";" in line and not is_loop_head and loop_depth > 0) or "}" in line:
                loop_depth = max(0, loop_depth - 1)

    def _calculate_line_metrics(self, line: str) -> Tuple[int, int, int]:
        """
        Returns (Reads, Writes, Total Cycles) for a given C line.
        Based on MSP430 Instruction Set Architecture.
        """
        line = line.replace(";", "").strip()
        reads = 0
        writes = 0
        cycles = 1  # Every line takes at least 1 fetch cycle

        # 1. Detect Memory Operations
        if "=" in line and not any(k in line for k in ["if", "while", "for"]):
            lhs, rhs = line.split("=", 1)
            # LHS is usually a write
            writes += 1
            # RHS variables are reads
            reads += len(re.findall(r"\b[a-zA-Z_]\w*\b", rhs))
            # Array accesses add overhead (address calculation)
            cycles += line.count("[") * 2

            # 2. Logic and Bitwise (CRC specific)
        if any(op in line for op in ["^", "<<", ">>", "&", "|"]):
            cycles += 1  # Bitwise ops are usually 1 cycle on MSP430

        # 3. Conditionals
        if re.search(r"\b(if|while|for)\b", line):
            vars_in_cond = re.findall(r"\b[a-zA-Z_]\w*\b", line)
            reads += len([v for v in vars_in_cond if v not in ["if", "while", "for"]])
            cycles += 1  # Branching overhead

        return reads, writes, cycles

    def _detect_algorithm(self):
        code = self.source_code.lower()
        markers = {
            "CRC": [r"0x[0-9a-f]{4}", r"\^=", r"<<", r"icrctb"],
            "QUICKSORT": [r"partition", r"pivot", r"swap\(", r"istack"],
            "DIJKSTRA": [r"adjmatrix", r"idist", r"num_nodes"]
        }
        scores = {algo: sum(2 for p in patterns if re.search(p, code)) for algo, patterns in markers.items()}
        best_algo = max(scores, key=scores.get)
        return best_algo if scores[best_algo] >= 4 else "UNKNOWN"

    def analyze(self):
        self.analysis.update({
            "algorithm": self._detect_algorithm(),
            "total_cycles": sum(m['cycles'] for m in self.memory_lines),
            "total_energy_nJ": sum(m['energy_nJ'] for m in self.memory_lines),
            "line_count": len(self.cleaned_lines)
        })
        return self.analysis

    def get_program_representation(self) -> Dict:
        return {
            "lines": self.cleaned_lines,
            "memory_lines": self.memory_lines,
            "analysis": self.analysis
        }