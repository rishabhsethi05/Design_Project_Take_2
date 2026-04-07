import re
from typing import Dict, List, Tuple


class CAlgorithmParser:
    """
    Structural static analysis for C algorithm files.

    Provides:
    - Algorithm metadata
    - Line-level representation
    - Cleaned source lines
    - Structural metrics

    This is not a full compiler.
    It is a controlled static analysis layer for checkpoint research.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.source_code = ""
        self.cleaned_lines: List[Tuple[int, str]] = []
        self.analysis: Dict = {}

    # ==========================================================
    # LOAD FILE
    # ==========================================================

    def load(self):
        with open(self.file_path, "r") as f:
            self.source_code = f.read()

        self._preprocess()

    # ==========================================================
    # PREPROCESSING
    # ==========================================================

    def _preprocess(self):
        """
        Removes comments and prepares line-level representation.
        """

        code = self._remove_comments(self.source_code)

        lines = code.split("\n")

        self.cleaned_lines = []

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped == "":
                continue
            self.cleaned_lines.append((idx, stripped))

    def _remove_comments(self, code: str) -> str:
        # Remove block comments
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        # Remove single-line comments
        code = re.sub(r"//.*", "", code)
        return code

    # ==========================================================
    # MAIN ANALYSIS
    # ==========================================================

    def analyze(self):
        if not self.source_code:
            raise ValueError("Source code not loaded.")

        self.analysis["algorithm"] = self._detect_algorithm()
        self.analysis["loop_count"] = self._count_loops()
        self.analysis["function_count"] = self._count_functions()
        self.analysis["uses_recursion"] = self._detect_recursion()
        self.analysis["line_count"] = len(self.cleaned_lines)

        return self.analysis

    # ==========================================================
    # DETECTION LOGIC
    # ==========================================================

    def _detect_algorithm(self):
        code = self.source_code.lower()

        # We use a scoring system to see which 'Signature' is strongest
        scores = {
            "quicksort": 0,
            "dijkstra": 0,
            "crc": 0
        }

        # --- QUICKSORT SIGNATURES ---
        # Look for pivot logic, partitioning, or stack management
        qsort_keywords = ["partition", "pivot", "quicksort", "stack", "istack", "swap"]
        for word in qsort_keywords:
            if word in code: scores["quicksort"] += 2

        # Structural Signature: Pivot calculation (index shifting)
        if ">> 1" in code or "+ ir) / 2" in code:
            scores["quicksort"] += 5
        # Array element swapping pattern
        if "arr[i]" in code and "arr[j]" in code and "temp" in code:
            scores["quicksort"] += 3

        # --- DIJKSTRA SIGNATURES ---
        # Look for graph traversal and distance arrays
        dijkstra_keywords = ["dijkstra", "priority_queue", "dist", "visited", "min_dist", "graph", "vertex"]
        for word in dijkstra_keywords:
            if word in code: scores["dijkstra"] += 2

        # Structural Signature: Relaxation Step (dist[u] + weight < dist[v])
        if "dist[" in code and ("+" in code or "<" in code):
            scores["dijkstra"] += 5
        # Infinity constant often used in Dijkstra
        if "9999" in code or "int_max" in code or "inf" in code:
            scores["dijkstra"] += 3

        # --- CRC SIGNATURES ---
        # Look for polynomial math and bit-shifting
        crc_keywords = ["crc", "polynomial", "poly", "bit", "width", "check"]
        for word in crc_keywords:
            if word in code: scores["crc"] += 2

        # Structural Signature: The CRC Polynomial math (XOR and Left Shift)
        if "^" in code and "<<" in code:
            scores["crc"] += 5
        # Common CRC constants (Hexadecimal signatures)
        if any(h in code for h in ["0x8000", "0x1021", "0xedb8", "0x4129"]):
            scores["crc"] += 5

        # Determine the winner
        best_match = max(scores, key=scores.get)

        # Fallback if no score is high enough
        if scores[best_match] < 3:
            return "unknown"

        return best_match

    def _count_loops(self):
        return len(re.findall(r"\bfor\b|\bwhile\b", self.source_code))

    def _count_functions(self):
        # Slightly improved function detection
        return len(
            re.findall(
                r"\b(int|void|float|double|char|long|short)\s+\w+\s*\(",
                self.source_code,
            )
        )

    def _detect_recursion(self):
        functions = re.findall(
            r"\b(int|void|float|double|char|long|short)\s+(\w+)\s*\(",
            self.source_code,
        )

        for _, func_name in functions:
            pattern = rf"\b{func_name}\s*\("
            occurrences = len(re.findall(pattern, self.source_code))
            if occurrences > 1:
                return True

        return False

    # ==========================================================
    # PROGRAM REPRESENTATION FOR CFG / BLOCK BUILDER
    # ==========================================================

    def get_program_representation(self) -> Dict:
        """
        Returns structured representation for downstream analysis.
        """
        return {
            "lines": self.cleaned_lines,
            "analysis": self.analysis,
        }