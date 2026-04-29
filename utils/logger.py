import os
import json
from datetime import datetime


class Logger:
    """
    Lightweight structured logger for experiments.
    Synchronized with the ML Checkpointing Pipeline for Phase 1-3 tracking.
    """

    def __init__(self, log_dir="logs", verbose=True):
        # This finds the directory of 'logger.py', goes up one level to the root,
        # and creates 'logs' there.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(base_dir, log_dir)

        self.verbose = verbose
        os.makedirs(self.log_dir, exist_ok=True)

        if self.verbose:
            print(f"[DEBUG] Logger initialized at: {self.log_dir}")

    def log(self, message, level="INFO"):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def section(self, title):
        if self.verbose:
            print("\n" + "█" * 65)
            print(f"  {title.upper()}")
            print("█" * 65)

    # --------------------------------------------------
    # JSON Persistence
    # --------------------------------------------------

    def save_json(self, data, filename=None):
        """
        Saves experiment data. Standardizes filenames for Phase 3 sweeps.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"run_{timestamp}.json"

        # Ensure we don't double-nest path names
        if not filename.endswith(".json"):
            filename += ".json"

        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

        self.log(f"Data persisted to {filepath}", level="SAVE")
        return filepath

    # --------------------------------------------------
    # Metrics Summarization
    # --------------------------------------------------

    def print_metrics(self, metrics: dict, strategy_name="GENERIC"):
        """
        Outputs a clean table of results for the given strategy.
        """
        if not self.verbose:
            return

        print(f"\n>>> Results for Strategy: {strategy_name.upper()}")
        print("-" * 45)

        # Keys to exclude from summary to keep it clean
        ignore_keys = ["checkpoint_log", "memory_lines", "structural_metrics"]

        for key, value in metrics.items():
            if key in ignore_keys:
                continue

            # Format formatting based on type
            label = key.replace("_", " ").title()
            if isinstance(value, float):
                # Check if it's a ratio or percentage
                if "ratio" in key or "efficiency" in key:
                    print(f"{label:25s}: {value * 100:>8.2f}%")
                else:
                    print(f"{label:25s}: {value:>8.4f}")
            else:
                print(f"{label:25s}: {value:>8}")

        print("-" * 45)