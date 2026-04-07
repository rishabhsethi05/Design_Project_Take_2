import os
import json
from datetime import datetime


class Logger:
    """
    Lightweight structured logger for experiments.

    Features:
    - Console logging (optional)
    - JSON result dumping
    - Auto directory creation
    - Timestamped experiment runs
    """

    def __init__(self, log_dir="logs", verbose=True):
        self.log_dir = log_dir
        self.verbose = verbose

        os.makedirs(self.log_dir, exist_ok=True)

    # --------------------------------------------------
    # Console Logging
    # --------------------------------------------------

    def log(self, message):
        if self.verbose:
            print(f"[LOG] {message}")

    def section(self, title):
        if self.verbose:
            print("\n" + "=" * 60)
            print(title)
            print("=" * 60)

    # --------------------------------------------------
    # JSON Logging
    # --------------------------------------------------

    def save_json(self, data, filename=None):
        """
        Save structured data as JSON.
        """

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_{timestamp}.json"

        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

        if self.verbose:
            print(f"[LOG] Results saved to {filepath}")

        return filepath

    # --------------------------------------------------
    # Pretty Experiment Summary
    # --------------------------------------------------

    def print_metrics(self, metrics: dict):

        if not self.verbose:
            return

        print("\n--- Execution Metrics ---")

        for key, value in metrics.items():
            if key == "checkpoint_log":
                continue

            if isinstance(value, float):
                print(f"{key:25s}: {value:.4f}")
            else:
                print(f"{key:25s}: {value}")

        print("--------------------------")