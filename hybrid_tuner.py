import random
import statistics
import sys
import os

# Ensure project root is in path for imports to resolve correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution.cfg_execution_engine import CFGExecutionEngine
from checkpointing.execution_context import ExecutionContext

from static_analysis.c_parser import CAlgorithmParser
from static_analysis.basic_block_builder import BasicBlockBuilder
from static_analysis.cfg_builder import CFGBuilder
from static_analysis.metrics_extractor import MetricsExtractor


# ==========================================================
# BUILD CFG PIPELINE
# ==========================================================

def build_cfg(c_file_path):
    """
    Orchestrates the static analysis pipeline:
    C Source -> Blocks -> CFG -> Structural Metrics.
    """
    parser = CAlgorithmParser(c_file_path)
    parser.load()
    parser.analyze()

    program = parser.get_program_representation()

    bb_builder = BasicBlockBuilder(program)
    blocks = bb_builder.build()

    cfg_builder = CFGBuilder(blocks)
    blocks = cfg_builder.build()

    metrics_extractor = MetricsExtractor(blocks)
    structural_metrics = metrics_extractor.extract()

    return blocks, structural_metrics


# ==========================================================
# HYBRID TUNER (2026 CALIBRATED VERSION)
# ==========================================================

def tune_hybrid_parameters(blocks, structural_metrics, trials_per_setting=50):
    """
    Performs a Grid Search over Alpha (historical weight) and Threshold (risk tolerance).
    Optimized for high-efficiency sub-second CRC algorithms.
    """
    print("\n" + "█" * 75)
    print("  HYBRID MODEL PARAMETER TUNING: SYSTEM CALIBRATION")
    print("█" * 75)

    # Search space optimized for sensitivity in CRC inner loops
    alpha_values = [0.35, 0.45, 0.55]
    threshold_values = [0.25, 0.35, 0.45, 0.55]

    best_config = None
    best_score = -float('inf')

    # Fallback tracker for raw performance
    best_raw_config = None
    best_raw_eff = -1

    BASE_SEED = 1000

    for alpha in alpha_values:
        for threshold in threshold_values:
            efficiencies = []
            failures = []
            checkpoints = []

            for t in range(trials_per_setting):
                # Setup context with candidate parameters
                context = ExecutionContext(
                    failure_rate=4.5,            # Moderate-High Stress
                    checkpoint_cost=0.01,        # 10ms save penalty
                    state_size_cost_factor=0.001,
                    structural_metrics=structural_metrics,
                    strategy="hybrid",
                    seed=BASE_SEED + t           # Deterministic seeds for valid comparison
                )

                context.HYBRID_ALPHA = alpha
                context.HYBRID_THRESHOLD = threshold

                engine = CFGExecutionEngine(blocks, context)

                # Silence engine output during tuning to prevent terminal flooding
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                try:
                    engine.execute()
                finally:
                    sys.stdout.close()
                    sys.stdout = original_stdout

                metrics = context.get_metrics()
                work = metrics["useful_work_time"]
                total = metrics["total_execution_time"]

                if total > 0:
                    efficiencies.append(work / total)

                failures.append(metrics["failure_count"])
                checkpoints.append(metrics["checkpoint_count"])

            if not efficiencies:
                continue

            # Compute Aggregate Metrics for this Setting
            avg_eff = statistics.mean(efficiencies)
            std_eff = statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0
            avg_fail = statistics.mean(failures)
            avg_cp = statistics.mean(checkpoints)

            # --- BEST RAW TRACKING ---
            if avg_eff > best_raw_eff:
                best_raw_eff = avg_eff
                best_raw_config = {
                    "alpha": alpha, "threshold": threshold, "efficiency": avg_eff,
                    "std": std_eff, "failures": avg_fail, "checkpoints": avg_cp
                }

            # --- VALIDATION FILTERS ---
            # Exclude configs that never checkpoint or thrash too much
            if avg_cp < 0.5 or avg_cp > 35:
                continue

            # --- REFINED SCORE FUNCTION ---
            # We prioritize Efficiency (avg_eff), but penalize volatility (std)
            # and recompute triggers (avg_fail).
            score = (
                avg_eff
                - (0.20 * std_eff)   # Volatility penalty
                - (0.12 * avg_fail)  # Failure/Recompute penalty
                - (0.001 * avg_cp)   # Overhead penalty (minimal)
            )

            print(
                f"Alpha={alpha:.2f} | Thr={threshold:.2f} | "
                f"Eff={avg_eff:>7.2%} | Fail={avg_fail:>4.1f} | "
                f"CP={avg_cp:>4.1f} | Score={score:>7.4f}"
            )

            if score > best_score:
                best_score = score
                best_config = {
                    "alpha": alpha,
                    "threshold": threshold,
                    "efficiency": avg_eff,
                    "std": std_eff,
                    "failures": avg_fail,
                    "checkpoints": avg_cp
                }

    print("\n" + "─" * 75)

    # --- FINAL OUTPUT SELECTION ---
    if best_config is None:
        print("⚠️ No config passed strict filters. Reverting to best raw efficiency.")
        best_config = best_raw_config

    if best_config:
        print("🎯 OPTIMAL HYBRID CONFIGURATION:")
        print(f"  > Alpha (EMA Weight):    {best_config['alpha']}")
        print(f"  > Risk Threshold:        {best_config['threshold']}")
        print(f"  > Expected Efficiency:   {best_config['efficiency']:.2%}")
        print(f"  > Avg Checkpoints:       {best_config['checkpoints']:.1f}")
        print(f"  > Reliability Score:     {best_score if best_config == best_config else 'N/A'}")
    else:
        print("❌ Tuning failed. Check engine connectivity and C source parsing.")

    print("─" * 75)
    return best_config


# ==========================================================
# EXECUTION ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    # Point to your CRC source code
    c_file_path = "sample_programs/crc.c"

    if not os.path.exists(c_file_path):
        print(f"Error: {c_file_path} not found. Please check file path.")
        sys.exit(1)

    print(f"Initializing Analysis for: {c_file_path}")
    blocks, structural_metrics = build_cfg(c_file_path)

    print("Starting Grid Search...")
    tune_hybrid_parameters(blocks, structural_metrics, trials_per_setting=50)