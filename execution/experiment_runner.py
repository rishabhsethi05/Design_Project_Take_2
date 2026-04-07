import sys
import statistics
import random

from checkpointing.execution_context import ExecutionContext
from static_analysis.c_parser import CAlgorithmParser
from static_analysis.basic_block_builder import BasicBlockBuilder
from static_analysis.cfg_builder import CFGBuilder
from static_analysis.metrics_extractor import MetricsExtractor
from execution.cfg_execution_engine import CFGExecutionEngine


# ==========================================================
# STATIC ANALYSIS PIPELINE
# ==========================================================

def build_cfg_from_c(c_file_path: str, verbose=True):

    parser = CAlgorithmParser(c_file_path)
    parser.load()
    analysis = parser.analyze()
    program_representation = parser.get_program_representation()

    if analysis["algorithm"] == "unknown":
        raise ValueError("Could not detect algorithm type from C file.")

    bb_builder = BasicBlockBuilder(program_representation)
    blocks = bb_builder.build()

    cfg_builder = CFGBuilder(blocks)
    blocks = cfg_builder.build()

    metrics_extractor = MetricsExtractor(blocks)
    structural_metrics = metrics_extractor.extract()

    if verbose:
        print("\n==============================")
        print("STATIC ANALYSIS SUMMARY")
        print("==============================")
        print(analysis)
        print("\nStructural Metrics:")
        for k, v in structural_metrics.items():
            print(f"{k}: {v}")

    return blocks, analysis, structural_metrics


# ==========================================================
# SINGLE EXECUTION RUN (DETAILED / OPTIONAL VERBOSE)
# ==========================================================

def run_single_execution(
    blocks,
    structural_metrics,
    failure_rate,
    checkpoint_cost,
    state_size_cost_factor,
    verbose=True
):

    context = ExecutionContext(
        failure_rate=failure_rate,
        checkpoint_cost=checkpoint_cost,
        state_size_cost_factor=state_size_cost_factor,
        structural_metrics=structural_metrics,
        strategy="ml_adaptive" # Explicitly calling our ML strategy
    )

    engine = CFGExecutionEngine(blocks, context)
    engine.execute()

    metrics = context.get_metrics()

    if verbose:
        print("\nExecution Summary")
        print("------------------------------")
        print(f"Failure Rate (λ): {failure_rate}")
        print(f"Total Failures: {metrics['failure_count']}")
        print(f"Total Checkpoints: {metrics['checkpoint_count']}")

        print("\n--- Time Breakdown ---")
        print(f"Useful Work Time: {metrics['useful_work_time']:.4f}")
        print(f"Recomputation Time: {metrics['recompute_time']:.4f}")
        print(f"Checkpoint Overhead Time: {metrics['checkpoint_time']:.4f}")

        print("\n--- Execution Metrics ---")
        # FIXED: Changed 'baseline_time' to 'useful_work_time'
        print(f"Baseline Execution Time: {metrics['useful_work_time']:.4f}")
        print(f"Total Execution Time: {metrics['total_execution_time']:.4f}")
        print(f"Overhead Ratio: {metrics['overhead_ratio']:.6f}")

        checkpoint_log = metrics.get("checkpoint_log", [])

        if checkpoint_log:
            print("\n==============================")
            print("CHECKPOINT PLACEMENT LOG")
            print("==============================")

            for i, cp in enumerate(checkpoint_log, start=1):
                print(f"\nCheckpoint {i}")
                print(f"  Event Type: {cp.get('event_type')}")
                # FIXED: Updated keys to match new ExecutionContext log format
                print(f"  Progress At Checkpoint: {cp.get('progress', 0):.4f}")
                print(f"  Checkpoint Cost: {cp.get('cost', 0):.4f}")
        else:
            print("\nNo checkpoints were placed.")

    return metrics

# ==========================================================
# MULTI-TRIAL EXPERIMENT (NO VERBOSE)
# ==========================================================

def run_trials(blocks, structural_metrics, failure_rate, trials=20): # Increase trials to 20
    overheads = []
    failures = []
    checkpoints = []

    for i in range(trials):
        # We pass a dynamic seed based on the trial index
        context = ExecutionContext(
            failure_rate=failure_rate,
            checkpoint_cost=5,
            state_size_cost_factor=0.001,
            structural_metrics=structural_metrics,
            seed=random.randint(0, 1000000) # FIX: Real randomness
        )

        engine = CFGExecutionEngine(blocks, context)
        engine.execute()
        metrics = context.get_metrics()

        overheads.append(metrics["overhead_ratio"])
        failures.append(metrics["failure_count"])
        checkpoints.append(metrics["checkpoint_count"])

    return {
        "failure_rate": failure_rate,
        "mean_overhead": statistics.mean(overheads),
        "std_overhead": statistics.stdev(overheads), # Now this will be > 0
        "mean_failures": statistics.mean(failures),
        "mean_checkpoints": statistics.mean(checkpoints)
    }


# ==========================================================
# PARAMETER SWEEP
# ==========================================================

def run_failure_sweep(
    blocks,
    structural_metrics,
    failure_rates,
    trials_per_rate=10
):

    results = []

    for rate in failure_rates:

        summary = run_trials(
            blocks,
            structural_metrics,
            failure_rate=rate,
            trials=trials_per_rate
        )

        results.append(summary)

        print("\n--------------------------------")
        print(f"λ = {rate}")
        print(f"Mean Overhead: {summary['mean_overhead']:.4f} "
              f"+/- {summary['std_overhead']:.4f}")
        print(f"Mean Failures: {summary['mean_failures']:.2f}")
        print(f"Mean Checkpoints: {summary['mean_checkpoints']:.2f}")

    return results


# ==========================================================
# MAIN ENTRY
# ==========================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python experiment_runner.py <path_to_c_file>")
        sys.exit(1)

    c_file_path = sys.argv[1]

    # 1️⃣ Static Analysis (done once)
    blocks, analysis, structural_metrics = build_cfg_from_c(
        c_file_path,
        verbose=True
    )

    # 2️⃣ Failure Rate Sweep
    failure_rates = [0.005, 0.01, 0.02, 0.03]

    print("\n==============================")
    print("STOCHASTIC FAILURE SWEEP")
    print("==============================")

    run_failure_sweep(
        blocks,
        structural_metrics,
        failure_rates,
        trials_per_rate=10
    )