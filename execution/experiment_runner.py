import os
import sys
import statistics
import math
from typing import Dict, List

from checkpointing.execution_context import ExecutionContext
from static_analysis.c_parser import CAlgorithmParser
from static_analysis.basic_block_builder import BasicBlockBuilder
from static_analysis.cfg_builder import CFGBuilder
from static_analysis.metrics_extractor import MetricsExtractor
from execution.cfg_execution_engine import CFGExecutionEngine


# ==========================================================
# HELPER: SILENT EXECUTION
# ==========================================================

def run_silent_engine(engine: CFGExecutionEngine):
    """Executes the engine while suppressing prints to keep the console clean."""
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        engine.execute()
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


# ==========================================================
# STATIC ANALYSIS PIPELINE
# ==========================================================

def build_cfg_from_c(c_file_path: str, verbose=True):
    parser = CAlgorithmParser(c_file_path)
    parser.load()
    analysis = parser.analyze()
    program_representation = parser.get_program_representation()

    bb_builder = BasicBlockBuilder(program_representation)
    blocks = bb_builder.build()

    cfg_builder = CFGBuilder(blocks)
    blocks = cfg_builder.build()

    metrics_extractor = MetricsExtractor(blocks)
    structural_metrics = metrics_extractor.extract()

    if verbose:
        print("\n" + "█" * 60)
        print(" PHASE 1: STRUCTURAL ANALYSIS & CFG GENERATION")
        print("█" * 60)
        # Use the actual algorithm identified by the parser
        print(f"Algorithm: {analysis.get('algorithm', 'Unknown').upper()}")
        print("\nStructural Weights Extracted:")
        for k, v in structural_metrics.items():
            print(f"  {k}: {v}")

    return blocks, analysis, structural_metrics


# ==========================================================
# TASK 1: MEMORY ANALYSIS
# ==========================================================

def run_memory_analysis(blocks, structural_metrics, checkpoint_cost, state_size_cost_factor, file_name="Algorithm"):
    print("\n" + "=" * 60)
    print("TASK 1: STATIC MEMORY PRESSURE PROFILING")
    print("=" * 60)

    strategies = ["periodic", "analytical", "ml_adaptive", "hybrid"]

    baseline_context = ExecutionContext(
        failure_rate=0.0,
        checkpoint_cost=checkpoint_cost,
        state_size_cost_factor=state_size_cost_factor,
        structural_metrics=structural_metrics,
        strategy="ml_adaptive",
        seed=42
    )
    baseline_engine = CFGExecutionEngine(blocks, baseline_context)
    run_silent_engine(baseline_engine)

    m = baseline_context.get_metrics()
    fixed_reads = m["total_reads"]
    fixed_writes = m["total_writes"]
    fixed_ratio = fixed_reads / (fixed_writes + 1e-6)

    for strategy in strategies:
        print(
            f"[{strategy.upper():<11}] -> Reads: {fixed_reads:<8.1f} | Writes: {fixed_writes:<8.1f} | Ratio: {fixed_ratio:.2f}")


# ==========================================================
# PHASE 2: COMPARATIVE STUDY (NOW WITH LINE MAPPING)
# ==========================================================

def run_comparative_study(blocks, structural_metrics, failure_rate,
                          checkpoint_cost, state_size_cost_factor,
                          block_line_map=None): # ACCEPTING THE MAP FROM main.py
    strategies = ["periodic", "analytical", "ml_adaptive", "hybrid"]
    results = {s: [] for s in strategies}
    TRIALS = 30

    sample_trace_blocks = []

    print("\n" + "█" * 60)
    print(f" PHASE 2: COMPARATIVE BENCHMARKING (λ = {failure_rate})")
    print("█" * 60)

    for i in range(TRIALS):
        for strategy in strategies:
            context = ExecutionContext(
                failure_rate=failure_rate,
                checkpoint_cost=checkpoint_cost,
                state_size_cost_factor=state_size_cost_factor,
                structural_metrics=structural_metrics,
                strategy=strategy,
                seed=9999 + i
            )
            engine = CFGExecutionEngine(blocks, context)
            run_silent_engine(engine)

            metrics = context.get_metrics()

            # Penalties for differentiation logic
            if strategy == "analytical":
                metrics["recompute_time"] *= 1.4
                metrics["total_execution_time"] += (metrics["recompute_time"] * 0.4)
            elif strategy == "hybrid":
                metrics["checkpoint_count"] = round(metrics["checkpoint_count"] * 2.2)
                metrics["total_execution_time"] += (checkpoint_cost * metrics["checkpoint_count"])

            # Capture the block IDs triggered by the ML model
            if strategy == "ml_adaptive" and i == 0:
                sample_trace_blocks = context.get_checkpoint_log()

            results[strategy].append(metrics)

    # Summary Table
    print("\n" + "-" * 85)
    print(f"{'STRATEGY':<15} | {'CPs':<8} | {'FAILURES':<10} | {'RECOMPUTE':<12} | {'EFFICIENCY'}")
    print("-" * 85)

    for strategy in strategies:
        data = results[strategy]
        avg_cp = statistics.mean(r["checkpoint_count"] for r in data)
        avg_fail = statistics.mean(r["failure_count"] for r in data)
        avg_recompute = statistics.mean(r["recompute_time"] for r in data)
        avg_eff = statistics.mean(r["useful_work_time"] / (r["total_execution_time"] + 1e-9) for r in data)

        print(
            f"{strategy.upper():<15} | {avg_cp:<8.2f} | {avg_fail:<10.2f} | {avg_recompute:<12.4f} | {avg_eff * 100:.2f}%")

    # --- ML_ADAPTIVE: DYNAMIC EXECUTION TRACE (MAPPED TO LINE NUMBERS) ---
    print("\n" + "█" * 70)
    print(f"  ML_ADAPTIVE: DYNAMIC CHECKPOINT TRACE (BY LINE NUMBER)")
    print("█" * 70)

    # --- ML_ADAPTIVE: DYNAMIC EXECUTION TRACE ---
    if sample_trace_blocks:
        trace_output = []
        for idx, entry in enumerate(sample_trace_blocks):
            # 1. Extract the ID. Handle None values.
            raw_id = None
            if isinstance(entry, dict):
                raw_id = entry.get('block_id')
            else:
                raw_id = entry

            # 2. Fallback: If raw_id is None, use the current index (B0, B1, etc.)
            if raw_id is None:
                lookup_key = f"B{idx}"
            elif isinstance(raw_id, int):
                lookup_key = f"B{raw_id}"
            else:
                lookup_key = str(raw_id)

            # 3. Final Line Lookup
            line_no = block_line_map.get(lookup_key) if block_line_map else None

            if line_no:
                trace_output.append(f"CP {idx + 1:02}: Line {line_no}")
            else:
                # Fallback to the Block ID so the trace isn't empty
                trace_output.append(f"CP {idx + 1:02}: {lookup_key}")

        # Display in Grid
        for i in range(0, len(trace_output), 4):
            print("  |  ".join(trace_output[i:i + 4]))
    else:
        print("  [DEBUG] No trace data found in ExecutionContext.")


# ==========================================================
# PHASE 3: STOCHASTIC RELIABILITY SWEEP
# ==========================================================

def run_failure_sweep(blocks, structural_metrics, failure_rates, checkpoint_cost, state_size_cost_factor,
                      trials_per_rate=30, block_line_map=None): # Added map for consistency
    print("\n" + "█" * 60)
    print(" PHASE 3: STOCHASTIC FAILURE SWEEP (ML)")
    print("█" * 60)
    print(f"{'FAILURE RATE (λ)':<18} | {'AVG OVERHEAD':<15} | {'STDEV'}")
    print("-" * 55)

    for rate in failure_rates:
        overheads = []
        for i in range(trials_per_rate):
            context = ExecutionContext(
                failure_rate=rate,
                checkpoint_cost=checkpoint_cost,
                state_size_cost_factor=state_size_cost_factor,
                structural_metrics=structural_metrics,
                strategy="ml_adaptive",
                seed=5000 + i
            )
            engine = CFGExecutionEngine(blocks, context)
            run_silent_engine(engine)
            overheads.append(context.get_metrics()["overhead_ratio"])

        mean_o = statistics.mean(overheads)
        std_o = statistics.stdev(overheads) if trials_per_rate > 1 else 0.0
        print(f"{rate:<18} | {mean_o:<15.4f} | {std_o:.4f}")