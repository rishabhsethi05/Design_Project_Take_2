from execution.experiment_runner import (
    build_cfg_from_c,
    run_single_execution,
    run_failure_sweep
)

if __name__ == "__main__":

    c_file_path = "sample_programs/crc.c"

    # --- UPDATED PARAMETERS FOR VISIBLE FAILURES ---
    # We lower checkpoint cost so the ML isn't afraid to save,
    # and we drastically increase the failure rates to simulate
    # a highly unstable energy harvester (e.g., solar on a cloudy day).
    checkpoint_cost = 0.01  # Lowered to match your small execution times
    state_size_cost_factor = 0.001

    # Failure rates: 50.0 means 50 failures per second on average.
    # This will ensure we see actual crashes in your 0.01s program!
    stress_test_rates = [10.0, 50.0, 100.0, 200.0]

    # 1. BUILD CFG
    blocks, analysis, structural_metrics = build_cfg_from_c(
        c_file_path,
        verbose=True
    )

    # 2. DETAILED SINGLE EXECUTION (High Stress)
    print("\n==============================")
    print("DETAILED EXECUTION (λ = 50.0)")
    print("==============================")

    run_single_execution(
        blocks=blocks,
        structural_metrics=structural_metrics,
        failure_rate=50.0, # High stress
        checkpoint_cost=checkpoint_cost,
        state_size_cost_factor=state_size_cost_factor
    )

    # 3. STOCHASTIC FAILURE SWEEP
    print("\n==============================")
    print("STOCHASTIC FAILURE SWEEP")
    print("==============================")

    run_failure_sweep(
        blocks=blocks,
        structural_metrics=structural_metrics,
        failure_rates=stress_test_rates,
        trials_per_rate=20 # More trials for better mean/std-dev
    )