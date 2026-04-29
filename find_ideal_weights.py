import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from execution.experiment_runner import build_cfg_from_c
from checkpointing.execution_context import ExecutionContext
from execution.cfg_execution_engine import CFGExecutionEngine
import seaborn as sns

def run_weight_optimization(c_file, trials=1000):
    blocks, _, metrics = build_cfg_from_c(c_file, verbose=False)
    results = []

    print(f"🚀 Running {trials} weight combinations...")
    for _ in tqdm(range(trials)):
        # We vary the importance of loops vs complexity
        w_loop = random.uniform(0.1, 5.0)
        w_comp = random.uniform(0.1, 5.0)

        # Simulate an environment
        failure_rate = 5.0
        cp_cost = 0.001

        # We pass these custom weights to the ML engine
        # (Assuming your ExecutionContext/Engine can accept 'custom_weights')
        context = ExecutionContext(
            failure_rate=failure_rate,
            checkpoint_cost=cp_cost,
            structural_metrics=metrics,
            strategy="ml_adaptive",
            custom_weights={'loop': w_loop, 'complexity': w_comp}
        )

        engine = CFGExecutionEngine(blocks, context)
        engine.execute()
        m = context.get_metrics()

        # Calculate Total Cost: (CPs * Cost) + Recompute Time
        total_cost = (m['checkpoint_count'] * cp_cost) + m['recompute_time']

        results.append({
            'w_loop': w_loop,
            'w_complexity': w_comp,
            'total_cost': total_cost,
            'efficiency': m['overhead_ratio']
        })

    df = pd.DataFrame(results)
    df.to_csv("weight_optimization_results.csv", index=False)
    print("\n✅ Results saved to weight_optimization_results.csv")
    return df


if __name__ == "__main__":
    # 1. Generate the data
    df_results = run_weight_optimization("input_programs/sample_crc.c", trials=2000)

    # 2. Immediately generate the Heatmap (Paste your code here)
    print("📊 Generating Heatmap...")
    df_results['w_loop_r'] = df_results['w_loop'].round(1)
    df_results['w_comp_r'] = df_results['w_complexity'].round(1)

    pivot = df_results.pivot_table(index='w_loop_r', columns='w_comp_r', values='total_cost', aggfunc='mean')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlGnBu", annot=False)
    plt.title("Heatmap: Total Cost vs. Structural Weights")
    plt.xlabel("Weight (Complexity)")
    plt.ylabel("Weight (Loops)")
    plt.savefig("ideal_weights_heatmap.png")
    plt.show()  # This will open the window so you can see it immediately
    print("🔥 Heatmap saved as 'ideal_weights_heatmap.png'")