import json
import os
import matplotlib.pyplot as plt


def generate_final_plot():
    log_path = r"C:\Users\Rishabh Sethi\PycharmProjects\Design Project Take 2\logs\comparative_results.json"

    if not os.path.exists(log_path):
        print("File not found! Please check the path.")
        return

    # In this run, your console showed these efficiencies:
    data = {
        "PERIODIC": 82.91,
        "ANALYTICAL": 84.67,
        "ML_ADAPTIVE": 84.64,
        "HYBRID": 84.65
    }

    plt.figure(figsize=(10, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    bars = plt.bar(data.keys(), data.values(), color=colors, edgecolor='black', alpha=0.8)

    plt.axhline(y=data["PERIODIC"], color='red', linestyle='--', label='Periodic Baseline')

    plt.title("Comparison of Checkpointing Strategies (λ = 5.0)", fontsize=14)
    plt.ylabel("System Efficiency (%)", fontsize=12)
    plt.ylim(80, 86)  # Zoomed in to show the significant 2% difference

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f"{yval}%", ha='center', va='bottom',
                 fontweight='bold')

    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.savefig("final_efficiency_comparison.png")
    print("Success! 'final_efficiency_comparison.png' created.")
    plt.show()


if __name__ == "__main__":
    generate_final_plot()