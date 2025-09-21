from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_results(results: Dict[str, Dict[str, float]]):
    """
    Generates and saves a static image of the benchmark results using Matplotlib,
    displaying grouped bar charts for each strategy.

    Args:
        results (Dict[str, Dict[str, float]]): A nested dictionary.
                                                Outer keys are model names.
                                                Inner keys are strategy names,
                                                and values are nDCG scores.
    """
    # Get all unique strategies and models
    all_strategies = set()
    all_models = sorted(list(results.keys()))
    for model_results in results.values():
        all_strategies.update(model_results.keys())

    strategies = sorted(list(all_strategies))

    x = np.arange(len(all_models))  # the label locations for the models
    width = 0.8 / len(strategies)  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, strategy_name in enumerate(strategies):
        # Calculate the offset for each bar within the group
        offset = i * width - (len(strategies) - 1) * width / 2

        # Gather scores for the current strategy across all models
        scores = [results[model].get(strategy_name, 0.0) for model in all_models]

        # Plot the bars
        rects = ax.bar(x + offset, scores, width, label=strategy_name)

        # Add score labels on top of each bar
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    # Add labels, title, and legend
    ax.set_ylabel("nDCG@10 Score", fontsize=12)
    ax.set_title("Long-Text Embedding Strategy Benchmark by Model", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Strategies")

    plt.tight_layout()
    plt.savefig("grouped_benchmark.png")
    print("Plot saved as grouped_benchmark.png")
    # You can choose to show the plot as well if you want
    # plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    sample_results = {
        "Model_A": {
            "Truncation": 0.854,
            "Chunk (Size: 512, Overlap: 64)": 0.791,
            "Chunk (Size: 384, Overlap: 64)": 0.923,
        },
        "Model_B": {
            "Truncation": 0.750,
            "Chunk (Size: 512, Overlap: 64)": 0.820,
            "Chunk (Size: 384, Overlap: 64)": 0.880,
        },
        "Model_C": {
            "Truncation": 0.650,
            "Chunk (Size: 512, Overlap: 64)": 0.720,
            "Chunk (Size: 384, Overlap: 64)": 0.810,
        },
    }

    plot_results(sample_results)
