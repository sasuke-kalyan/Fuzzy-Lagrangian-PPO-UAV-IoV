from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


input_file = Path("results/comparison/all_scenarios_kpi_table.csv")
output_dir = Path("results/final_graphs")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)


def plot_metric(metric_name, output_name, ylabel):
    pivot = df.pivot(index="Scenario", columns="Algorithm", values=metric_name)

    pivot.plot(kind="bar", figsize=(10, 6))

    plt.xlabel("Scenario")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name} Comparison")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / output_name, dpi=300)
    plt.close()


plot_metric("Average Reward", "average_reward_comparison.png", "Average Reward")
plot_metric("Completed Tasks", "completed_tasks_comparison.png", "Completed Tasks")
plot_metric("Energy (J)", "energy_comparison.png", "Energy (J)")

print("Final comparison graphs generated successfully!")
print("Saved in: results/final_graphs")