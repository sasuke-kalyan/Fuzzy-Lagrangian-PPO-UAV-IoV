from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


input_file = Path("results/comparison/all_scenarios_kpi_table.csv")
output_dir = Path("results/final_graphs")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

pivot = df.pivot(index="Scenario", columns="Algorithm", values="Energy per Task")

pivot.plot(kind="bar", figsize=(10, 6))

plt.xlabel("Scenario")
plt.ylabel("Energy per Task (J/task)")
plt.title("Energy per Task Comparison")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(output_dir / "energy_per_task_comparison.png", dpi=300)
plt.close()

print("Energy per task graph generated successfully!")
print("Saved at: results/final_graphs/energy_per_task_comparison.png")