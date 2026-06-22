import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("comparison_results")
CSV_PATH = BASE_DIR / "fair_qos_comparison.csv"
OUT_DIR = BASE_DIR / "final_graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

def plot_metric(metric, title, ylabel, filename, lower_is_better=True):
    plt.figure(figsize=(10, 6))

    for scenario in df["Scenario"].unique():
        sub = df[df["Scenario"] == scenario]
        plt.bar(
            sub["Scenario"] + "\n" + sub["Algorithm"],
            sub[metric]
        )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    note = "Lower is better" if lower_is_better else "Higher is better"
    plt.xlabel(note)

    plt.savefig(OUT_DIR / filename, dpi=300)
    plt.close()

plot_metric(
    "QoS Penalty",
    "QoS Penalty Comparison",
    "QoS Penalty",
    "qos_penalty_comparison.png",
    lower_is_better=True
)

plot_metric(
    "Average Delay (ms)",
    "Average Delay Comparison",
    "Delay (ms)",
    "average_delay_comparison.png",
    lower_is_better=True
)

plot_metric(
    "Average PDR (%)",
    "Packet Delivery Ratio Comparison",
    "PDR (%)",
    "pdr_comparison.png",
    lower_is_better=False
)

plot_metric(
    "Energy per Task",
    "Energy per Task Comparison",
    "Energy per Task",
    "energy_per_task_comparison.png",
    lower_is_better=True
)

plot_metric(
    "Average Signal",
    "Average Signal Strength Comparison",
    "Average Signal Strength",
    "signal_strength_comparison.png",
    lower_is_better=False
)

print("Final QoS graphs generated successfully!")
print(f"Saved in: {OUT_DIR}")