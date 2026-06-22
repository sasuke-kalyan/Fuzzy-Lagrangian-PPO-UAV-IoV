from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "UAV_IOV" / "uav_iov_dataset.csv"

OUT_DIR = ROOT / "comparison_results" / "research_delay_visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

SCENARIOS = {
    "urban_canyon": "Urban Canyon",
    "suburban_crossroads": "Suburban Crossroads",
    "emergency_response": "Emergency Response",
}

for scenario_id, scenario_name in SCENARIOS.items():
    data = df[df["Scenario_ID"] == scenario_id].copy()

    uav_order = sorted(data["UAV_ID"].unique())

    delay_data = [
        data[data["UAV_ID"] == uav]["Delay"].dropna().values
        for uav in uav_order
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    parts = ax.violinplot(
        delay_data,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )

    ax.boxplot(
        delay_data,
        widths=0.12,
        showfliers=False
    )

    ax.set_title(f"{scenario_name} Delay Distribution", fontsize=13)
    ax.set_xlabel("UAV ID", fontsize=14, fontweight="bold")
    ax.set_ylabel("Delay (ms)", fontsize=14, fontweight="bold")

    ax.set_xticks(range(1, len(uav_order) + 1))
    ax.set_xticklabels(uav_order)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout()

    out_path = OUT_DIR / f"{scenario_id}_delay_violin_boxplot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out_path}")

print("\nResearch-style delay violin/box plots generated successfully!")
print(f"Open folder: {OUT_DIR}")