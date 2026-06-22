from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "UAV_IOV" / "uav_iov_dataset.csv"

OUT_DIR = ROOT / "comparison_results" / "binned_signal_distance_graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

SCENARIOS = {
    "urban_canyon": "Urban Canyon",
    "suburban_crossroads": "Suburban Crossroads",
    "emergency_response": "Emergency Response"
}

BIN_SIZE = 100

for scenario_id, scenario_name in SCENARIOS.items():

    scenario_df = df[df["Scenario_ID"] == scenario_id].copy()

    max_distance = scenario_df["Distance"].max()
    bins = np.arange(0, max_distance + BIN_SIZE, BIN_SIZE)

    scenario_df["Distance_Bin"] = pd.cut(
        scenario_df["Distance"],
        bins=bins,
        include_lowest=True
    )

    grouped = scenario_df.groupby("Distance_Bin", observed=True).agg(
        Mean_Distance=("Distance", "mean"),
        Mean_Signal=("Signal_Strength", "mean"),
        Std_Signal=("Signal_Strength", "std")
    ).dropna()

    plt.figure(figsize=(8, 5.5))

    plt.plot(
        grouped["Mean_Distance"],
        grouped["Mean_Signal"],
        marker="o",
        linewidth=2.5,
        markersize=5,
        color="black",
        label="Mean Signal Strength"
    )

    plt.fill_between(
        grouped["Mean_Distance"],
        grouped["Mean_Signal"] - grouped["Std_Signal"],
        grouped["Mean_Signal"] + grouped["Std_Signal"],
        alpha=0.20,
        label="±1 Std. Dev."
    )

    plt.title(
        f"{scenario_name}: Average Signal Strength vs Distance",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Distance (m)", fontsize=13, fontweight="bold")
    plt.ylabel("Average Signal Strength", fontsize=13, fontweight="bold")

    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(fontsize=10, frameon=True)

    plt.tight_layout()

    out_path = OUT_DIR / f"{scenario_id}_binned_signal_distance.png"

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Saved: {out_path}")

print("\nBinned signal-distance graphs generated successfully!")
print(f"Open folder: {OUT_DIR}")