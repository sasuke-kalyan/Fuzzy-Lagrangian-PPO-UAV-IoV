from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "UAV_IOV" / "uav_iov_dataset.csv"

OUT_DIR = ROOT / "comparison_results" / "paper_style_spatial_graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

SCENARIOS = [
    "urban_canyon",
    "suburban_crossroads",
    "emergency_response"
]

for scenario in SCENARIOS:

    scenario_df = df[df["Scenario_ID"] == scenario]

    # ==========================
    # 1. UAV-Vehicle Coverage Map
    # ==========================

    plt.figure(figsize=(8, 6))

    plt.scatter(
        scenario_df["Vehicle_X"],
        scenario_df["Vehicle_Y"],
        s=10,
        alpha=0.6,
        label="Vehicles"
    )

    uavs = scenario_df[
        ["UAV_ID", "UAV_X", "UAV_Y"]
    ].drop_duplicates()

    plt.scatter(
        uavs["UAV_X"],
        uavs["UAV_Y"],
        s=150,
        marker="^",
        label="UAVs"
    )

    plt.title(
        f"{scenario.replace('_',' ').title()} Coverage Map"
    )
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{scenario}_coverage_map.png",
        dpi=300
    )

    plt.close()

    # ==========================
    # 2. Distance vs Signal
    # ==========================

    plt.figure(figsize=(7, 5))

    plt.scatter(
        scenario_df["Distance"],
        scenario_df["Signal_Strength"],
        alpha=0.35,
        s=12
    )

    plt.title(
        f"{scenario.replace('_',' ').title()} Distance vs Signal"
    )

    plt.xlabel("Distance (m)")
    plt.ylabel("Signal Strength")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{scenario}_distance_signal.png",
        dpi=300
    )

    plt.close()

    # ==========================
    # 3. Distance vs Delay
    # ==========================

    plt.figure(figsize=(7, 5))

    plt.scatter(
        scenario_df["Distance"],
        scenario_df["Delay"],
        alpha=0.35,
        s=12
    )

    plt.title(
        f"{scenario.replace('_',' ').title()} Distance vs Delay"
    )

    plt.xlabel("Distance (m)")
    plt.ylabel("Delay (ms)")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{scenario}_distance_delay.png",
        dpi=300
    )

    plt.close()

print("\nAll spatial graphs generated successfully!")
print(f"Open folder: {OUT_DIR}")