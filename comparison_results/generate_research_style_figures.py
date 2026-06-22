from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "UAV_IOV" / "uav_iov_dataset.csv"

OUT_DIR = ROOT / "comparison_results" / "research_style_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

SCENARIOS = {
    "urban_canyon": "Urban Canyon",
    "suburban_crossroads": "Suburban Crossroads",
    "emergency_response": "Emergency Response",
}

COMM_RANGE = 500


def smooth_density(values, bins=80):
    values = pd.Series(values).dropna().astype(float)

    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    kernel = np.ones(5) / 5
    smooth = np.convolve(counts, kernel, mode="same")

    return centers, smooth


def plot_cluster_and_signal(scenario_id, scenario_name):
    data = df[df["Scenario_ID"] == scenario_id].copy()

    # Use limited rows so graph is clean
    data = data.sample(
        n=min(4000, len(data)),
        random_state=42
    )

    uavs = data[["UAV_ID", "UAV_X", "UAV_Y"]].drop_duplicates("UAV_ID")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]

    for uav_id in sorted(data["UAV_ID"].unique()):
        sub = data[data["UAV_ID"] == uav_id]

        ax.scatter(
            sub["Vehicle_X"],
            sub["Vehicle_Y"],
            s=4,
            alpha=0.55,
            label=uav_id
        )

    ax.scatter(
        uavs["UAV_X"],
        uavs["UAV_Y"],
        s=90,
        marker="^",
        color="black",
        label="UAVs"
    )

    for _, row in uavs.iterrows():
        circle = Circle(
            (row["UAV_X"], row["UAV_Y"]),
            COMM_RANGE,
            fill=False,
            linestyle="--",
            linewidth=0.8,
            alpha=0.35
        )
        ax.add_patch(circle)

    ax.set_title("(a) UAV-Vehicle coverage clustering", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    ax = axes[1]

    for uav_id in sorted(data["UAV_ID"].unique()):
        sub = data[data["UAV_ID"] == uav_id]
        x, y = smooth_density(sub["Signal_Strength"], bins=60)

        ax.plot(
            x,
            y,
            linewidth=1.6,
            label=uav_id
        )

    ax.set_title("(b) Signal strength distribution", fontsize=10)
    ax.set_xlabel("Signal strength")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"{scenario_name} Communication Distribution Analysis", fontsize=11)

    plt.tight_layout()

    out_path = OUT_DIR / f"{scenario_id}_cluster_signal_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {out_path}")


def plot_delay_distribution(scenario_id, scenario_name):
    data = df[df["Scenario_ID"] == scenario_id].copy()

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    for uav_id in sorted(data["UAV_ID"].unique()):
        sub = data[data["UAV_ID"] == uav_id]
        x, y = smooth_density(sub["Delay"], bins=70)

        ax.plot(
            x,
            y,
            linewidth=1.6,
            label=uav_id
        )

    ax.set_title(f"{scenario_name} Delay Distribution", fontsize=11)
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
    ax.legend(fontsize=7, loc="best")

    plt.tight_layout()

    out_path = OUT_DIR / f"{scenario_id}_delay_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {out_path}")


def plot_vehicle_density(scenario_id, scenario_name):
    data = df[df["Scenario_ID"] == scenario_id].copy()

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    h = ax.hist2d(
        data["Vehicle_X"],
        data["Vehicle_Y"],
        bins=60
    )

    plt.colorbar(h[3], ax=ax, label="Vehicle density")

    ax.set_title(f"{scenario_name} Vehicle Density Map", fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)

    plt.tight_layout()

    out_path = OUT_DIR / f"{scenario_id}_vehicle_density_map.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {out_path}")


for scenario_id, scenario_name in SCENARIOS.items():
    plot_cluster_and_signal(scenario_id, scenario_name)
    plot_delay_distribution(scenario_id, scenario_name)
    plot_vehicle_density(scenario_id, scenario_name)

print("\nResearch-style figures generated successfully!")
print(f"Open folder: {OUT_DIR}")