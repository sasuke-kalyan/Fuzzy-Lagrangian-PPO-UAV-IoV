from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "UAV_IOV" / "uav_iov_dataset.csv"

OUT_DIR = ROOT / "comparison_results" / "publication_cluster_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

SCENARIOS = {
    "urban_canyon": "Urban Canyon",
    "suburban_crossroads": "Suburban Crossroads",
    "emergency_response": "Emergency Response",
}

N_CLUSTERS = 6
COMM_RANGE = 350
SAMPLE_SIZE = 2500

for scenario_id, scenario_name in SCENARIOS.items():

    scenario_df = df[df["Scenario_ID"] == scenario_id].copy()

    vehicles = scenario_df[["Vehicle_X", "Vehicle_Y"]].drop_duplicates()

    if len(vehicles) > SAMPLE_SIZE:
        vehicles = vehicles.sample(SAMPLE_SIZE, random_state=42)

    X = vehicles[["Vehicle_X", "Vehicle_Y"]].values

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10
    )

    vehicles["Cluster"] = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster_id in range(N_CLUSTERS):
        cluster_points = vehicles[vehicles["Cluster"] == cluster_id]

        ax.scatter(
            cluster_points["Vehicle_X"],
            cluster_points["Vehicle_Y"],
            s=10,
            alpha=0.70,
            label=f"Cluster {cluster_id + 1}"
        )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="^",
        s=220,
        color="black",
        edgecolor="white",
        linewidth=1.2,
        label="UAV Position"
    )

    for i, (cx, cy) in enumerate(centers):
        circle = Circle(
            (cx, cy),
            COMM_RANGE,
            fill=False,
            linestyle="--",
            linewidth=1.4,
            alpha=0.65,
            color="black"
        )
        ax.add_patch(circle)

        ax.text(
            cx + 25,
            cy + 25,
            f"UAV-{i + 1}",
            fontsize=8,
            fontweight="bold"
        )

    ax.set_title(
        f"{scenario_name} UAV-Vehicle Cluster Map",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")

    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.35
    )

    ax.legend(
        fontsize=8,
        loc="upper right",
        frameon=True,
        ncol=2
    )

    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    out_path = OUT_DIR / f"{scenario_id}_publication_cluster_map.png"

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Saved: {out_path}")

print("\nPublication-style cluster maps generated successfully!")
print(f"Open folder: {OUT_DIR}")