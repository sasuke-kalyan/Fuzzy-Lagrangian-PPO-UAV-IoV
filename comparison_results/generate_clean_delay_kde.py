from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "UAV_IOV" / "uav_iov_dataset.csv"

OUT_DIR = ROOT / "comparison_results" / "clean_delay_kde_graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

SCENARIOS = {
    "urban_canyon": "Urban Canyon",
    "suburban_crossroads": "Suburban Crossroads",
    "emergency_response": "Emergency Response"
}

for scenario_id, scenario_name in SCENARIOS.items():

    scenario_df = df[df["Scenario_ID"] == scenario_id].copy()

    delays = scenario_df["Delay"].dropna().values

    kde = gaussian_kde(delays)

    x_range = np.linspace(
        delays.min(),
        delays.max(),
        600
    )

    density = kde(x_range)

    plt.figure(figsize=(8, 5))

    plt.plot(
        x_range,
        density,
        linewidth=2.8,
        color="black",
        label="Delay Density"
    )

    plt.fill_between(
        x_range,
        density,
        alpha=0.25
    )

    mean_delay = delays.mean()
    median_delay = np.median(delays)

    plt.axvline(
        mean_delay,
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_delay:.2f} ms"
    )

    plt.axvline(
        median_delay,
        linestyle=":",
        linewidth=2,
        label=f"Median = {median_delay:.2f} ms"
    )

    plt.title(
        f"{scenario_name} Delay Density Distribution",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(
        "Delay (ms)",
        fontsize=13,
        fontweight="bold"
    )

    plt.ylabel(
        "Density",
        fontsize=13,
        fontweight="bold"
    )

    plt.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )

    plt.legend(
        fontsize=10,
        frameon=True
    )

    plt.tight_layout()

    out_path = OUT_DIR / f"{scenario_id}_clean_delay_kde.png"

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Saved: {out_path}")

print("\nClean delay KDE graphs generated successfully!")
print(f"Open folder: {OUT_DIR}")