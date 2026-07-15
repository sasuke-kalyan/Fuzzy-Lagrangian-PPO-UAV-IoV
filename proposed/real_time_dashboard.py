import pandas as pd

from data_loader import load_dataset
import matplotlib.pyplot as plt

from communication_model import AREA_SIZE
from graph_data import snapshot_at_timestamp

df = load_dataset()
timestamps = sorted(df["Timestamp"].unique())[:20]

plt.figure(figsize=(10, 10))

for t in timestamps:
    plt.clf()
    snap = snapshot_at_timestamp(df, int(t))

    # Draw one active link per vehicle (best-signal UAV) to avoid clutter.
    best_links = snap.loc[snap.groupby("Vehicle_ID")["Signal_Strength"].idxmax()]
    for _, row in best_links.iterrows():
        plt.plot(
            [row["Vehicle_X"], row["UAV_X"]],
            [row["Vehicle_Y"], row["UAV_Y"]],
            c="gray",
            alpha=0.45,
            linewidth=1.2,
        )

    for _, row in snap.drop_duplicates("Vehicle_ID").iterrows():
        plt.scatter(row["Vehicle_X"], row["Vehicle_Y"], s=80, c="tab:blue")
        plt.text(row["Vehicle_X"] + 8, row["Vehicle_Y"] + 8, row["Vehicle_ID"], fontsize=7)

    for _, row in snap.drop_duplicates("UAV_ID").iterrows():
        plt.scatter(row["UAV_X"], row["UAV_Y"], s=140, marker="^", c="tab:orange")
        plt.text(row["UAV_X"] + 8, row["UAV_Y"] + 8, row["UAV_ID"], fontsize=7)

    avg_signal = snap["Signal_Strength"].mean()
    quality = "Strong" if avg_signal >= 0.7 else "Medium" if avg_signal >= 0.4 else "Weak"

    plt.title(
        f"UAV-IoV Network (t={int(t)})\n"
        f"{snap['Vehicle_ID'].nunique()} vehicles, {snap['UAV_ID'].nunique()} UAVs | "
        f"Avg signal: {avg_signal:.2f} ({quality})"
    )
    plt.xlim(0, AREA_SIZE)
    plt.ylim(0, AREA_SIZE)
    plt.grid(True)
    plt.pause(0.4)

plt.close()
print("\nDashboard animation completed.")
