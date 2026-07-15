import pandas as pd

from data_loader import load_dataset

from graph_data import snapshot_at_timestamp

df = load_dataset()
snap = snapshot_at_timestamp(df)

print("\nBEST UAV SELECTION RESULTS\n")

for vehicle in snap["Vehicle_ID"].unique():
    vehicle_data = snap[snap["Vehicle_ID"] == vehicle]
    best_row = vehicle_data.loc[vehicle_data["Signal_Strength"].idxmax()]

    print(f"Vehicle: {vehicle}")
    print(f"Best UAV: {best_row['UAV_ID']}")
    print(f"Signal Strength: {best_row['Signal_Strength']}")
    print(f"Delay: {best_row['Delay']}")
    print(f"PDR: {best_row['PDR']}")
    print("-" * 40)
