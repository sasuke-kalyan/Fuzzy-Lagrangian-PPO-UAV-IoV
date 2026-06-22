import pandas as pd

from data_loader import load_dataset

from graph_data import snapshot_at_timestamp

df = load_dataset()
snap = snapshot_at_timestamp(df)

# Reliability Score Calculation
df['Reliability_Score'] = (
    (df['Signal_Strength'] * 50)
    + (df['PDR'] * 0.5)
    - (df['Delay'] * 0.4)
)

print("\n======================================")
print(" Intelligent UAV Selection System ")
print("======================================\n")

# Find Best UAV Communication Records
best_connections = snap.loc[
    snap.groupby("Vehicle_ID")["Reliability_Score"].idxmax()
]

# Display Best UAV Selection
for index, row in best_connections.iterrows():

    print(f"Vehicle : {row['Vehicle_ID']}")
    print(f"Selected UAV : {row['UAV_ID']}")

    print(f"Signal Strength : {row['Signal_Strength']}")
    print(f"Delay : {row['Delay']}")
    print(f"PDR : {row['PDR']}")

    print(f"Reliability Score : {round(row['Reliability_Score'],2)}")

    print("--------------------------------------")

print("\nBest UAV Selection Completed Successfully!")