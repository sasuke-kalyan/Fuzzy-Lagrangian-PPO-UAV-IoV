import pandas as pd

from data_loader import load_dataset

# Load dataset
df = load_dataset()

print("\n======================================")
print(" Multi-UAV Coordination System ")
print("======================================\n")

# Reliability Score
df['Reliability_Score'] = (
    (df['Signal_Strength'] * 50)
    + (df['PDR'] * 0.5)
    - (df['Delay'] * 0.4)
)

# UAV Performance Summary
uav_summary = df.groupby('UAV_ID').agg({

    'Signal_Strength': 'mean',
    'Delay': 'mean',
    'PDR': 'mean',
    'Energy': 'mean',
    'Reliability_Score': 'mean'

}).round(2)

print("UAV Performance Summary:\n")
print(uav_summary)

print("\n======================================")
print(" Coordinated UAV Selection ")
print("======================================\n")

# Best Coordinated UAV
best_uav = uav_summary['Reliability_Score'].idxmax()

print(f"Best Coordinated UAV : {best_uav}")

print("\nPerformance Details:\n")
print(uav_summary.loc[best_uav])

print("\n======================================")
print(" Communication Load Distribution ")
print("======================================\n")

# Communication Load
load_distribution = df['UAV_ID'].value_counts()

print(load_distribution)

print("\n======================================")
print(" Multi-UAV Coordination Completed ")
print("======================================")