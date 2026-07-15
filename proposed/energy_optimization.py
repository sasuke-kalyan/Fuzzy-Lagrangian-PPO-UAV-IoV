import pandas as pd

from data_loader import load_dataset

# Load dataset
df = load_dataset()

print("\n======================================")
print(" Energy-Aware UAV Optimization ")
print("======================================\n")

# Energy Status Function
def energy_status(energy):

    if energy >= 35:
        return "High"

    elif energy >= 20:
        return "Medium"

    else:
        return "Low"

# Apply Energy Classification
df['Energy_Status'] = df['Energy'].apply(energy_status)

# Show Energy Summary
energy_counts = df['Energy_Status'].value_counts()

print("UAV Energy Distribution:\n")
print(energy_counts)

print("\n======================================")
print(" Low Energy UAV Detection ")
print("======================================\n")

# Detect low-energy communication records
low_energy_records = df[df['Energy_Status'] == "Low"]

print(low_energy_records[
    [
        'Vehicle_ID',
        'UAV_ID',
        'Energy',
        'Signal_Strength',
        'Delay'
    ]
].head())

print("\n======================================")
print(" Energy Efficient UAV Selection ")
print("======================================\n")

# Reliability Score
df['Reliability_Score'] = (
    (df['Signal_Strength'] * 50)
    + (df['PDR'] * 0.5)
    - (df['Delay'] * 0.4)
)

# Combined Optimization Score
df['Optimization_Score'] = (
    df['Reliability_Score']
    + df['Energy']
)

# Best Energy-Efficient UAV
best_uav = df.loc[df['Optimization_Score'].idxmax()]

print("Best UAV Selection:\n")

print(best_uav[
    [
        'Vehicle_ID',
        'UAV_ID',
        'Energy',
        'Signal_Strength',
        'Delay',
        'PDR',
        'Optimization_Score'
    ]
])

print("\n======================================")
print(" Energy Optimization Completed ")
print("======================================")