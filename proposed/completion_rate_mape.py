import pandas as pd

from data_loader import load_dataset
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = load_dataset()

print("\n======================================")
print(" COMPLETION RATE & MAPE ANALYSIS ")
print("======================================\n")

# Calculate completion rate based on PDR
# Completion Rate = PDR / 100 (normalized to 0-1)
df['Completion_Rate'] = df['PDR'] / 100

# Calculate MAPE (Mean Absolute Percentage Error)
# MAPE measures prediction accuracy for task completion
# Here we simulate it as the deviation from ideal completion (100%)
df['MAPE'] = np.abs((100 - df['PDR']) / 100) * 100

print("Completion Rate Statistics:")
print(df['Completion_Rate'].describe())
print("\nMAPE Statistics:")
print(df['MAPE'].describe())

# -------------------------------
# COMPLETION RATE PER VEHICLE
# -------------------------------
print("\nCreating Completion Rate per Vehicle...")

avg_completion = df.groupby('Vehicle_ID')['Completion_Rate'].mean()

# Sort vehicle IDs in numerical order (V1, V2, ..., V10)
avg_completion = avg_completion.reindex(sorted(avg_completion.index, key=lambda x: int(x[1:])))

plt.figure(figsize=(10, 6))
avg_completion.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Average Completion Rate per Vehicle', fontsize=14, fontweight='bold')
plt.xlabel('Vehicle ID', fontsize=12)
plt.ylabel('Completion Rate (0-1)', fontsize=12)
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('completion_rate_per_vehicle.png', dpi=120)
print("Saved: completion_rate_per_vehicle.png")
plt.close()

# -------------------------------
# MAPE PER VEHICLE
# -------------------------------
print("Creating MAPE per Vehicle...")

avg_mape = df.groupby('Vehicle_ID')['MAPE'].mean()

# Sort vehicle IDs in numerical order (V1, V2, ..., V10)
avg_mape = avg_mape.reindex(sorted(avg_mape.index, key=lambda x: int(x[1:])))

plt.figure(figsize=(10, 6))
avg_mape.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Mean Absolute Percentage Error (MAPE) per Vehicle', fontsize=14, fontweight='bold')
plt.xlabel('Vehicle ID', fontsize=12)
plt.ylabel('MAPE (%)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('mape_per_vehicle.png', dpi=120)
print("Saved: mape_per_vehicle.png")
plt.close()

# -------------------------------
# COMPLETION RATE OVER TIME
# -------------------------------
print("Creating Completion Rate Over Time...")

completion_over_time = df.groupby('Timestamp')['Completion_Rate'].mean()

plt.figure(figsize=(12, 6))
plt.plot(completion_over_time.index, completion_over_time.values, 
         color='steelblue', marker='o', linewidth=2, markersize=6)
plt.title('Completion Rate Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Timestamp', fontsize=12)
plt.ylabel('Completion Rate (0-1)', fontsize=12)
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('completion_rate_over_time.png', dpi=120)
print("Saved: completion_rate_over_time.png")
plt.close()

# -------------------------------
# MAPE OVER TIME
# -------------------------------
print("Creating MAPE Over Time...")

mape_over_time = df.groupby('Timestamp')['MAPE'].mean()

plt.figure(figsize=(12, 6))
plt.plot(mape_over_time.index, mape_over_time.values, 
         color='coral', marker='s', linewidth=2, markersize=6)
plt.title('MAPE Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Timestamp', fontsize=12)
plt.ylabel('MAPE (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mape_over_time.png', dpi=120)
print("Saved: mape_over_time.png")
plt.close()

# -------------------------------
# COMPLETION RATE VS MAPE SCATTER
# -------------------------------
print("Creating Completion Rate vs MAPE Scatter Plot...")

plt.figure(figsize=(10, 6))
plt.scatter(df['Completion_Rate'], df['MAPE'], alpha=0.6, c='purple', s=50)
plt.title('Completion Rate vs MAPE', fontsize=14, fontweight='bold')
plt.xlabel('Completion Rate (0-1)', fontsize=12)
plt.ylabel('MAPE (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('completion_rate_vs_mape.png', dpi=120)
print("Saved: completion_rate_vs_mape.png")
plt.close()

# -------------------------------
# COMPLETION RATE HEATMAP
# -------------------------------
print("Creating Completion Rate Heatmap...")

completion_pivot = df.pivot_table(
    values='Completion_Rate',
    index='Vehicle_ID',
    columns='UAV_ID',
    aggfunc='mean'
)

plt.figure(figsize=(10, 8))
plt.imshow(completion_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='Completion Rate (0-1)')
plt.xticks(range(len(completion_pivot.columns)), completion_pivot.columns)
plt.yticks(range(len(completion_pivot.index)), completion_pivot.index)
plt.title('Completion Rate Heatmap (Vehicle vs UAV)', fontsize=14, fontweight='bold')
plt.xlabel('UAV ID', fontsize=12)
plt.ylabel('Vehicle ID', fontsize=12)

# Add value annotations
for i in range(len(completion_pivot.index)):
    for j in range(len(completion_pivot.columns)):
        plt.text(j, i, f'{completion_pivot.values[i, j]:.2f}',
                ha='center', va='center', color='black', fontsize=8)

plt.tight_layout()
plt.savefig('completion_rate_heatmap.png', dpi=120)
print("Saved: completion_rate_heatmap.png")
plt.close()

# -------------------------------
# MAPE HEATMAP
# -------------------------------
print("Creating MAPE Heatmap...")

mape_pivot = df.pivot_table(
    values='MAPE',
    index='Vehicle_ID',
    columns='UAV_ID',
    aggfunc='mean'
)

plt.figure(figsize=(10, 8))
plt.imshow(mape_pivot.values, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='MAPE (%)')
plt.xticks(range(len(mape_pivot.columns)), mape_pivot.columns)
plt.yticks(range(len(mape_pivot.index)), mape_pivot.index)
plt.title('MAPE Heatmap (Vehicle vs UAV)', fontsize=14, fontweight='bold')
plt.xlabel('UAV ID', fontsize=12)
plt.ylabel('Vehicle ID', fontsize=12)

# Add value annotations
for i in range(len(mape_pivot.index)):
    for j in range(len(mape_pivot.columns)):
        plt.text(j, i, f'{mape_pivot.values[i, j]:.1f}',
                ha='center', va='center', color='black', fontsize=8)

plt.tight_layout()
plt.savefig('mape_heatmap.png', dpi=120)
print("Saved: mape_heatmap.png")
plt.close()

print("\n======================================")
print(" Completion Rate & MAPE Analysis Completed ")
print("======================================\n")
