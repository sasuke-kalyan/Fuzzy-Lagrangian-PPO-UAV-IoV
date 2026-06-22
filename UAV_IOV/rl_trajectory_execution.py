import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_dataset

df = load_dataset()
if "Scenario_ID" in df.columns:
    first_sid = df["Scenario_ID"].iloc[0]
    df = df[df["Scenario_ID"] == first_sid]

# Ensure consistent styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# Get unique UAVs and vehicles
uavs = df['UAV_ID'].unique()
vehicles = df['Vehicle_ID'].unique()

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Plot vehicles as reference points
vehicle_data = df.groupby('Vehicle_ID').agg({
    'Vehicle_X': 'mean',
    'Vehicle_Y': 'mean'
}).reset_index()

ax.scatter(vehicle_data['Vehicle_X'], vehicle_data['Vehicle_Y'], 
           c='blue', s=200, marker='s', label='Vehicles', alpha=0.7, edgecolors='black', linewidth=1.5)

# Add vehicle labels
for _, row in vehicle_data.iterrows():
    ax.annotate(row['Vehicle_ID'], (row['Vehicle_X'], row['Vehicle_Y']), 
                xytext=(0, 0), textcoords='offset points', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Plot each UAV trajectory with execution order numbers
colors = plt.cm.rainbow(np.linspace(0, 1, len(uavs)))

for idx, uav in enumerate(uavs):
    uav_data = df[df['UAV_ID'] == uav].sort_values('Timestamp')
    
    # Get trajectory points
    trajectory = uav_data.groupby('Timestamp').agg({
        'UAV_X': 'mean',
        'UAV_Y': 'mean'
    }).reset_index()
    
    # Plot trajectory line
    ax.plot(trajectory['UAV_X'], trajectory['UAV_Y'], 
            color=colors[idx], linewidth=2.5, alpha=0.8, label=f'UAV {uav}')
    
    # Add numbered markers for execution order
    for i, (_, row) in enumerate(trajectory.iterrows()):
        if i % 2 == 0:  # Show every other point to avoid overcrowding
            ax.annotate(str(i+1), (row['UAV_X'], row['UAV_Y']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold', 
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor=colors[idx], edgecolor='black', alpha=0.8),
                        color='white')

# Set labels and title
ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
ax.set_title('RL UAV Trajectories with Execution Order', fontsize=14, fontweight='bold', pad=20)

# Add legend
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Add grid
ax.grid(True, linestyle='--', alpha=0.6)

# Set equal aspect ratio
ax.set_aspect('equal')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('graph_outputs/rl_trajectory_execution_order.png', dpi=300, bbox_inches='tight')
print("Saved: rl_trajectory_execution_order.png")

plt.close()
