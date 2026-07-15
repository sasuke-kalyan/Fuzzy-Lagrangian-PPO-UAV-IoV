import pandas as pd

from data_loader import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from communication_model import AREA_SIZE

# Load dataset
df = load_dataset()

print("\n======================================")
print(" UAV TRAJECTORY VISUALIZATIONS ")
print("======================================\n")

# Get unique UAVs
uavs = df['UAV_ID'].unique()
vehicles = df['Vehicle_ID'].unique()

print(f"Found {len(uavs)} UAVs and {len(vehicles)} vehicles")

# -------------------------------
# TRAJECTORY FOR EACH UAV
# -------------------------------
for uav in uavs:
    print(f"\nCreating trajectory for {uav}...")
    
    # Get data for this UAV
    uav_data = df[df['UAV_ID'] == uav].sort_values('Timestamp')
    
    # Group by timestamp and get average position
    uav_trajectory = uav_data.groupby('Timestamp').agg({
        'UAV_X': 'mean',
        'UAV_Y': 'mean',
        'Signal_Strength': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 8))
    
    # Plot trajectory
    plt.plot(uav_trajectory['UAV_X'], uav_trajectory['UAV_Y'], 
             marker='o', linewidth=2, markersize=8, color='red', 
             label=f'{uav} Trajectory')
    
    # Add direction arrows
    for i in range(len(uav_trajectory) - 1):
        x1, y1 = uav_trajectory['UAV_X'].iloc[i], uav_trajectory['UAV_Y'].iloc[i]
        x2, y2 = uav_trajectory['UAV_X'].iloc[i+1], uav_trajectory['UAV_Y'].iloc[i+1]
        plt.arrow(x1, y1, x2-x1, y2-y1, head_width=2, head_length=2, 
                 fc='red', ec='red', alpha=0.5)
    
    # Plot vehicles served by this UAV
    vehicles_served = uav_data['Vehicle_ID'].unique()
    for vehicle in vehicles_served:
        vehicle_data = uav_data[uav_data['Vehicle_ID'] == vehicle].iloc[0]
        plt.scatter(vehicle_data['Vehicle_X'], vehicle_data['Vehicle_Y'], 
                   c='blue', s=100, marker='s', edgecolors='black', linewidths=2,
                   label=f'{vehicle}' if vehicle == vehicles_served[0] else "")
    
    # Add start and end markers
    plt.scatter(uav_trajectory['UAV_X'].iloc[0], uav_trajectory['UAV_Y'].iloc[0],
               c='green', s=150, marker='^', edgecolors='black', linewidths=2,
               label='Start')
    plt.scatter(uav_trajectory['UAV_X'].iloc[-1], uav_trajectory['UAV_Y'].iloc[-1],
               c='black', s=150, marker='X', edgecolors='black', linewidths=2,
               label='End')
    
    plt.title(f'{uav} Trajectory and Served Vehicles', fontsize=14, fontweight='bold')
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.xlim(0, AREA_SIZE)
    plt.ylim(0, AREA_SIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    filename = f'trajectory_{uav}.png'
    plt.savefig(filename, dpi=120)
    print(f"Saved: {filename}")
    plt.close()

# -------------------------------
# COMBINED TRAJECTORY PLOT
# -------------------------------
print("\nCreating combined trajectory plot...")

plt.figure(figsize=(12, 10))

# Plot all UAV trajectories
colors = plt.cm.rainbow(np.linspace(0, 1, len(uavs)))
for i, uav in enumerate(uavs):
    uav_data = df[df['UAV_ID'] == uav].sort_values('Timestamp')
    uav_trajectory = uav_data.groupby('Timestamp').agg({
        'UAV_X': 'mean',
        'UAV_Y': 'mean'
    }).reset_index()
    
    plt.plot(uav_trajectory['UAV_X'], uav_trajectory['UAV_Y'], 
             marker='o', linewidth=2, markersize=6, color=colors[i],
             label=f'{uav}', alpha=0.7)

# Plot all vehicles
for vehicle in vehicles:
    vehicle_data = df[df['Vehicle_ID'] == vehicle].iloc[0]
    plt.scatter(vehicle_data['Vehicle_X'], vehicle_data['Vehicle_Y'], 
               c='blue', s=80, marker='s', edgecolors='black', linewidths=1,
               alpha=0.6)

plt.title('All UAV Trajectories and Vehicle Positions', fontsize=14, fontweight='bold')
plt.xlabel('X Position (m)', fontsize=12)
plt.ylabel('Y Position (m)', fontsize=12)
plt.xlim(0, AREA_SIZE)
plt.ylim(0, AREA_SIZE)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig('all_trajectories.png', dpi=120)
print("Saved: all_trajectories.png")
plt.close()

# -------------------------------
# TRAJECTORY WITH SIGNAL STRENGTH
# -------------------------------
print("Creating trajectory with signal strength...")

for uav in uavs[:3]:  # Limit to first 3 UAVs for clarity
    uav_data = df[df['UAV_ID'] == uav].sort_values('Timestamp')
    uav_trajectory = uav_data.groupby('Timestamp').agg({
        'UAV_X': 'mean',
        'UAV_Y': 'mean',
        'Signal_Strength': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 8))
    
    # Color code by signal strength
    scatter = plt.scatter(uav_trajectory['UAV_X'], uav_trajectory['UAV_Y'],
                          c=uav_trajectory['Signal_Strength'], 
                          s=200, cmap='RdYlGn', edgecolors='black', linewidths=2)
    
    # Add trajectory line
    plt.plot(uav_trajectory['UAV_X'], uav_trajectory['UAV_Y'], 
             'k--', alpha=0.5, linewidth=1)
    
    # Add arrows
    for i in range(len(uav_trajectory) - 1):
        x1, y1 = uav_trajectory['UAV_X'].iloc[i], uav_trajectory['UAV_Y'].iloc[i]
        x2, y2 = uav_trajectory['UAV_X'].iloc[i+1], uav_trajectory['UAV_Y'].iloc[i+1]
        plt.arrow(x1, y1, x2-x1, y2-y1, head_width=2, head_length=2, 
                 fc='black', ec='black', alpha=0.3)
    
    plt.colorbar(scatter, label='Signal Strength')
    plt.title(f'{uav} Trajectory with Signal Strength', fontsize=14, fontweight='bold')
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.xlim(0, AREA_SIZE)
    plt.ylim(0, AREA_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'trajectory_{uav}_signal.png'
    plt.savefig(filename, dpi=120)
    print(f"Saved: {filename}")
    plt.close()

# -------------------------------
# 3D TRAJECTORY VISUALIZATION
# -------------------------------
print("Creating 3D trajectory visualization...")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for i, uav in enumerate(uavs):
    uav_data = df[df['UAV_ID'] == uav].sort_values('Timestamp')
    uav_trajectory = uav_data.groupby('Timestamp').agg({
        'UAV_X': 'mean',
        'UAV_Y': 'mean',
        'Signal_Strength': 'mean'
    }).reset_index()
    
    # Use timestamp as Z-axis
    ax.plot(uav_trajectory['UAV_X'], uav_trajectory['UAV_Y'], 
            uav_trajectory['Timestamp'], 
            marker='o', linewidth=2, markersize=6, color=colors[i],
            label=f'{uav}', alpha=0.7)

# Plot vehicles at Z=0
for vehicle in vehicles:
    vehicle_data = df[df['Vehicle_ID'] == vehicle].iloc[0]
    ax.scatter(vehicle_data['Vehicle_X'], vehicle_data['Vehicle_Y'], 0,
               c='blue', s=80, marker='s', edgecolors='black', linewidths=1,
               alpha=0.6)

ax.set_title('3D UAV Trajectories (X, Y, Time)', fontsize=14, fontweight='bold')
ax.set_xlabel('X Position (m)', fontsize=12)
ax.set_ylabel('Y Position (m)', fontsize=12)
ax.set_zlabel('Timestamp', fontsize=12)
ax.set_xlim(0, AREA_SIZE)
ax.set_ylim(0, AREA_SIZE)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('3d_trajectories.png', dpi=120)
print("Saved: 3d_trajectories.png")
plt.close()

print("\n======================================")
print(" Trajectory Visualizations Completed ")
print("======================================\n")
