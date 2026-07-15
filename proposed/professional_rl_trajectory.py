import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch
import matplotlib.lines as mlines

from data_loader import load_dataset

# Load dataset (all scenarios; uses first scenario slice for trajectory plot)
df = load_dataset()
if "Scenario_ID" in df.columns:
    first_sid = df["Scenario_ID"].iloc[0]
    df = df[df["Scenario_ID"] == first_sid]

# Professional styling for IEEE papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

# Get unique entities
uavs = sorted(df['UAV_ID'].unique())
vehicles = sorted(df['Vehicle_ID'].unique())

# Assign distinct colors to UAVs
uav_colors = plt.cm.tab10(np.linspace(0, 1, len(uavs)))

# Create figure with larger size for better visibility
fig, ax = plt.subplots(figsize=(16, 12))

# Get vehicle positions (task locations)
vehicle_data = df.groupby('Vehicle_ID').agg({
    'Vehicle_X': 'mean',
    'Vehicle_Y': 'mean',
    'Signal_Strength': 'mean'
}).reset_index()

# Simulate emergency vs normal vehicles (based on signal strength threshold)
vehicle_data['Type'] = vehicle_data['Signal_Strength'].apply(
    lambda x: 'Emergency' if x > 0.25 else 'Normal'
)

# Plot vehicles as numbered circular nodes
for idx, (_, row) in enumerate(vehicle_data.iterrows()):
    color = 'red' if row['Type'] == 'Emergency' else 'orange'
    edge_color = 'darkred' if row['Type'] == 'Emergency' else 'darkorange'
    
    # Draw circular node with larger radius
    circle = Circle((row['Vehicle_X'], row['Vehicle_Y']), radius=5, 
                    facecolor=color, edgecolor=edge_color, linewidth=3, alpha=0.9)
    ax.add_patch(circle)
    
    # Add task number with larger font
    ax.text(row['Vehicle_X'], row['Vehicle_Y'], str(idx+1), 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Add vehicle type label below
    label_text = 'EMERGENCY' if row['Type'] == 'Emergency' else 'NORMAL'
    ax.text(row['Vehicle_X'], row['Vehicle_Y'] - 8, label_text, 
            ha='center', va='top', fontsize=9, fontweight='bold', 
            color=color)

# Plot RSUs (simulated at fixed locations)
rsu_positions = [(20, 20), (80, 20), (50, 80)]
for i, (x, y) in enumerate(rsu_positions):
    # Larger triangle for better visibility
    triangle = Polygon([(x, y+7), (x-6, y-4), (x+6, y-4)], 
                      facecolor='black', edgecolor='black', linewidth=3, alpha=0.95)
    ax.add_patch(triangle)
    ax.text(x, y-8, f'RSU{i+1}', ha='center', va='top', fontsize=11, fontweight='bold')

# Plot UAVs with their trajectories
for uav_idx, uav in enumerate(uavs):
    uav_data = df[df['UAV_ID'] == uav].sort_values('Timestamp')
    color = uav_colors[uav_idx]
    
    # Get UAV starting position (first timestamp)
    start_pos = uav_data.iloc[0]
    
    # Plot UAV as large square marker
    uav_square = Rectangle((start_pos['UAV_X']-6, start_pos['UAV_Y']-6), 
                           12, 12, facecolor=color, edgecolor='black', 
                           linewidth=3, alpha=0.95)
    ax.add_patch(uav_square)
    ax.text(start_pos['UAV_X'], start_pos['UAV_Y'], f'U{uav_idx+1}', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Add UAV label above
    ax.text(start_pos['UAV_X'], start_pos['UAV_Y'] + 10, f'UAV {uav}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold', 
            color=color)
    
    # Get trajectory points
    trajectory = uav_data.groupby('Timestamp').agg({
        'UAV_X': 'mean',
        'UAV_Y': 'mean'
    }).reset_index()
    
    # Primary Route (Solid Line) - actual flight path
    ax.plot(trajectory['UAV_X'], trajectory['UAV_Y'], 
            color=color, linewidth=2.5, alpha=0.9, 
            label=f'UAV {uav} (Primary Route)')
    
    # Task Association Path (Dashed Line) - connect to assigned vehicles
    # Get vehicles served by this UAV
    served_vehicles = uav_data['Vehicle_ID'].unique()
    for vehicle in served_vehicles:
        veh_data = vehicle_data[vehicle_data['Vehicle_ID'] == vehicle]
        if not veh_data.empty:
            veh_pos = veh_data.iloc[0]
            # Draw dashed line from UAV to vehicle
            ax.plot([start_pos['UAV_X'], veh_pos['Vehicle_X']], 
                    [start_pos['UAV_Y'], veh_pos['Vehicle_Y']], 
                    color=color, linewidth=1.5, linestyle='--', alpha=0.6)
            
            # Communication link (dotted line) for emergency vehicles
            if veh_pos['Type'] == 'Emergency':
                ax.plot([start_pos['UAV_X'], veh_pos['Vehicle_X']], 
                        [start_pos['UAV_Y'], veh_pos['Vehicle_Y']], 
                        color='blue', linewidth=2, linestyle=':', alpha=0.4)

# Set labels and title
ax.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
ax.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
ax.set_title('RL-Optimized UAV-IoV Task Scheduling and Emergency Service Execution Trajectories', 
             fontsize=12, fontweight='bold', pad=15)

# Create custom legend with larger markers and text
legend_elements = [
    mlines.Line2D([], [], color='red', marker='o', linestyle='None', 
                  markersize=15, label='Emergency Vehicle'),
    mlines.Line2D([], [], color='orange', marker='o', linestyle='None', 
                  markersize=15, label='Normal Vehicle'),
    mlines.Line2D([], [], color='black', marker='^', linestyle='None', 
                  markersize=15, label='RSU (Road Side Unit)'),
    mlines.Line2D([], [], color='blue', marker='s', linestyle='None', 
                  markersize=15, label='UAV (Unmanned Aerial Vehicle)'),
    mlines.Line2D([], [], color='gray', linewidth=3, label='Primary Route (Solid Line)'),
    mlines.Line2D([], [], color='gray', linewidth=2, linestyle='--', label='Task Assignment (Dashed Line)'),
    mlines.Line2D([], [], color='blue', linewidth=3, linestyle=':', label='Emergency Communication Link (Dotted Line)')
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
          framealpha=0.95, fancybox=True, shadow=True, borderpad=1)

# Add grid
ax.grid(True, linestyle='--', alpha=0.3)

# Set equal aspect ratio
ax.set_aspect('equal')

# Set axis limits with some padding
all_x = np.concatenate([vehicle_data['Vehicle_X'].values, 
                       [pos[0] for pos in rsu_positions],
                       df['UAV_X'].values])
all_y = np.concatenate([vehicle_data['Vehicle_Y'].values, 
                       [pos[1] for pos in rsu_positions],
                       df['UAV_Y'].values])

ax.set_xlim(all_x.min() - 10, all_x.max() + 10)
ax.set_ylim(all_y.min() - 10, all_y.max() + 10)

# Adjust layout
plt.tight_layout()

# Save the figure with high DPI for publication
plt.savefig('graph_outputs/professional_rl_trajectory.png', dpi=300, 
            bbox_inches='tight', facecolor='white')
print("Saved: professional_rl_trajectory.png")

plt.close()
