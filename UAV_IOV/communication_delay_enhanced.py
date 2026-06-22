import pandas as pd

from data_loader import load_dataset
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = load_dataset()

# Group by timestamp and calculate min, average, and max delay
delay_stats = df.groupby('Timestamp')['Delay'].agg(['min', 'mean', 'max']).reset_index()
delay_stats = delay_stats.sort_values('Timestamp')

# Apply moving average smoothing (window size = 7)
delay_stats['min_smoothed'] = delay_stats['min'].rolling(window=7, center=True).mean()
delay_stats['mean_smoothed'] = delay_stats['mean'].rolling(window=7, center=True).mean()
delay_stats['max_smoothed'] = delay_stats['max'].rolling(window=7, center=True).mean()

# Fill NaN values at edges
delay_stats['min_smoothed'] = delay_stats['min_smoothed'].fillna(delay_stats['min'])
delay_stats['mean_smoothed'] = delay_stats['mean_smoothed'].fillna(delay_stats['mean'])
delay_stats['max_smoothed'] = delay_stats['max_smoothed'].fillna(delay_stats['max'])

# Publication-quality styling
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
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

plt.figure(figsize=(12, 7))

# Plot min, average, and max delay
plt.plot(delay_stats['Timestamp'], delay_stats['min_smoothed'],
         linewidth=2, color='green', linestyle='--', label='Minimum Delay (Smoothed)')
plt.plot(delay_stats['Timestamp'], delay_stats['mean_smoothed'],
         linewidth=2.5, color='blue', label='Average Delay (Smoothed)')
plt.plot(delay_stats['Timestamp'], delay_stats['max_smoothed'],
         linewidth=2, color='red', linestyle='--', label='Maximum Delay (Smoothed)')

plt.title("Communication Delay Over Time: Min, Average, and Max", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Timestamp", fontsize=11, fontweight='bold')
plt.ylabel("Delay (ms)", fontsize=11, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

plt.savefig("graph_outputs/communication_delay_enhanced.png", dpi=300, bbox_inches='tight')
print("Saved: graph_outputs/communication_delay_enhanced.png")
plt.close()

print("\n=== MODIFICATIONS SUMMARY ===")
print("1. Grouped delay values by timestamp using group-by aggregation")
print("2. Calculated minimum, average, and maximum delay per timestamp")
print("3. Applied moving average smoothing (window size = 7) to reduce noise")
print("4. Used publication-quality styling (IEEE/Springer format)")
print("5. Increased figure size to (12, 7) for better readability")
print("6. Increased DPI to 300 for high-resolution publication")
print("7. Added proper labels, legend, and grid")
print("8. Created enhanced version showing min/avg/max with different colors")
print("\n=== WHY THE NEW GRAPH IS BETTER ===")
print("- Removes dense vertical-line appearance by aggregating per timestamp")
print("- Shows clear trend of communication delay over time")
print("- Smoothing reduces noise while preserving overall pattern")
print("- Min/avg/max lines provide comprehensive view of delay distribution")
print("- Publication-quality styling suitable for IEEE/Springer papers")
print("- Clean and readable even for 100+ timestamps")
