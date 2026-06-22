import pandas as pd

from data_loader import load_dataset
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = load_dataset()

# Reliability Score
df['Reliability_Score'] = (
    (df['Signal_Strength'] * 50)
    + (df['PDR'] * 0.5)
    - (df['Delay'] * 0.4)
)

# -------------------------------
# SIGNAL STRENGTH GRAPH
# -------------------------------

plt.figure(figsize=(8,5))

plt.plot(df['Timestamp'],
         df['Signal_Strength'])

plt.title("Signal Strength Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Signal Strength")

plt.grid(True)

plt.savefig("signal_strength_over_time.png", dpi=120)
print("Saved: signal_strength_over_time.png")
plt.show()

# -------------------------------
# DELAY GRAPH (IMPROVED)
# -------------------------------

# Group by timestamp and calculate average delay
delay_by_timestamp = df.groupby('Timestamp')['Delay'].mean().reset_index()
delay_by_timestamp = delay_by_timestamp.sort_values('Timestamp')

# Apply moving average smoothing (window size = 7)
delay_by_timestamp['Delay_Smoothed'] = delay_by_timestamp['Delay'].rolling(window=7, center=True).mean()
delay_by_timestamp['Delay_Smoothed'] = delay_by_timestamp['Delay_Smoothed'].fillna(delay_by_timestamp['Delay'])

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

plt.figure(figsize=(10, 6))

# Plot smoothed average delay
plt.plot(delay_by_timestamp['Timestamp'],
         delay_by_timestamp['Delay_Smoothed'],
         linewidth=2.5,
         color='blue',
         label='Average Delay (Smoothed)')

plt.title("Communication Delay Over Time", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Timestamp", fontsize=11, fontweight='bold')
plt.ylabel("Delay (ms)", fontsize=11, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

plt.savefig("communication_delay_over_time.png", dpi=300, bbox_inches='tight')
print("Saved: communication_delay_over_time.png")
plt.close()

# -------------------------------
# PDR GRAPH
# -------------------------------

plt.figure(figsize=(8,5))

plt.plot(df['Timestamp'],
         df['PDR'])

plt.title("Packet Delivery Ratio Over Time")
plt.xlabel("Timestamp")
plt.ylabel("PDR")

plt.grid(True)

plt.savefig("packet_delivery_ratio_over_time.png", dpi=120)
print("Saved: packet_delivery_ratio_over_time.png")
plt.show()

# -------------------------------
# RELIABILITY SCORE GRAPH
# -------------------------------

plt.figure(figsize=(8,5))

plt.plot(df['Timestamp'],
         df['Reliability_Score'])

plt.title("Reliability Score Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Reliability Score")

plt.grid(True)

plt.savefig("reliability_score_over_time.png", dpi=120)
print("Saved: reliability_score_over_time.png")
plt.show()