import pandas as pd

from data_loader import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = load_dataset()

print("\n======================================")
print(" HEATMAP VISUALIZATIONS ")
print("======================================\n")

# -------------------------------
# SIGNAL STRENGTH HEATMAP
# -------------------------------
print("Creating Signal Strength Heatmap...")

# Create pivot table for signal strength
signal_pivot = df.pivot_table(
    values='Signal_Strength',
    index='Vehicle_ID',
    columns='UAV_ID',
    aggfunc='mean'
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    signal_pivot,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    cbar_kws={'label': 'Signal Strength'},
    linewidths=0.5,
    linecolor='gray'
)
plt.title('Signal Strength Heatmap (Vehicle vs UAV)', fontsize=14, fontweight='bold')
plt.xlabel('UAV ID', fontsize=12)
plt.ylabel('Vehicle ID', fontsize=12)
plt.tight_layout()
plt.savefig('signal_strength_heatmap.png', dpi=120)
print("Saved: signal_strength_heatmap.png")
plt.close()

# -------------------------------
# DELAY HEATMAP
# -------------------------------
print("Creating Delay Heatmap...")

delay_pivot = df.pivot_table(
    values='Delay',
    index='Vehicle_ID',
    columns='UAV_ID',
    aggfunc='mean'
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    delay_pivot,
    annot=True,
    fmt='.1f',
    cmap='YlOrRd',
    cbar_kws={'label': 'Delay (ms)'},
    linewidths=0.5,
    linecolor='gray'
)
plt.title('Communication Delay Heatmap (Vehicle vs UAV)', fontsize=14, fontweight='bold')
plt.xlabel('UAV ID', fontsize=12)
plt.ylabel('Vehicle ID', fontsize=12)
plt.tight_layout()
plt.savefig('delay_heatmap.png', dpi=120)
print("Saved: delay_heatmap.png")
plt.close()

# -------------------------------
# PDR HEATMAP
# -------------------------------
print("Creating PDR Heatmap...")

pdr_pivot = df.pivot_table(
    values='PDR',
    index='Vehicle_ID',
    columns='UAV_ID',
    aggfunc='mean'
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    pdr_pivot,
    annot=True,
    fmt='.1f',
    cmap='Blues',
    cbar_kws={'label': 'PDR (%)'},
    linewidths=0.5,
    linecolor='gray'
)
plt.title('Packet Delivery Ratio Heatmap (Vehicle vs UAV)', fontsize=14, fontweight='bold')
plt.xlabel('UAV ID', fontsize=12)
plt.ylabel('Vehicle ID', fontsize=12)
plt.tight_layout()
plt.savefig('pdr_heatmap.png', dpi=120)
print("Saved: pdr_heatmap.png")
plt.close()

# -------------------------------
# ENERGY HEATMAP
# -------------------------------
print("Creating Energy Heatmap...")

energy_pivot = df.pivot_table(
    values='Energy',
    index='Vehicle_ID',
    columns='UAV_ID',
    aggfunc='mean'
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    energy_pivot,
    annot=True,
    fmt='.1f',
    cmap='PuBuGn',
    cbar_kws={'label': 'Energy Consumption'},
    linewidths=0.5,
    linecolor='gray'
)
plt.title('Energy Consumption Heatmap (Vehicle vs UAV)', fontsize=14, fontweight='bold')
plt.xlabel('UAV ID', fontsize=12)
plt.ylabel('Vehicle ID', fontsize=12)
plt.tight_layout()
plt.savefig('energy_heatmap.png', dpi=120)
print("Saved: energy_heatmap.png")
plt.close()

# -------------------------------
# RELIABILITY SCORE HEATMAP
# -------------------------------
print("Creating Reliability Score Heatmap...")

df['Reliability_Score'] = (
    (df['Signal_Strength'] * 50) +
    (df['PDR'] * 0.4) -
    (df['Delay'] * 0.5)
)

reliability_pivot = df.pivot_table(
    values='Reliability_Score',
    index='Vehicle_ID',
    columns='UAV_ID',
    aggfunc='mean'
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    reliability_pivot,
    annot=True,
    fmt='.1f',
    cmap='viridis',
    cbar_kws={'label': 'Reliability Score'},
    linewidths=0.5,
    linecolor='gray'
)
plt.title('Reliability Score Heatmap (Vehicle vs UAV)', fontsize=14, fontweight='bold')
plt.xlabel('UAV ID', fontsize=12)
plt.ylabel('Vehicle ID', fontsize=12)
plt.tight_layout()
plt.savefig('reliability_score_heatmap.png', dpi=120)
print("Saved: reliability_score_heatmap.png")
plt.close()

print("\n======================================")
print(" Heatmap Visualizations Completed ")
print("======================================\n")
