import pandas as pd

from data_loader import load_dataset

from graph_data import snapshot_at_timestamp
from lagrangian_solver import base_routing_score, add_lagrangian_columns, LagrangianMultipliers

df = load_dataset()
snap = snapshot_at_timestamp(df)

print("\n======================================")
print(" Lagrangian Smart Routing Optimization ")
print("======================================\n")

# Routing objective (same as before)
snap["Routing_Score"] = snap.apply(base_routing_score, axis=1)

lam = LagrangianMultipliers(delay=2.0, pdr=1.5, energy=1.5, signal=0.5)
snap = add_lagrangian_columns(snap, lam, objective_col="Routing_Score")

print("Routing with constraints (sample):\n")
print(
    snap[
        [
            "Vehicle_ID",
            "UAV_ID",
            "Routing_Score",
            "Lagrangian_Score",
            "Violation_Delay",
            "Violation_Energy",
        ]
    ].head(8)
)

# Best route per vehicle: maximize Lagrangian_Score
best_routes = snap.loc[snap.groupby("Vehicle_ID")["Lagrangian_Score"].idxmax()]

print("\nBest constrained route per vehicle:\n")
for _, row in best_routes.iterrows():
    print(f"Vehicle : {row['Vehicle_ID']}")
    print(f"Best UAV : {row['UAV_ID']}")
    print(f"Routing Score : {round(row['Routing_Score'], 2)}")
    print(f"Lagrangian Score : {round(row['Lagrangian_Score'], 2)}")
    print(f"Delay : {row['Delay']} | PDR : {row['PDR']} | Energy : {row['Energy']}")
    print("--------------------------------------")

print("\n======================================")
print(" Network-Wide Routing Statistics ")
print("======================================\n")

avg_route = round(snap["Routing_Score"].mean(), 2)
avg_lag = round(snap["Lagrangian_Score"].mean(), 2)
best_uav = snap.groupby("UAV_ID")["Lagrangian_Score"].mean().idxmax()

print(f"Average Routing Score     : {avg_route}")
print(f"Average Lagrangian Score  : {avg_lag}")
print(f"Best Overall UAV (Lagr.)  : {best_uav}")

print("\n======================================")
print(" Lagrangian Smart Routing Completed ")
print("======================================")
