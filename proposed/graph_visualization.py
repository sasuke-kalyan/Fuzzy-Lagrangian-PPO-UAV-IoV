import pandas as pd

from data_loader import load_dataset
import networkx as nx
import matplotlib.pyplot as plt

from graph_data import snapshot_at_timestamp

df = load_dataset()
snap = snapshot_at_timestamp(df)

G = nx.Graph()

for _, row in snap.iterrows():
    vehicle = row["Vehicle_ID"]
    uav = row["UAV_ID"]
    signal = row["Signal_Strength"]

    G.add_node(vehicle, type="vehicle")
    G.add_node(uav, type="uav")
    G.add_edge(vehicle, uav, weight=round(signal, 3))

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(12, 9))

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=2500,
    font_size=9,
)

edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

plt.title(
    f"UAV-IoV Communication Graph (timestamp {int(snap['Timestamp'].iloc[0])}, "
    f"{snap['Vehicle_ID'].nunique()} vehicles, {snap['UAV_ID'].nunique()} UAVs)"
)

plt.savefig("uav_iov_graph.png", dpi=120)
print("\nSaved: uav_iov_graph.png")
print("Train GAT on graph link scores: python train_gat.py")

plt.show()
