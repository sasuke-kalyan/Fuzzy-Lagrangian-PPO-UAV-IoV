"""
GAT-based UAV routing: pick highest-scored vehicle–UAV link per vehicle.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_dataset
import torch

from communication_model import shaped_reward
from gat_model import UAVIoVGAT
from graph_data import GraphSample, build_routing_graph, snapshot_at_timestamp

MODEL_PATH = Path("models/gat_uav_iov.pt")


def load_gat(device: torch.device | None = None) -> UAVIoVGAT:
    device = device or torch.device("cpu")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Train GAT first: python train_gat.py (missing {MODEL_PATH})")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = UAVIoVGAT(
        hidden=ckpt.get("hidden", 32),
        heads=ckpt.get("heads", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def best_uav_per_vehicle(model: UAVIoVGAT, df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    snap = snapshot_at_timestamp(df)
    rows = []

    for vehicle_id, group in snap.groupby("Vehicle_ID"):
        g = build_routing_graph(group)
        with torch.no_grad():
            sample = GraphSample(
                g.x.to(device),
                g.edge_index.to(device),
                g.edge_attr.to(device),
                g.y.to(device),
                g.vehicle_ids,
                g.uav_ids,
                g.node_names,
            )
            scores = model.forward_routing(sample).cpu().numpy()

        best_i = int(scores.argmax())
        rewards = group.apply(
            lambda r: shaped_reward(
                r["Signal_Strength"], r["PDR"], r["Delay"], r["Energy"]
            ),
            axis=1,
        )
        rows.append(
            {
                "Vehicle_ID": vehicle_id,
                "GAT_UAV": g.uav_ids[best_i],
                "GAT_Score": round(float(scores[best_i]), 2),
                "True_Best_UAV": group.loc[rewards.idxmax()]["UAV_ID"],
            }
        )

    return pd.DataFrame(rows)


def main():
    print("\n======================================")
    print(" GAT Smart Routing ")
    print("======================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_gat(device)
    df = load_dataset()

    routes = best_uav_per_vehicle(model, df, device)
    print(routes.to_string(index=False))

    match = (routes["GAT_UAV"] == routes["True_Best_UAV"]).mean()
    print(f"\nGAT matches reward-optimal UAV: {match * 100:.1f}% of vehicles")
    print("\nGAT routing completed.\n")


if __name__ == "__main__":
    main()
