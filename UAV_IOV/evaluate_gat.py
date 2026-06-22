"""
Evaluate trained GAT: MSE on links and routing vs ground-truth best UAV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_dataset
import torch
import torch.nn as nn

from communication_model import shaped_reward
from gat_model import UAVIoVGAT
from graph_data import GraphSample, build_routing_graph, iter_graphs, iter_vehicle_snapshots, train_val_split

MODEL_PATH = Path("models/gat_uav_iov.pt")


def load_model(device: torch.device) -> UAVIoVGAT:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No GAT checkpoint at {MODEL_PATH}. Run: python train_gat.py")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = UAVIoVGAT(
        hidden=ckpt.get("hidden", 32),
        heads=ckpt.get("heads", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def routing_accuracy(model: UAVIoVGAT, df: pd.DataFrame, device: torch.device) -> float:
    correct = 0
    total = 0

    for _, vehicle_id, group in iter_vehicle_snapshots(df):
        if len(group) < 2:
            continue

        rewards = group.apply(
            lambda r: shaped_reward(
                r["Signal_Strength"], r["PDR"], r["Delay"], r["Energy"]
            ),
            axis=1,
        )
        best_row = group.loc[rewards.idxmax()]

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

        pred_uav = g.uav_ids[int(np.argmax(scores))]
        if pred_uav == best_row["UAV_ID"]:
            correct += 1
        total += 1

    return correct / total if total else 0.0


def main():
    print("\n======================================")
    print(" GAT Evaluation ")
    print("======================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    criterion = nn.MSELoss()

    df = load_dataset()
    graphs = iter_graphs(df, routing_only=False)
    _, val_graphs = train_val_split(graphs, val_ratio=0.2, seed=42)

    val_mse = 0.0
    with torch.no_grad():
        for g in val_graphs:
            sample = GraphSample(
                g.x.to(device),
                g.edge_index.to(device),
                g.edge_attr.to(device),
                g.y.to(device),
                g.vehicle_ids,
                g.uav_ids,
                g.node_names,
            )
            pred = model(sample)
            val_mse += criterion(pred, sample.y).item()
    val_mse /= max(len(val_graphs), 1)

    print(f"Validation link MSE: {val_mse:.4f}")

    acc = routing_accuracy(model, df, device)
    print(f"Per-vehicle routing match (vs max reward row): {acc * 100:.1f}%")

    print("\nEvaluation completed.\n")


if __name__ == "__main__":
    main()
