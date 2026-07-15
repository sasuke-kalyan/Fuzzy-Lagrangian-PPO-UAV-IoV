"""
Train GAT to predict link quality (shaped reward) on UAV–IoV graphs.

Usage:
    python train_gat.py
    python train_gat.py --epochs 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_loader import load_dataset
import torch
import torch.nn as nn

from gat_model import UAVIoVGAT
from graph_data import GraphSample, iter_graphs, train_val_split
import network_config as net

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "gat_uav_iov.pt"


def _loss_on_graph(
    model: UAVIoVGAT,
    graph: GraphSample,
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    sample = GraphSample(
        x=graph.x.to(device),
        edge_index=graph.edge_index.to(device),
        edge_attr=graph.edge_attr.to(device),
        y=graph.y.to(device),
        vehicle_ids=graph.vehicle_ids,
        uav_ids=graph.uav_ids,
        node_names=graph.node_names,
    )
    pred = model(sample)
    return criterion(pred, sample.y)


def train(epochs: int = 200, lr: float = 1e-3) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_dataset()
    graphs = iter_graphs(df, routing_only=False)

    train_graphs, val_graphs = train_val_split(graphs, val_ratio=0.2)

    model = UAVIoVGAT(hidden=64, heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for g in train_graphs:
            optimizer.zero_grad()
            loss = _loss_on_graph(model, g, criterion, device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_graphs), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for g in val_graphs:
                val_loss += _loss_on_graph(model, g, criterion, device).item()
        val_loss /= max(len(val_graphs), 1)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hidden": 64,
                    "heads": 4,
                    "epochs": epoch,
                    "val_loss": val_loss,
                },
                MODEL_PATH,
            )

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}"
            )

    print(
        f"\nFleet: {net.NUM_VEHICLES} vehicles, {net.NUM_UAVS} UAVs | "
        f"graphs: {len(graphs)} (train {len(train_graphs)}, val {len(val_graphs)})"
    )
    print(f"Best val MSE: {best_val:.4f}")
    print(f"Model saved to {MODEL_PATH}")
    print("Next: python evaluate_gat.py  or  python gat_routing.py")
    return MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Train GAT on UAV-IoV graphs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
