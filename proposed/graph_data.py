"""
Build UAV–IoV communication graphs from the dataset for GAT training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from communication_model import shaped_reward


@dataclass
class GraphSample:
    """One graph snapshot (e.g. one timestamp)."""

    x: torch.Tensor              # [N, node_dim]
    edge_index: torch.LongTensor  # [2, E]
    edge_attr: torch.Tensor      # [E, edge_dim]
    y: torch.Tensor              # [E] link quality targets
    vehicle_ids: list[str]       # length E
    uav_ids: list[str]           # length E
    node_names: list[str]        # length N


NODE_DIM = 5
EDGE_DIM = 5


def _edge_features(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            float(row["Signal_Strength"]),
            float(row["Delay"]) / 100.0,
            float(row["PDR"]) / 100.0,
            float(row["Distance"]) / 1000.0,
            float(row["Energy"]) / 50.0,
        ],
        dtype=np.float32,
    )


def _node_features(node_name: str, incident: pd.DataFrame) -> np.ndarray:
    is_vehicle = 1.0 if node_name.startswith("V") else 0.0
    is_uav = 1.0 if node_name.startswith("U") else 0.0

    if len(incident) == 0:
        return np.array([is_vehicle, is_uav, 0.0, 0.0, 0.0], dtype=np.float32)

    return np.array(
        [
            is_vehicle,
            is_uav,
            incident["Signal_Strength"].mean(),
            incident["Delay"].mean() / 100.0,
            incident["PDR"].mean() / 100.0 if is_vehicle else incident["Energy"].mean() / 50.0,
        ],
        dtype=np.float32,
    )


def build_graph_from_rows(rows: pd.DataFrame) -> GraphSample:
    """Undirected bipartite graph; each row is one vehicle–UAV link."""
    rows = rows.reset_index(drop=True)
    node_names = sorted(
        set(rows["Vehicle_ID"].unique()) | set(rows["UAV_ID"].unique()),
        key=lambda s: (s[0], s),
    )
    node_index = {name: i for i, name in enumerate(node_names)}

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_attrs: list[np.ndarray] = []
    targets: list[float] = []
    vehicles: list[str] = []
    uavs: list[str] = []

    for _, row in rows.iterrows():
        v = row["Vehicle_ID"]
        u = row["UAV_ID"]
        vi, ui = node_index[v], node_index[u]
        attr = _edge_features(row)
        target = shaped_reward(
            row["Signal_Strength"],
            row["PDR"],
            row["Delay"],
            row["Energy"],
        )

        for i, j in ((vi, ui), (ui, vi)):
            edge_src.append(i)
            edge_dst.append(j)
            edge_attrs.append(attr)
            targets.append(target)
            vehicles.append(v)
            uavs.append(u)

    x = np.stack(
        [
            _node_features(name, rows[(rows["Vehicle_ID"] == name) | (rows["UAV_ID"] == name)])
            for name in node_names
        ],
        axis=0,
    )

    route_v: list[str] = []
    route_u: list[str] = []
    for _, row in rows.iterrows():
        route_v.append(row["Vehicle_ID"])
        route_u.append(row["UAV_ID"])

    return GraphSample(
        x=torch.from_numpy(x),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_attr=torch.from_numpy(np.stack(edge_attrs, axis=0)),
        y=torch.tensor(targets, dtype=torch.float32),
        vehicle_ids=route_v,
        uav_ids=route_u,
        node_names=node_names,
    )


def build_routing_graph(rows: pd.DataFrame) -> GraphSample:
    """Directed vehicle->UAV edges only (for inference / routing)."""
    rows = rows.reset_index(drop=True)
    node_names = sorted(
        set(rows["Vehicle_ID"].unique()) | set(rows["UAV_ID"].unique()),
        key=lambda s: (s[0], s),
    )
    node_index = {name: i for i, name in enumerate(node_names)}

    edge_src, edge_dst, edge_attrs, targets = [], [], [], []
    vehicles, uavs = [], []

    for _, row in rows.iterrows():
        v, u = row["Vehicle_ID"], row["UAV_ID"]
        edge_src.append(node_index[v])
        edge_dst.append(node_index[u])
        edge_attrs.append(_edge_features(row))
        targets.append(
            shaped_reward(
                row["Signal_Strength"],
                row["PDR"],
                row["Delay"],
                row["Energy"],
            )
        )
        vehicles.append(v)
        uavs.append(u)

    x = np.stack(
        [
            _node_features(name, rows[(rows["Vehicle_ID"] == name) | (rows["UAV_ID"] == name)])
            for name in node_names
        ],
        axis=0,
    )

    return GraphSample(
        x=torch.from_numpy(x),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_attr=torch.from_numpy(np.stack(edge_attrs, axis=0)),
        y=torch.tensor(targets, dtype=torch.float32),
        vehicle_ids=vehicles,
        uav_ids=uavs,
        node_names=node_names,
    )


def iter_graphs(
    df: pd.DataFrame,
    group_col: str = "Timestamp",
    routing_only: bool = False,
) -> list[GraphSample]:
    builder = build_routing_graph if routing_only else build_graph_from_rows
    graphs: list[GraphSample] = []
    for _, group in df.groupby(group_col):
        if len(group) == 0:
            continue
        graphs.append(builder(group))
    return graphs


def snapshot_at_timestamp(df: pd.DataFrame, timestamp: int | None = None) -> pd.DataFrame:
    """All vehicle–UAV links at one timestep."""
    if timestamp is None:
        timestamp = int(df["Timestamp"].max())
    return df[df["Timestamp"] == timestamp].copy()


def iter_vehicle_snapshots(df: pd.DataFrame):
    """Yield (timestamp, vehicle_id, rows) with every UAV link for that vehicle."""
    for timestamp, snap in df.groupby("Timestamp"):
        for vehicle_id, group in snap.groupby("Vehicle_ID"):
            yield int(timestamp), str(vehicle_id), group.reset_index(drop=True)


def train_val_split(
    graphs: list[GraphSample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[GraphSample], list[GraphSample]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(graphs))
    rng.shuffle(idx)
    n_val = max(1, int(len(graphs) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train = [graphs[i] for i in range(len(graphs)) if i not in val_idx]
    val = [graphs[i] for i in range(len(graphs)) if i in val_idx]
    return train, val
