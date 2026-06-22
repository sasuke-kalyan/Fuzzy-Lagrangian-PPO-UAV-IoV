"""
Graph Attention Network (GAT) for UAV–IoV link quality prediction.

Pure PyTorch (no torch-geometric) for simpler installs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_data import EDGE_DIM, GraphSample, NODE_DIM


def _softmax_by_target(scores: torch.Tensor, target: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax over incoming edges for each target node. scores: [E, H]."""
    out = torch.zeros_like(scores)
    for node in range(num_nodes):
        mask = target == node
        if mask.any():
            out[mask] = F.softmax(scores[mask], dim=0)
    return out


class GATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = dropout

        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        n_nodes = x.size(0)
        h = self.lin(x).view(n_nodes, self.heads, self.out_dim)
        row, col = edge_index

        e = (
            (h[row] * self.attn_src).sum(dim=-1)
            + (h[col] * self.attn_dst).sum(dim=-1)
        )
        e = self.leaky_relu(e)
        alpha = _softmax_by_target(e, col, n_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros(n_nodes, self.heads, self.out_dim, device=x.device, dtype=x.dtype)
        msg = h[row] * alpha.unsqueeze(-1)
        out.index_add_(0, col, msg)

        if self.concat:
            return out.reshape(n_nodes, self.heads * self.out_dim)
        return out.mean(dim=1)


class UAVIoVGAT(nn.Module):
    """
    GAT node encoder + edge MLP predicts link quality (shaped reward).
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden: int = 32,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gat1 = GATLayer(node_dim, hidden, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATLayer(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
        self.dropout = dropout

        mlp_in = 2 * hidden + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, edge_index)
        return F.elu(h)

    def predict_edges(
        self,
        h: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index
        pair = torch.cat([h[row], h[col], edge_attr], dim=-1)
        return self.edge_mlp(pair).squeeze(-1)

    def forward(self, sample: GraphSample) -> torch.Tensor:
        h = self.encode(sample.x, sample.edge_index)
        return self.predict_edges(h, sample.edge_index, sample.edge_attr)

    def forward_routing(self, sample: GraphSample) -> torch.Tensor:
        """Scores for directed vehicle->UAV edges only."""
        h = self.encode(sample.x, sample.edge_index)
        return self.predict_edges(h, sample.edge_index, sample.edge_attr)
