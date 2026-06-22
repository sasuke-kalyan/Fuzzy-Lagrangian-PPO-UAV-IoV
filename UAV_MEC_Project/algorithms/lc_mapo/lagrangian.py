from __future__ import annotations

import torch


class LagrangianMultiplierController:
    def __init__(self, cost_limits: list[float], lr: float, device: str):
        self.cost_limits = torch.tensor(cost_limits, dtype=torch.float32, device=device).view(1, -1)
        self.log_lambdas = torch.zeros((1, len(cost_limits)), dtype=torch.float32, device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.log_lambdas], lr=lr)

    @property
    def lambdas(self) -> torch.Tensor:
        return torch.exp(self.log_lambdas)

    def update(self, estimated_costs: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        diff = estimated_costs.detach().mean(dim=0, keepdim=True) - self.cost_limits
        loss = -(self.log_lambdas * diff).sum()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())
