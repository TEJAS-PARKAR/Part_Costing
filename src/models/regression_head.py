"""
regression_head.py — Fully-connected regression head for cost prediction.

Accepts a concatenated feature vector:
  [PointNet global features (1024-d)] + [Tabular features (tabular_dim)]

Architecture:
  FC(1024 + tabular_dim → 512) → BN → ReLU → Dropout(0.3)
  FC(512               → 256) → BN → ReLU → Dropout(0.3)
  FC(256               → 64)  → BN → ReLU
  FC(64                → 1)   → Linear (no activation, raw regression output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import MODEL


class RegressionHead(nn.Module):
    """
    Multi-layer fully-connected regression head.

    Args:
        input_dim    (int):   Total input feature size (PointNet + tabular).
        fc_layers    (list):  Hidden layer sizes. Default: [512, 256, 64].
        dropout_rate (float): Dropout probability. Default: 0.3.
    """

    def __init__(
        self,
        input_dim: int,
        fc_layers: list = MODEL["fc_layers"],
        dropout_rate: float = MODEL["dropout_rate"],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        # ── Build fully-connected layers dynamically ───────────────────
        layers = []
        in_dim = input_dim

        for i, out_dim in enumerate(fc_layers):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))

            # Dropout after every layer except the last hidden layer
            # (keeps the final 64-d representation more stable)
            if i < len(fc_layers) - 1:
                layers.append(nn.Dropout(p=dropout_rate))

            in_dim = out_dim

        # ── Output layer (no activation → raw regression value) ────────
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Shape (B, input_dim) — fused feature vector.

        Returns:
            cost (Tensor): Shape (B,) — predicted manufacturing cost.
        """
        out = self.network(x)       # (B, 1)
        return out.squeeze(-1)      # (B,)
