"""
feature_fusion.py — Hybrid feature fusion module.

Combines the 1024-d global feature from PointNet with
the tabular engineering features via simple concatenation.

This module also houses the complete HybridCostModel:
  PointNetEncoder -> (global_feat)
                                  ↘
                                   Concatenate -> RegressionHead -> cost
  TabularEncoder  -> (tab_feat)   ↗

The TabularEncoder is a small 2-layer MLP that projects raw
tabular features into a richer embedding space before fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pointnet import PointNetEncoder
from src.models.regression_head import RegressionHead
from src.utils.config import MODEL, DATA


# ─────────────────────────────────────────────
# TABULAR ENCODER (lightweight MLP)
# ─────────────────────────────────────────────

class TabularEncoder(nn.Module):
    """
    Small MLP to embed raw tabular features into a learnable space
    before fusion with PointNet output.

    Architecture:
      FC(tabular_dim -> 128) -> BN -> ReLU -> Dropout(0.2)
      FC(128         -> 128) -> BN -> ReLU

    Args:
        tabular_dim (int): Number of (encoded) tabular input features.
        embed_dim   (int): Output embedding dimension. Default: 128.
    """

    def __init__(self, tabular_dim: int, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        self.fc1 = nn.Linear(tabular_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(128, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Shape (B, tabular_dim) — raw tabular features.

        Returns:
            embed (Tensor): Shape (B, embed_dim)
        """
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = F.relu(self.bn2(self.fc2(x)))
        return x


# ─────────────────────────────────────────────
# FEATURE FUSION
# ─────────────────────────────────────────────

class FeatureFusion(nn.Module):
    """
    Concatenates PointNet global features with tabular embeddings.

    Input:
      - point_feat  (Tensor): (B, pointnet_dim)  — from PointNet encoder
      - tabular_feat(Tensor): (B, tabular_embed) — from TabularEncoder

    Output:
      - fused       (Tensor): (B, pointnet_dim + tabular_embed)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        point_feat: torch.Tensor,
        tabular_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate along the feature dimension."""
        return torch.cat([point_feat, tabular_feat], dim=1)


# ─────────────────────────────────────────────
# COMPLETE HYBRID MODEL
# ─────────────────────────────────────────────

class HybridCostModel(nn.Module):
    """
    End-to-end hybrid model for manufacturing cost regression.

    Data Flow:
      point_cloud (B, C, N)
          └─► PointNetEncoder ─────────────────► global_feat (B, 1024)
                                                              │
      tabular (B, tabular_dim)                               │
          └─► TabularEncoder ──────────────────► tab_embed  (B, 128)
                                                              │
                                              ┌──────────────┘
                                        Concat (B, 1152)
                                              │
                                     RegressionHead
                                              │
                                        cost (B,)

    Args:
        tabular_dim (int): Dimension of (preprocessed) tabular features.
        input_channels (int): Channels per point (3 or 6).
        tabular_embed_dim (int): Hidden dim for TabularEncoder.
    """

    def __init__(
        self,
        tabular_dim: int,
        input_channels: int = 6 if DATA["use_normals"] else 3,
        tabular_embed_dim: int = 128,
    ):
        super().__init__()

        self.pointnet = PointNetEncoder(
            input_channels=input_channels,
            use_feature_transform=MODEL["use_feature_transform"],
            output_dim=MODEL["pointnet_output_dim"],
        )

        self.tabular_encoder = TabularEncoder(
            tabular_dim=tabular_dim,
            embed_dim=tabular_embed_dim,
        )

        self.fusion = FeatureFusion()

        fused_dim = MODEL["pointnet_output_dim"] + tabular_embed_dim

        self.regression_head = RegressionHead(
            input_dim=fused_dim,
            fc_layers=MODEL["fc_layers"],
            dropout_rate=MODEL["dropout_rate"],
        )

    def forward(
        self,
        point_cloud: torch.Tensor,
        tabular: torch.Tensor,
    ):
        """
        Args:
            point_cloud (Tensor): (B, C, N) — point cloud batch.
            tabular     (Tensor): (B, tabular_dim) — tabular features.

        Returns:
            cost       (Tensor): (B,) — predicted costs.
            trans_input(Tensor): (B, 3, 3) — input transform matrices.
            trans_feat (Tensor|None): (B, 64, 64) — feature transforms.
        """
        # ── 3D Branch ─────────────────────────────────────────────────
        global_feat, trans_input, trans_feat = self.pointnet(point_cloud)

        # ── Tabular Branch ─────────────────────────────────────────────
        tab_embed = self.tabular_encoder(tabular)

        # ── Fusion ─────────────────────────────────────────────────────
        fused = self.fusion(global_feat, tab_embed)

        # ── Regression ─────────────────────────────────────────────────
        cost = self.regression_head(fused)

        return cost, trans_input, trans_feat

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
