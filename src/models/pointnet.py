"""
pointnet.py — PointNet Encoder for 3D point cloud feature extraction.

Architecture (Qi et al., 2017 — "PointNet: Deep Learning on Point Sets"):
  1. Input T-Net (3×3 spatial transform)
  2. Shared MLP: 64, 64
  3. Feature T-Net (64×64 transform)  [optional]
  4. Shared MLP: 64, 128, 1024
  5. Global Max Pooling
  -> Output: 1024-dimensional global feature vector per sample

Input shape : (B, C, N)  — C = 3 (xyz) or 6 (xyz + normals)
Output shape: (B, 1024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tnet import InputTNet, FeatureTNet
from src.utils.config import MODEL


class PointNetEncoder(nn.Module):
    """
    PointNet encoder that extracts a global 1024-d feature vector
    from an unordered point cloud.

    Args:
        input_channels      (int):  Number of input channels per point.
                                    Use 3 for XYZ, 6 for XYZ+normals.
        use_feature_transform (bool): Apply 64×64 feature T-Net.
        output_dim          (int):  Dimension of global feature (default 1024).
    """

    def __init__(
        self,
        input_channels: int = 6 if MODEL.get("use_feature_transform") else 3,
        use_feature_transform: bool = MODEL["use_feature_transform"],
        output_dim: int = MODEL["pointnet_output_dim"],
    ):
        super().__init__()
        self.input_channels = input_channels
        self.use_feature_transform = use_feature_transform
        self.output_dim = output_dim

        # ── Input Transform: T-Net on XYZ only ────────────────────────
        # Even if input has normals (C=6), the spatial transform
        # is applied to XYZ (first 3 dims) only.
        self.input_tnet = InputTNet()   # Operates on 3-d XYZ

        # ── Shared MLP Block 1: after input transform ──────────────────
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64,          kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(64)

        # ── Feature Transform: optional 64×64 T-Net ────────────────────
        if self.use_feature_transform:
            self.feature_tnet = FeatureTNet()   # Operates on 64-d features

        # ── Shared MLP Block 2 ─────────────────────────────────────────
        self.conv3 = nn.Conv1d(64, 64,        kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128,       kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, output_dim, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm1d(64)
        self.bn4   = nn.BatchNorm1d(128)
        self.bn5   = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): Shape (B, C, N)
                B = batch size, C = channels (3 or 6), N = num_points

        Returns:
            global_feat (Tensor): Shape (B, output_dim)  — 1024-d vector
            trans_input (Tensor): Shape (B, 3, 3)         — input transform
            trans_feat  (Tensor|None): Shape (B, 64, 64) — feature transform
        """
        B, C, N = x.shape

        # ── Input Transform ────────────────────────────────────────────
        # T-Net only sees XYZ (first 3 channels); normals remain untouched
        xyz = x[:, :3, :]                        # (B, 3, N)
        trans_input = self.input_tnet(xyz)        # (B, 3, 3)
        xyz_t = torch.bmm(trans_input, xyz)       # (B, 3, N) — aligned coords

        # Recombine with normals if present
        if C > 3:
            x = torch.cat([xyz_t, x[:, 3:, :]], dim=1)  # (B, 6, N)
        else:
            x = xyz_t                                     # (B, 3, N)

        # ── Shared MLP Block 1 ─────────────────────────────────────────
        x = F.relu(self.bn1(self.conv1(x)))       # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))       # (B, 64, N)

        # ── Feature Transform ──────────────────────────────────────────
        trans_feat = None
        if self.use_feature_transform:
            trans_feat = self.feature_tnet(x)     # (B, 64, 64)
            # batch matrix multiply: (B, 64, 64) × (B, 64, N) -> (B, 64, N)
            x = torch.bmm(trans_feat, x)

        # ── Shared MLP Block 2 ─────────────────────────────────────────
        x = F.relu(self.bn3(self.conv3(x)))       # (B, 64,  N)
        x = F.relu(self.bn4(self.conv4(x)))       # (B, 128, N)
        x = F.relu(self.bn5(self.conv5(x)))       # (B, 1024, N)

        # ── Global Max Pooling ─────────────────────────────────────────
        global_feat = torch.max(x, dim=2)[0]      # (B, 1024)

        return global_feat, trans_input, trans_feat
