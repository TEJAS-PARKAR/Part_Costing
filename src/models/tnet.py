"""
tnet.py — T-Net (Spatial Transformer Network) for PointNet.

Two variants:
  1. InputTNet   — 3×3 transform on raw XYZ coordinates
  2. FeatureTNet — 64×64 transform on learned feature space

Both learn to predict a rotation matrix from the input itself,
making the network invariant to rigid-body transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    Generic T-Net: predicts a K×K transformation matrix from
    an input of shape (B, K, N).

    Architecture (following original PointNet paper):
      Conv1d(K,  64)  → BN → ReLU
      Conv1d(64, 128) → BN → ReLU
      Conv1d(128,1024)→ BN → ReLU
      GlobalMaxPool
      FC(1024, 512) → BN → ReLU
      FC(512,  256) → BN → ReLU
      FC(256,  K*K)
      Reshape → K×K matrix
      Add identity matrix (residual)

    The residual addition ensures the network starts as an identity
    transform and learns deviations from it.

    Args:
        k (int): Dimension of the transform (3 for input, 64 for features).
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        # ── Shared MLP (implemented as Conv1d for point-wise ops) ──────
        self.conv1 = nn.Conv1d(k, 64,   kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)

        # ── Batch Normalisation ────────────────────────────────────────
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # ── Fully-connected head ───────────────────────────────────────
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)   # Outputs flattened K×K matrix

        # ── Identity matrix buffer (not a learnable parameter) ─────────
        # Registered as a buffer so it moves to the correct device automatically
        self.register_buffer("identity", torch.eye(k).flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Shape (B, K, N) — batch of point features.

        Returns:
            transform (Tensor): Shape (B, K, K) — transformation matrices.
        """
        # ── Shared MLP ─────────────────────────────────────────────────
        x = F.relu(self.bn1(self.conv1(x)))    # (B, 64,   N)
        x = F.relu(self.bn2(self.conv2(x)))    # (B, 128,  N)
        x = F.relu(self.bn3(self.conv3(x)))    # (B, 1024, N)

        # ── Global max pooling ─────────────────────────────────────────
        x = torch.max(x, dim=2)[0]             # (B, 1024)

        # ── FC layers ──────────────────────────────────────────────────
        x = F.relu(self.bn4(self.fc1(x)))      # (B, 512)
        x = F.relu(self.bn5(self.fc2(x)))      # (B, 256)
        x = self.fc3(x)                        # (B, K*K)

        # ── Add identity (residual) ────────────────────────────────────
        # This initialises the output as a near-identity transform
        x = x + self.identity                  # broadcast over batch
        transform = x.view(-1, self.k, self.k) # (B, K, K)

        return transform


class InputTNet(TNet):
    """
    Input T-Net: 3×3 spatial transform on raw XYZ point coordinates.

    Usage:
        tnet = InputTNet()
        transform = tnet(xyz_input)     # xyz_input: (B, 3, N)
        aligned   = torch.bmm(transform, xyz_input)
    """

    def __init__(self):
        super().__init__(k=3)


class FeatureTNet(TNet):
    """
    Feature T-Net: 64×64 transform on the intermediate feature space.

    Usage:
        tnet = FeatureTNet()
        transform  = tnet(features)         # features: (B, 64, N)
        aligned_f  = torch.bmm(transform, features)
    """

    def __init__(self):
        super().__init__(k=64)
