"""
loss.py — Loss functions for the part costing regression task.

Primary loss : MSELoss (Mean Squared Error)
Metric       : MAE     (Mean Absolute Error)
Regulariser  : Feature-transform orthogonality loss (PointNet paper, Eq. 2)
               L_reg = ||I - A·Aᵀ||²_F
               This penalises the feature T-Net when its matrix strays
               from being orthogonal, stabilising training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# PRIMARY REGRESSION LOSS
# ─────────────────────────────────────────────

class MSELoss(nn.Module):
    """Thin wrapper around PyTorch MSELoss for consistency."""

    def __init__(self):
        super().__init__()
        self._loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   (Tensor): (B,) — model predictions.
            target (Tensor): (B,) — ground-truth costs.

        Returns:
            Scalar MSE loss.
        """
        return self._loss(pred, target)


# ─────────────────────────────────────────────
# EVALUATION METRIC
# ─────────────────────────────────────────────

def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error.

    Args:
        pred   (Tensor): (B,) — predictions.
        target (Tensor): (B,) — ground truth.

    Returns:
        Scalar MAE.
    """
    return F.l1_loss(pred, target)


def mean_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Mean Squared Error."""
    return F.mse_loss(pred, target)


def root_mean_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Root Mean Squared Error."""
    return torch.sqrt(F.mse_loss(pred, target))


def mean_absolute_percentage_error(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute MAPE — Mean Absolute Percentage Error.
    Avoids division by zero with epsilon.

    Returns:
        Scalar MAPE (as a fraction, multiply by 100 for %).
    """
    return torch.mean(torch.abs((target - pred) / (torch.abs(target) + eps)))


# ─────────────────────────────────────────────
# FEATURE TRANSFORM REGULARISATION LOSS
# ─────────────────────────────────────────────

def feature_transform_regularizer(trans_feat: torch.Tensor, weight: float = 0.001) -> torch.Tensor:
    """
    Orthogonality regularisation for the feature T-Net matrix.
    (Eq. 2 in Qi et al. "PointNet", CVPR 2017)

    Penalises deviation from orthogonality:
        L_reg = ||I − A·Aᵀ||²_F

    Args:
        trans_feat (Tensor): Shape (B, K, K) — feature transform matrices.
        weight     (float): Scaling coefficient. Default 0.001.

    Returns:
        Scalar regularisation loss (pre-scaled).
    """
    if trans_feat is None:
        return torch.tensor(0.0)

    B, K, _ = trans_feat.shape
    I = torch.eye(K, device=trans_feat.device).unsqueeze(0).expand(B, -1, -1)
    diff = torch.bmm(trans_feat, trans_feat.transpose(1, 2)) - I  # (B, K, K)
    reg_loss = torch.mean(torch.sum(diff ** 2, dim=(1, 2)))
    return weight * reg_loss


# ─────────────────────────────────────────────
# COMBINED LOSS
# ─────────────────────────────────────────────

class CostingLoss(nn.Module):
    """
    Total training loss:
        L_total = MSE(pred, target) + λ · L_reg(trans_feat)

    Args:
        reg_weight (float): Weight for orthogonality regularisation.
    """

    def __init__(self, reg_weight: float = 0.001):
        super().__init__()
        self.reg_weight = reg_weight
        self.mse = MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        trans_feat: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            pred       (Tensor): (B,) — predictions.
            target     (Tensor): (B,) — ground truth.
            trans_feat (Tensor): (B, 64, 64) — feature transform matrices.

        Returns:
            dict with keys:
                'total'  — combined loss (backward on this)
                'mse'    — MSE component
                'reg'    — regularisation component
                'mae'    — MAE metric (no gradient tracked)
        """
        mse = self.mse(pred, target)
        reg = feature_transform_regularizer(trans_feat, self.reg_weight)
        total = mse + reg

        with torch.no_grad():
            mae = mean_absolute_error(pred * 1000.0, target * 1000.0)

        return {
            "total": total,
            "mse":   mse,
            "reg":   reg,
            "mae":   mae,
        }
