"""
evaluate.py — Model evaluation on held-out test set.

Computes:
  - MAE  (Mean Absolute Error)
  - MSE  (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - R²   (Coefficient of Determination)

Outputs:
  - Console summary table
  - predictions.csv in outputs/predictions/
  - Residual / scatter plots saved to outputs/predictions/
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import PATHS, get_device
from src.training.loss import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)


# ─────────────────────────────────────────────
# COLLECT PREDICTIONS
# ─────────────────────────────────────────────

def collect_predictions(model, loader, device=None):
    """
    Run inference on an entire DataLoader and return all predictions + targets.

    Args:
        model  : HybridCostModel (already moved to device).
        loader : DataLoader (val or test).
        device : torch.device.

    Returns:
        preds   (np.ndarray): Shape (N,) — predicted costs.
        targets (np.ndarray): Shape (N,) — ground-truth costs.
    """
    if device is None:
        device = get_device()

    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for pc, tab, cost in tqdm(loader, desc="  Inference", leave=False, ncols=80):
            pc   = pc.to(device)
            tab  = tab.to(device)
            cost = cost.to(device)

            pred, _, _ = model(pc, tab)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(cost.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return preds, targets


# ─────────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute all regression metrics.

    Args:
        preds   (np.ndarray): Predicted values.
        targets (np.ndarray): Ground-truth values.

    Returns:
        dict with keys: mae, mse, rmse, mape, r2
    """
    p = torch.from_numpy(preds).float()
    t = torch.from_numpy(targets).float()

    mae  = mean_absolute_error(p, t).item()
    mse  = mean_squared_error(p, t).item()
    rmse = root_mean_squared_error(p, t).item()
    mape = mean_absolute_percentage_error(p, t).item() * 100  # as %

    # R² score
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    return {
        "MAE":  mae,
        "MSE":  mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2":   r2,
    }


# ─────────────────────────────────────────────
# SAVE PREDICTIONS
# ─────────────────────────────────────────────

def save_predictions(preds: np.ndarray, targets: np.ndarray, output_dir: str):
    """
    Save predicted vs actual costs to a CSV file.

    Args:
        preds      : Predicted costs.
        targets    : Actual costs.
        output_dir : Directory to save predictions.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({
        "actual_cost":    targets,
        "predicted_cost": preds,
        "residual":       targets - preds,
        "abs_error":      np.abs(targets - preds),
        "pct_error":      np.abs((targets - preds) / (np.abs(targets) + 1e-8)) * 100,
    })
    path = os.path.join(output_dir, "predictions.csv")
    df.to_csv(path, index=False)
    print(f"[Evaluate] Predictions saved → {path}")
    return df


# ─────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────

def plot_results(preds: np.ndarray, targets: np.ndarray, output_dir: str):
    """
    Generate and save:
      1. Scatter plot (Actual vs Predicted)
      2. Residual plot
      3. Error distribution histogram

    Args:
        preds, targets : numpy arrays.
        output_dir     : Directory to save figures.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[Plot] matplotlib/seaborn not installed — skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    residuals = targets - preds

    # ── 1. Actual vs Predicted ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, preds, alpha=0.6, edgecolors="none", color="#4C72B0", s=40)
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Cost (₹)", fontsize=12)
    ax.set_ylabel("Predicted Cost (₹)", fontsize=12)
    ax.set_title("Actual vs Predicted Manufacturing Cost", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    # ── 2. Residual Plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(preds, residuals, alpha=0.6, edgecolors="none", color="#DD8452", s=40)
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted Cost (₹)", fontsize=12)
    ax.set_ylabel("Residual (Actual − Predicted)", fontsize=12)
    ax.set_title("Residual Plot", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals.png"), dpi=150)
    plt.close()

    # ── 3. Error Distribution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color="#55A868", ax=ax, bins=30)
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Residual", fontsize=12)
    ax.set_title("Residual Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_distribution.png"), dpi=150)
    plt.close()

    print(f"[Plot] Figures saved → {output_dir}")


# ─────────────────────────────────────────────
# MAIN EVALUATE FUNCTION
# ─────────────────────────────────────────────

def evaluate(model, test_loader, device=None, save_outputs: bool = True) -> dict:
    """
    Full evaluation pipeline:
      1. Collect predictions on test set.
      2. Compute all metrics.
      3. Print summary table.
      4. Save predictions CSV + plots.

    Args:
        model        : HybridCostModel (already trained).
        test_loader  : Test DataLoader.
        device       : torch.device.
        save_outputs : Whether to save CSV and plots.

    Returns:
        metrics (dict): {'MAE', 'MSE', 'RMSE', 'MAPE', 'R2'}
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    print("\n" + "=" * 60)
    print("  EVALUATION — Test Set")
    print("=" * 60)

    # ── Inference ─────────────────────────────────────────────────────
    preds, targets = collect_predictions(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(preds, targets)

    # ── Print Summary ─────────────────────────────────────────────────
    print(f"\n  {'Metric':<8}  {'Value':>12}")
    print(f"  {'-'*22}")
    print(f"  {'MAE':<8}  {metrics['MAE']:>12.4f}")
    print(f"  {'MSE':<8}  {metrics['MSE']:>12.4f}")
    print(f"  {'RMSE':<8}  {metrics['RMSE']:>12.4f}")
    print(f"  {'MAPE':<8}  {metrics['MAPE']:>11.2f}%")
    print(f"  {'R²':<8}  {metrics['R2']:>12.4f}")
    print(f"  {'-'*22}\n")

    # ── Save outputs ───────────────────────────────────────────────────
    if save_outputs:
        out_dir = PATHS["output_preds"]
        save_predictions(preds, targets, out_dir)
        plot_results(preds, targets, out_dir)

        # Save metrics JSON
        metrics_path = os.path.join(out_dir, "metrics.json")
        os.makedirs(out_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Evaluate] Metrics saved → {metrics_path}")

    return metrics
