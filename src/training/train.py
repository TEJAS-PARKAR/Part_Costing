"""
train.py — Training loop for the HybridCostModel.

Features:
  - Full train / validation epoch loops
  - Early stopping with patience
  - LR scheduling (StepLR)
  - Gradient clipping
  - TensorBoard logging
  - Checkpoint saving (best val loss + final model)
  - Detailed per-epoch console output
"""

import os
import sys
import time
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import TRAINING, LOGGING, PATHS, get_device
from src.training.loss import CostingLoss


# ─────────────────────────────────────────────
# TRAINING STATE TRACKER
# ─────────────────────────────────────────────

class TrainingHistory:
    """Stores per-epoch metrics for later plotting / analysis."""

    def __init__(self):
        self.train_loss = []
        self.val_loss   = []
        self.train_mae  = []
        self.val_mae    = []
        self.lr         = []

    def update(self, train_loss, val_loss, train_mae, val_mae, lr):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_mae.append(train_mae)
        self.val_mae.append(val_mae)
        self.lr.append(lr)

    def to_dict(self):
        return {
            "train_loss": self.train_loss,
            "val_loss":   self.val_loss,
            "train_mae":  self.train_mae,
            "val_mae":    self.val_mae,
            "lr":         self.lr,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[History] Saved training history -> {path}")


# ─────────────────────────────────────────────
# SINGLE EPOCH: TRAIN
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    """
    Run one full training epoch.

    Args:
        model      : HybridCostModel
        loader     : Training DataLoader
        optimizer  : torch optimizer
        criterion  : CostingLoss instance
        device     : torch.device
        grad_clip  : Optional float — max gradient norm

    Returns:
        dict: {'loss': float, 'mae': float}
    """
    model.train()
    total_loss = 0.0
    total_mae  = 0.0
    n_batches  = 0

    for pc, tab, cost in tqdm(loader, desc="  Train", leave=False, ncols=80):
        pc   = pc.to(device)       # (B, C, N)
        tab  = tab.to(device)      # (B, tabular_dim)
        cost = cost.to(device)     # (B,)

        optimizer.zero_grad()

        # Forward pass
        pred, trans_input, trans_feat = model(pc, tab)

        # Compute loss
        losses = criterion(pred, cost, trans_feat)
        loss   = losses["total"]

        # Backward + clip + step
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += losses["mse"].item()
        total_mae  += losses["mae"].item()
        n_batches  += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae":  total_mae  / max(n_batches, 1),
    }


# ─────────────────────────────────────────────
# SINGLE EPOCH: VALIDATE
# ─────────────────────────────────────────────

def validate_one_epoch(model, loader, criterion, device):
    """
    Run one full validation epoch (no gradient updates).

    Returns:
        dict: {'loss': float, 'mae': float}
    """
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0
    n_batches  = 0

    with torch.no_grad():
        for pc, tab, cost in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
            pc   = pc.to(device)
            tab  = tab.to(device)
            cost = cost.to(device)

            pred, _, trans_feat = model(pc, tab)
            losses = criterion(pred, cost, trans_feat)

            total_loss += losses["mse"].item()
            total_mae  += losses["mae"].item()
            n_batches  += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae":  total_mae  / max(n_batches, 1),
    }


# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────

def train(
    model,
    train_loader,
    val_loader,
    num_epochs:    int   = TRAINING["num_epochs"],
    lr:            float = TRAINING["learning_rate"],
    weight_decay:  float = TRAINING["weight_decay"],
    lr_step_size:  int   = TRAINING["lr_step_size"],
    lr_gamma:      float = TRAINING["lr_gamma"],
    patience:      int   = TRAINING["patience"],
    grad_clip:     float = TRAINING["gradient_clip"],
    device=None,
):
    """
    Full training pipeline with validation, scheduling, and checkpointing.

    Args:
        model        : HybridCostModel instance (not yet moved to device).
        train_loader : Training DataLoader.
        val_loader   : Validation DataLoader.
        num_epochs   : Max training epochs.
        lr           : Initial learning rate.
        weight_decay : L2 regularisation coefficient.
        lr_step_size : Epochs between LR reductions.
        lr_gamma     : LR reduction factor.
        patience     : Early stopping patience (epochs without improvement).
        grad_clip    : Max gradient norm (None to disable).
        device       : torch.device (auto-detected if None).

    Returns:
        model        : Best model loaded from checkpoint.
        history      : TrainingHistory object.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    print(f"\n{'='*60}")
    print(f"  Training on: {device}")
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  Epochs: {num_epochs}  |  LR: {lr}  |  Patience: {patience}")
    print(f"{'='*60}\n")

    # ── Optimizer & Scheduler ──────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # ── Loss ──────────────────────────────────────────────────────────
    criterion = CostingLoss(reg_weight=0.001)

    # ── TensorBoard ───────────────────────────────────────────────────
    writer = None
    if LOGGING["tensorboard"]:
        log_dir = PATHS["output_logs"]
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"  TensorBoard logs -> {log_dir}")
        print(f"  Run: tensorboard --logdir {log_dir}\n")

    # ── State ─────────────────────────────────────────────────────────
    history = TrainingHistory()
    best_val_loss  = float("inf")
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(PATHS["output_models"], LOGGING["checkpoint_name"])
    os.makedirs(PATHS["output_models"], exist_ok=True)

    # ── Training Loop ─────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        t_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        val_metrics   = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t_start

        # ── Log ───────────────────────────────────────────────────────
        history.update(
            train_metrics["loss"], val_metrics["loss"],
            train_metrics["mae"],  val_metrics["mae"],
            current_lr,
        )

        if writer:
            writer.add_scalar("Loss/Train",    train_metrics["loss"], epoch)
            writer.add_scalar("Loss/Val",      val_metrics["loss"],   epoch)
            writer.add_scalar("MAE/Train",     train_metrics["mae"],  epoch)
            writer.add_scalar("MAE/Val",       val_metrics["mae"],    epoch)
            writer.add_scalar("LR",            current_lr,            epoch)

        # ── Console output ────────────────────────────────────────────
        print(
            f"[Epoch {epoch:>4}/{num_epochs}] "
            f"TrainLoss: {train_metrics['loss']:.4f}  "
            f"ValLoss: {val_metrics['loss']:.4f}  "
            f"TrainMAE: {train_metrics['mae']:.4f}  "
            f"ValMAE: {val_metrics['mae']:.4f}  "
            f"LR: {current_lr:.2e}  "
            f"({elapsed:.1f}s)"
        )

        # ── Checkpointing ─────────────────────────────────────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            _save_checkpoint(model, optimizer, epoch, val_metrics, best_ckpt_path)
            print(f"           * New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # ── Early stopping ────────────────────────────────────────────
        if TRAINING["early_stopping"] and epochs_no_improve >= patience:
            print(f"\n[Early Stop] No improvement for {patience} epochs. Stopping.")
            break

    if writer:
        writer.close()

    # ── Save final model ───────────────────────────────────────────────
    final_path = os.path.join(PATHS["output_models"], LOGGING["final_model_name"])
    _save_checkpoint(model, optimizer, epoch, val_metrics, final_path)
    print(f"\n[Train] Final model saved -> {final_path}")

    # ── Load best model ────────────────────────────────────────────────
    print(f"[Train] Loading best checkpoint from -> {best_ckpt_path}")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── Save history ───────────────────────────────────────────────────
    history_path = os.path.join(PATHS["output_logs"], "training_history.json")
    os.makedirs(PATHS["output_logs"], exist_ok=True)
    history.save(history_path)

    return model, history


# ─────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────

def _save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model state, optimizer state, and metadata."""
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":            metrics["loss"],
        "val_mae":             metrics["mae"],
    }, path)


def load_checkpoint(model, path: str, device=None):
    """
    Load a saved checkpoint into model.

    Args:
        model: HybridCostModel instance.
        path : Path to .pth checkpoint file.
        device: torch.device

    Returns:
        model, epoch, val_loss
    """
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Checkpoint] Loaded epoch {ckpt['epoch']} | val_loss={ckpt['val_loss']:.4f}")
    return model, ckpt["epoch"], ckpt["val_loss"]
