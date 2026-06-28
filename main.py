"""
main.py — Entry point for the PointNet Part Costing pipeline.

Usage:
    python main.py                     # Full train + eval pipeline
    python main.py --generate-demo     # Create synthetic demo dataset then train
    python main.py --eval-only         # Evaluate a saved checkpoint
    python main.py --epochs 50         # Override epoch count
    python main.py --batch-size 8      # Override batch size
    python main.py --no-augment        # Disable data augmentation
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

# ── Ensure project root is on path ────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import (
    PATHS, DATA, TABULAR, TRAINING, AUGMENTATION, LOGGING,
    get_device, ensure_dirs,
)
from src.data_loader.dataset import PartCostingDataModule
from src.fusion.feature_fusion import HybridCostModel
from src.training.train import train, load_checkpoint
from src.evaluation.evaluate import evaluate


# ─────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PointNet-Based Deep Learning Model for Part Costing"
    )
    parser.add_argument("--generate-demo", action="store_true",
                        help="Generate synthetic demo dataset before training.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training and evaluate a saved checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to load for eval-only mode.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: cpu | cuda | mps")
    return parser.parse_args()


# ─────────────────────────────────────────────
# DEMO DATA GENERATOR
# ─────────────────────────────────────────────

def generate_demo_dataset(n_parts: int = 200):
    """
    Create a synthetic dataset for end-to-end pipeline testing.

    Generates:
      - data/labels.csv  with tabular features + costs
      - data/raw/<part_id>.txt files with (2048, 6) point clouds

    The synthetic cost is computed from a physically-inspired formula
    so the model has a learnable signal.

    Args:
        n_parts (int): Number of synthetic parts to generate.
    """
    print(f"\n[Demo] Generating {n_parts} synthetic parts...")
    raw_dir = PATHS["raw_data"]
    os.makedirs(raw_dir, exist_ok=True)

    materials = ["Steel", "Aluminum", "Titanium", "Plastic", "Cast_Iron"]
    processes = ["Turning", "Milling", "Drilling", "Grinding", "Casting"]

    material_cost = {"Steel": 1.0, "Aluminum": 1.5, "Titanium": 4.0,
                     "Plastic": 0.4, "Cast_Iron": 0.8}
    process_cost  = {"Turning": 1.0, "Milling": 1.3, "Drilling": 0.8,
                     "Grinding": 1.8, "Casting": 1.2}

    records = []
    rng = np.random.RandomState(42)

    for i in range(n_parts):
        part_id = f"part_{i:04d}"
        material = rng.choice(materials)
        process  = rng.choice(processes)
        weight   = rng.uniform(0.5, 50.0)       # kg
        volume   = weight / rng.uniform(1.5, 8.0)  # cm³ (density variation)
        machining_time = rng.uniform(5, 300)    # minutes

        # ── Synthetic cost formula ─────────────────────────────────
        # Cost = base_material × weight × process_factor
        #        + machining_time × 0.5
        #        + random noise
        base_cost = (
            material_cost[material] * weight * 20 +
            process_cost[process] * machining_time * 0.5 +
            volume * 5 +
            rng.normal(0, 10)
        )
        cost = max(base_cost, 1.0)  # ensure positive cost

        records.append({
            "part_id":              part_id,
            "material_type":        material,
            "manufacturing_process": process,
            "weight":               round(weight, 3),
            "volume":               round(volume, 3),
            "machining_time":       round(machining_time, 1),
            "cost":                 round(cost, 2),
        })

        # ── Generate point cloud ───────────────────────────────────
        # Use a random mesh-like shape (sphere + perturbations)
        xyz = rng.randn(DATA["num_points"], 3).astype(np.float32)
        norms = np.linalg.norm(xyz, axis=1, keepdims=True)
        xyz = xyz / (norms + 1e-8)

        # Scale by volume factor
        scale = (volume ** (1/3)) * 0.1
        xyz = xyz * scale

        # Normals (outward)
        normals = xyz / (np.linalg.norm(xyz, axis=1, keepdims=True) + 1e-8)

        pc = np.hstack([xyz, normals])
        pc_path = os.path.join(raw_dir, f"{part_id}.txt")
        np.savetxt(pc_path, pc, fmt="%.6f")

    # ── Save labels CSV ────────────────────────────────────────────
    df = pd.DataFrame(records)
    labels_path = PATHS["labels_csv"]
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    df.to_csv(labels_path, index=False)

    print(f"[Demo] Labels saved  → {labels_path}")
    print(f"[Demo] Point clouds  → {raw_dir}/")
    print(f"[Demo] Cost range    → ₹{df['cost'].min():.2f} – ₹{df['cost'].max():.2f}")
    print(f"[Demo] Ready to train!\n")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Apply CLI overrides ────────────────────────────────────────────
    if args.epochs:
        TRAINING["num_epochs"] = args.epochs
    if args.batch_size:
        TRAINING["batch_size"] = args.batch_size
    if args.lr:
        TRAINING["learning_rate"] = args.lr
    if args.no_augment:
        AUGMENTATION["enabled"] = False
    if args.device:
        TRAINING["device"] = args.device

    device = get_device()

    # ── Ensure output directories exist ───────────────────────────────
    ensure_dirs()

    # ── Generate demo data if requested ───────────────────────────────
    if args.generate_demo:
        generate_demo_dataset(n_parts=200)

    # ── Data Module ───────────────────────────────────────────────────
    print("\n[Main] Setting up data...")
    data_module = PartCostingDataModule(
        labels_csv=PATHS["labels_csv"],
        raw_data_dir=PATHS["raw_data"],
        batch_size=TRAINING["batch_size"],
        num_workers=TRAINING["num_workers"],
    )
    data_module.setup()
    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    # Save fitted preprocessor for inference
    preprocessor_path = os.path.join(PATHS["output_models"], "tabular_preprocessor.pkl")
    data_module.save_preprocessor(preprocessor_path)

    # ── Model ─────────────────────────────────────────────────────────
    print("\n[Main] Initializing HybridCostModel...")
    input_channels = 6 if DATA["use_normals"] else 3
    model = HybridCostModel(
        tabular_dim=data_module.tabular_dim,
        input_channels=input_channels,
    )
    print(f"[Main] Model parameters: {model.count_parameters():,}")

    # ── Eval-only mode ────────────────────────────────────────────────
    if args.eval_only:
        ckpt_path = args.checkpoint or os.path.join(
            PATHS["output_models"], LOGGING["checkpoint_name"]
        )
        print(f"\n[Main] Eval-only mode. Loading: {ckpt_path}")
        model, epoch, val_loss = load_checkpoint(model, ckpt_path, device)
        model = model.to(device)
        metrics = evaluate(model, test_loader, device=device, save_outputs=True)
        print("\n[Main] Evaluation complete.")
        return metrics

    # ── Training ──────────────────────────────────────────────────────
    print("\n[Main] Starting training...")
    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=TRAINING["num_epochs"],
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
        lr_step_size=TRAINING["lr_step_size"],
        lr_gamma=TRAINING["lr_gamma"],
        patience=TRAINING["patience"],
        grad_clip=TRAINING["gradient_clip"],
        device=device,
    )

    # ── Evaluation ────────────────────────────────────────────────────
    metrics = evaluate(model, test_loader, device=device, save_outputs=True)

    # ── Final Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Best model saved  → {PATHS['output_models']}/{LOGGING['checkpoint_name']}")
    print(f"  Training history  → {PATHS['output_logs']}/training_history.json")
    print(f"  Predictions CSV   → {PATHS['output_preds']}/predictions.csv")
    print(f"  Plots             → {PATHS['output_preds']}/")
    print(f"\n  Final Test Metrics:")
    for k, v in metrics.items():
        unit = "%" if k == "MAPE" else ""
        print(f"    {k:<6} = {v:.4f}{unit}")
    print("=" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    main()
