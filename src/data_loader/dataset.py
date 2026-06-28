"""
dataset.py — PyTorch Dataset for PointNet Part Costing.

Loads:
  - Point cloud files (N × 6) from data/raw/
  - Tabular features + cost labels from data/labels.csv

Returns per sample:
  (point_cloud_tensor, tabular_tensor, cost_tensor)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import DATA, TABULAR, AUGMENTATION, PATHS, TRAINING
from src.data_loader.preprocessing import (
    load_point_cloud_file,
    normalize_point_cloud,
    resample_point_cloud,
    augment_point_cloud,
    build_tabular_preprocessor,
    split_dataframe,
)


class PartCostingDataset(Dataset):
    """
    PyTorch Dataset for manufacturing part cost prediction.

    Each sample consists of:
      - point_cloud : FloatTensor  (num_points, 6)  — x,y,z,nx,ny,nz
      - tabular     : FloatTensor  (tabular_dim,)   — encoded features
      - cost        : FloatTensor  scalar            — manufacturing cost

    Args:
        df              (pd.DataFrame): Rows from labels.csv for this split.
        tabular_transformer: Fitted sklearn ColumnTransformer for tabular data.
        raw_data_dir    (str):  Path to the folder containing point cloud files.
        split           (str):  One of 'train' | 'val' | 'test'.
        augment         (bool): Apply data augmentation (train only).
        num_points      (int):  Points to sample per cloud.
        use_normals     (bool): Include nx,ny,nz columns.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tabular_transformer,
        raw_data_dir: str,
        split: str = "train",
        augment: bool = False,
        num_points: int = DATA["num_points"],
        use_normals: bool = DATA["use_normals"],
    ):
        self.df = df.reset_index(drop=True)
        self.raw_data_dir = raw_data_dir
        self.split = split
        self.augment = augment and AUGMENTATION["enabled"]
        self.num_points = num_points
        self.use_normals = use_normals
        self.target_col = TABULAR["target_col"]
        self.part_id_col = TABULAR["part_id_col"]

        # Pre-transform all tabular features at once for efficiency
        self.tabular_features = tabular_transformer.transform(df).astype(np.float32)

        # Cache cost values as numpy array
        self.costs = df[self.target_col].values.astype(np.float32)

        # Part IDs (used to find point cloud files)
        self.part_ids = df[self.part_id_col].astype(str).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Returns:
            point_cloud (FloatTensor): Shape (C, N) — channels first for Conv1d
                C = 3 if use_normals=False, else 6
            tabular (FloatTensor): Shape (tabular_dim,)
            cost (FloatTensor): Scalar
        """
        # ── Load point cloud ───────────────────────────────────────────
        part_id = self.part_ids[idx]
        pc = self._load_point_cloud(part_id)            # (N, 6)
        pc = resample_point_cloud(pc, self.num_points)  # ensure exact size
        pc = normalize_point_cloud(pc)                  # zero mean, unit sphere

        if self.augment:
            pc = augment_point_cloud(pc)                # random rotation/jitter

        # Select channels
        if not self.use_normals:
            pc = pc[:, :3]  # (N, 3) — XYZ only

        # Transpose to (C, N) for Conv1d compatibility
        pc_tensor = torch.from_numpy(pc).float().T     # (C, N)

        # ── Tabular features ───────────────────────────────────────────
        tab_tensor = torch.from_numpy(self.tabular_features[idx]).float()

        # ── Cost (target) ──────────────────────────────────────────────
        cost_tensor = torch.tensor(self.costs[idx], dtype=torch.float32)

        return pc_tensor, tab_tensor, cost_tensor

    def _load_point_cloud(self, part_id: str) -> np.ndarray:
        """
        Attempt to load point cloud for `part_id`.
        Tries .txt and .csv extensions in raw_data_dir.

        Falls back to a zero array if file not found (for demo purposes).
        In production, raise the FileNotFoundError instead.
        """
        for ext in [".txt", ".csv"]:
            path = os.path.join(self.raw_data_dir, f"{part_id}{ext}")
            if os.path.exists(path):
                return load_point_cloud_file(path)

        # ── Fallback: synthetic data (remove in production) ──────────
        # Generate a random point cloud so the pipeline can be tested
        # without real data files.
        return _generate_synthetic_point_cloud(self.num_points)


# ─────────────────────────────────────────────
# DATA MODULE: Builds all three DataLoaders
# ─────────────────────────────────────────────

class PartCostingDataModule:
    """
    Convenience class that:
      1. Loads labels.csv
      2. Splits into train / val / test
      3. Fits the tabular preprocessor on train data only
      4. Exposes get_dataloaders() → (train_loader, val_loader, test_loader)
      5. Saves the fitted preprocessor to disk for inference

    Args:
        labels_csv   (str): Path to labels.csv
        raw_data_dir (str): Path to folder with point cloud files
        batch_size   (int): Batch size for DataLoaders
        num_workers  (int): DataLoader worker threads
    """

    def __init__(
        self,
        labels_csv: str = PATHS["labels_csv"],
        raw_data_dir: str = PATHS["raw_data"],
        batch_size: int = TRAINING["batch_size"],
        num_workers: int = TRAINING["num_workers"],
    ):
        self.labels_csv = labels_csv
        self.raw_data_dir = raw_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.tabular_transformer = None
        self.tabular_dim = None

    def setup(self):
        """Load CSV, split, fit tabular preprocessor."""
        if not os.path.exists(self.labels_csv):
            raise FileNotFoundError(
                f"labels.csv not found at: {self.labels_csv}\n"
                "Run `python main.py --generate-demo` to create example data."
            )

        df = pd.read_csv(self.labels_csv)
        self._validate_labels(df)

        # Fill missing optional columns
        for col in TABULAR["numerical_cols"]:
            if col not in df.columns:
                df[col] = 0.0

        # Split
        self.train_df, self.val_df, self.test_df = split_dataframe(df)

        # Fit tabular preprocessor on TRAIN data only (no leakage)
        self.tabular_transformer = build_tabular_preprocessor(self.train_df)
        self.tabular_transformer.fit(self.train_df)

        # Determine output dimensionality
        sample = self.tabular_transformer.transform(self.train_df[:1])
        self.tabular_dim = sample.shape[1]

        print(f"[DataModule] Train: {len(self.train_df)} | "
              f"Val: {len(self.val_df)} | Test: {len(self.test_df)}")
        print(f"[DataModule] Tabular feature dim: {self.tabular_dim}")

    def _validate_labels(self, df: pd.DataFrame):
        """Ensure required columns exist in labels.csv."""
        required = [TABULAR["part_id_col"], TABULAR["target_col"]]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in labels.csv: {missing}")

    def get_dataloaders(self):
        """
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.train_df is None:
            self.setup()

        train_dataset = PartCostingDataset(
            self.train_df, self.tabular_transformer,
            self.raw_data_dir, split="train", augment=AUGMENTATION["enabled"]
        )
        val_dataset = PartCostingDataset(
            self.val_df, self.tabular_transformer,
            self.raw_data_dir, split="val", augment=False
        )
        test_dataset = PartCostingDataset(
            self.test_df, self.tabular_transformer,
            self.raw_data_dir, split="test", augment=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=TRAINING["pin_memory"],
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=TRAINING["pin_memory"]
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=TRAINING["pin_memory"]
        )

        return train_loader, val_loader, test_loader

    def save_preprocessor(self, path: str):
        """Save fitted tabular transformer for later inference use."""
        with open(path, "wb") as f:
            pickle.dump(self.tabular_transformer, f)
        print(f"[DataModule] Preprocessor saved to: {path}")

    @staticmethod
    def load_preprocessor(path: str):
        """Load a previously saved tabular transformer."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
# HELPER: Synthetic point cloud generator
# ─────────────────────────────────────────────

def _generate_synthetic_point_cloud(num_points: int = 2048) -> np.ndarray:
    """
    Generate a synthetic unit-sphere point cloud for testing.
    Normals are set as outward-pointing unit vectors.

    Returns:
        np.ndarray: Shape (num_points, 6)
    """
    # Random points on a unit sphere surface
    xyz = np.random.randn(num_points, 3).astype(np.float32)
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = xyz / (norms + 1e-8)

    # Normals = same direction as point on sphere
    normals = xyz.copy()

    return np.hstack([xyz, normals])
