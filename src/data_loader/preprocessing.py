"""
preprocessing.py — Point cloud and tabular data preprocessing utilities.

Includes:
  - Point cloud normalization (zero-mean, unit sphere)
  - Data augmentation (rotation, jitter, scaling)
  - Tabular feature encoding + scaling
  - Train/Val/Test splitting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import DATA, TABULAR, AUGMENTATION


# ─────────────────────────────────────────────
# POINT CLOUD NORMALIZATION
# ─────────────────────────────────────────────

def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Normalize a point cloud to zero mean and unit sphere.

    Steps:
      1. Translate centroid of XYZ to origin.
      2. Scale so that the furthest point lies on a unit sphere.

    Args:
        points (np.ndarray): Shape (N, 6) — [x, y, z, nx, ny, nz]

    Returns:
        np.ndarray: Normalized point cloud, same shape.
    """
    pts = points.copy().astype(np.float32)

    # Translate: subtract centroid of coordinates
    centroid = np.mean(pts[:, :3], axis=0)
    pts[:, :3] -= centroid

    # Scale: divide by furthest distance from origin
    furthest = np.max(np.sqrt(np.sum(pts[:, :3] ** 2, axis=1)))
    if furthest > 1e-6:
        pts[:, :3] /= furthest

    # Normals: re-normalize to unit length (they may have been perturbed)
    if pts.shape[1] == 6:
        norms = np.linalg.norm(pts[:, 3:6], axis=1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)  # avoid division by zero
        pts[:, 3:6] /= norms

    return pts


# ─────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────

def random_rotate_z(points: np.ndarray) -> np.ndarray:
    """
    Apply a random rotation around the Z-axis (gravity direction).

    Args:
        points (np.ndarray): Shape (N, 6)

    Returns:
        np.ndarray: Rotated point cloud.
    """
    pts = points.copy()
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # 2D rotation matrix for XY plane
    rot = np.array([[cos_a, -sin_a, 0],
                    [sin_a,  cos_a, 0],
                    [0,      0,     1]], dtype=np.float32)

    pts[:, :3] = pts[:, :3] @ rot.T     # rotate coordinates
    pts[:, 3:6] = pts[:, 3:6] @ rot.T  # rotate normals consistently
    return pts


def random_jitter(points: np.ndarray,
                  std: float = AUGMENTATION["jitter_std"],
                  clip: float = AUGMENTATION["jitter_clip"]) -> np.ndarray:
    """
    Add Gaussian noise to point coordinates only (not normals).

    Args:
        points (np.ndarray): Shape (N, 6)
        std  (float): Standard deviation of noise.
        clip (float): Max absolute noise value.

    Returns:
        np.ndarray: Jittered point cloud.
    """
    pts = points.copy()
    noise = np.clip(np.random.normal(0, std, size=pts[:, :3].shape),
                    -clip, clip).astype(np.float32)
    pts[:, :3] += noise
    return pts


def random_scale(points: np.ndarray,
                 low: float = AUGMENTATION["scale_low"],
                 high: float = AUGMENTATION["scale_high"]) -> np.ndarray:
    """
    Apply random uniform scaling to the point cloud.

    Args:
        points (np.ndarray): Shape (N, 6)
        low, high (float): Scale range.

    Returns:
        np.ndarray: Scaled point cloud.
    """
    pts = points.copy()
    scale = np.random.uniform(low, high)
    pts[:, :3] *= scale
    return pts


def augment_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Apply all enabled augmentations from config sequentially.

    Args:
        points (np.ndarray): Shape (N, 6)

    Returns:
        np.ndarray: Augmented point cloud.
    """
    if AUGMENTATION["random_rotation"]:
        points = random_rotate_z(points)
    if AUGMENTATION["jitter"]:
        points = random_jitter(points)
    if AUGMENTATION["random_scale"]:
        points = random_scale(points)
    return points


# ─────────────────────────────────────────────
# POINT CLOUD RESAMPLING
# ─────────────────────────────────────────────

def resample_point_cloud(points: np.ndarray, num_points: int = DATA["num_points"]) -> np.ndarray:
    """
    Resample a point cloud to exactly `num_points` points.
    - If too many: random subsample without replacement.
    - If too few: random oversample with replacement.

    Args:
        points (np.ndarray): Shape (N, 6)
        num_points (int): Target number of points.

    Returns:
        np.ndarray: Shape (num_points, 6)
    """
    n = len(points)
    if n == num_points:
        return points
    elif n > num_points:
        # Subsample without replacement
        idx = np.random.choice(n, num_points, replace=False)
    else:
        # Oversample with replacement
        idx = np.random.choice(n, num_points, replace=True)
    return points[idx]


# ─────────────────────────────────────────────
# TABULAR FEATURE PREPROCESSING
# ─────────────────────────────────────────────

def build_tabular_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer for tabular features.

    Pipeline:
      - Categorical columns: OneHotEncoder (handle unknown -> ignore)
      - Numerical columns: StandardScaler

    Args:
        df (pd.DataFrame): DataFrame with tabular feature columns.

    Returns:
        ColumnTransformer: Fitted transformer (call fit_transform / transform).
    """
    cat_cols = [c for c in TABULAR["categorical_cols"] if c in df.columns]
    num_cols = [c for c in TABULAR["numerical_cols"] if c in df.columns]

    transformers = []

    if num_cols:
        num_pipeline = Pipeline([("scaler", StandardScaler())])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


# ─────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────

def split_dataframe(df: pd.DataFrame,
                    val_split: float = DATA["val_split"],
                    test_split: float = DATA["test_split"],
                    seed: int = DATA["random_seed"]):
    """
    Split a DataFrame into train, validation, and test sets.

    Args:
        df (pd.DataFrame): Full dataset.
        val_split  (float): Fraction for validation.
        test_split (float): Fraction for test.
        seed       (int):   Random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train, val, test)
    """
    train_val, test = train_test_split(df, test_size=test_split, random_state=seed)
    val_frac = val_split / (1.0 - test_split)
    train, val = train_test_split(train_val, test_size=val_frac, random_state=seed)
    # NOTE: Do NOT reset_index — the original index is used by PartCostingDataset
    # to derive part file names (e.g. row 42 -> part_0042.txt).
    return train, val, test


# ─────────────────────────────────────────────
# UTILITY: Load a raw point cloud file
# ─────────────────────────────────────────────

def load_point_cloud_file(filepath: str) -> np.ndarray:
    """
    Load a point cloud from a .txt or .csv file.

    Expected format: 2048 rows × 6 columns (x, y, z, nx, ny, nz)
    Delimiter: space, comma, or tab.

    Args:
        filepath (str): Path to the file.

    Returns:
        np.ndarray: Shape (N, 6)

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has fewer than 3 columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Point cloud file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        try:
            data = pd.read_csv(filepath, header=None).values.astype(np.float32)
        except ValueError:
            data = pd.read_csv(filepath, header=0).values.astype(np.float32)
    else:
        # Try space/tab delimited
        try:
            data = np.loadtxt(filepath, dtype=np.float32)
        except ValueError:
            data = np.loadtxt(filepath, dtype=np.float32, skiprows=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(f"Point cloud must have ≥3 columns (x,y,z). Got {data.shape[1]}.")

    # Pad normals with zeros if only XYZ is provided
    if data.shape[1] == 3:
        zeros = np.zeros((data.shape[0], 3), dtype=np.float32)
        data = np.hstack([data, zeros])

    return data[:, :6]  # Keep only first 6 columns
