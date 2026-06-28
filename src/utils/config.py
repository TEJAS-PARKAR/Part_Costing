"""
config.py — Central configuration for the PointNet Part Costing project.

All hyperparameters, paths, and model settings are defined here.
Import this module anywhere in the project to access configs consistently.
"""

import os

# ─────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATHS = {
    "raw_data":       os.path.join(BASE_DIR, "data", "raw"),
    "processed_data": os.path.join(BASE_DIR, "data", "processed"),
    "labels_csv":     os.path.join(BASE_DIR, "data", "labels.csv"),
    "output_models":  os.path.join(BASE_DIR, "outputs", "models"),
    "output_logs":    os.path.join(BASE_DIR, "outputs", "logs"),
    "output_preds":   os.path.join(BASE_DIR, "outputs", "predictions"),
}

# ─────────────────────────────────────────────
# DATA SETTINGS
# ─────────────────────────────────────────────
DATA = {
    "num_points":       2048,       # Number of points per point cloud
    "point_features":   6,          # x, y, z, nx, ny, nz
    "use_normals":      True,       # Whether to include normals (nx, ny, nz)
    "val_split":        0.15,       # Fraction of data for validation
    "test_split":       0.10,       # Fraction of data for test
    "random_seed":      42,
}

# ─────────────────────────────────────────────
# TABULAR FEATURES
# ─────────────────────────────────────────────
TABULAR = {
    # Categorical column names (will be one-hot encoded)
    "categorical_cols": ["material_type", "manufacturing_process"],

    # Numerical column names (will be standardized)
    "numerical_cols":   ["weight", "volume", "machining_time"],

    # Target column in labels.csv
    "target_col":       "cost",

    # Part identifier column
    "part_id_col":      "part_id",
}

# ─────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────
AUGMENTATION = {
    "enabled":          True,
    "random_rotation":  True,       # Random rotation around Z-axis
    "jitter":           True,       # Add Gaussian noise to point positions
    "jitter_std":       0.01,       # Standard deviation of jitter noise
    "jitter_clip":      0.05,       # Max absolute jitter value
    "random_scale":     False,      # Random scaling
    "scale_low":        0.8,
    "scale_high":       1.25,
}

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
MODEL = {
    "pointnet_output_dim":  1024,   # Global feature vector size from PointNet
    "use_feature_transform": True,  # Enable T-Net for 64-d feature space

    # Regression head layer sizes
    "fc_layers":            [512, 256, 64],
    "dropout_rate":         0.3,

    # TNet settings
    "tnet_input_dim":       3,      # Operates on XYZ only (not normals)
    "tnet_feature_dim":     64,     # Feature transform dimension
}

# ─────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
TRAINING = {
    "batch_size":           16,
    "num_epochs":           100,
    "learning_rate":        0.001,
    "weight_decay":         1e-4,   # L2 regularization
    "lr_scheduler":         True,   # Use StepLR
    "lr_step_size":         20,     # Decay every N epochs
    "lr_gamma":             0.5,    # Multiply LR by this factor
    "early_stopping":       True,
    "patience":             15,     # Stop if val loss doesn't improve for N epochs
    "gradient_clip":        1.0,    # Max gradient norm (None to disable)
    "num_workers":          0,      # DataLoader workers (0 = main process)
    "pin_memory":           False,  # Set True if using GPU
    "device":               "auto", # 'auto' | 'cpu' | 'cuda' | 'mps'
}

# ─────────────────────────────────────────────
# LOGGING & CHECKPOINTING
# ─────────────────────────────────────────────
LOGGING = {
    "tensorboard":      True,       # Enable TensorBoard logging
    "log_interval":     5,          # Log every N epochs
    "save_best_only":   True,       # Only save the best checkpoint
    "checkpoint_name":  "best_model.pth",
    "final_model_name": "final_model.pth",
}

# ─────────────────────────────────────────────
# HELPER: Get device string
# ─────────────────────────────────────────────
def get_device():
    """Return the torch device to use based on config."""
    import torch
    if TRAINING["device"] == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(TRAINING["device"])


# ─────────────────────────────────────────────
# HELPER: Ensure all output directories exist
# ─────────────────────────────────────────────
def ensure_dirs():
    """Create all output directories if they don't exist."""
    for key, path in PATHS.items():
        if not path.endswith(".csv"):   # Skip file paths
            os.makedirs(path, exist_ok=True)
