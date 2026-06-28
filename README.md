# PointNet-Based Deep Learning Model for Part Costing

A production-ready **hybrid deep learning** system for predicting **manufacturing cost** of mechanical parts using:
- **3D Point Cloud** data processed by a **PointNet encoder**
- **Tabular engineering features** (material, process, weight, etc.)

---

## Project Structure

```
part_costing/
├── data/
│   ├── raw/                    # Point cloud files (.txt or .csv) — one per part
│   ├── processed/              # Reserved for preprocessed cache
│   └── labels.csv              # Part metadata + cost labels
│
├── src/
│   ├── data_loader/
│   │   ├── dataset.py          # PyTorch Dataset + DataModule
│   │   └── preprocessing.py    # Normalization, augmentation, encoding
│   │
│   ├── models/
│   │   ├── tnet.py             # Input & Feature T-Net (spatial transform)
│   │   ├── pointnet.py         # PointNet encoder → 1024-d feature
│   │   └── regression_head.py  # FC regression head → cost
│   │
│   ├── fusion/
│   │   └── feature_fusion.py   # TabularEncoder + FeatureFusion + HybridCostModel
│   │
│   ├── training/
│   │   ├── train.py            # Training loop + checkpointing
│   │   └── loss.py             # MSE loss + orthogonality regularizer + metrics
│   │
│   ├── evaluation/
│   │   └── evaluate.py         # MAE, MSE, RMSE, MAPE, R² + plots
│   │
│   └── utils/
│       └── config.py           # All hyperparameters and paths
│
├── outputs/
│   ├── models/                 # Saved model checkpoints (.pth)
│   ├── logs/                   # TensorBoard event files + training_history.json
│   └── predictions/            # predictions.csv + evaluation plots
│
├── main.py                     # Pipeline entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with Synthetic Demo Data

The fastest way to test the full pipeline end-to-end:

```bash
python main.py --generate-demo
```

This generates **200 synthetic parts** (point clouds + tabular features) and runs the complete train + evaluate pipeline.

### 3. Train on Your Own Data

Prepare your data in the format described below, then:

```bash
python main.py
```

### 4. Evaluate a Saved Checkpoint

```bash
python main.py --eval-only
# or specify a checkpoint:
python main.py --eval-only --checkpoint outputs/models/best_model.pth
```

### 5. Override Hyperparameters via CLI

```bash
python main.py --epochs 50 --batch-size 8 --lr 0.0005 --no-augment
```

---

## Data Format

### `data/labels.csv`

| Column | Type | Description |
|---|---|---|
| `part_id` | string | Unique part identifier (matches point cloud filename) |
| `material_type` | categorical | e.g., `Steel`, `Aluminum`, `Titanium` |
| `manufacturing_process` | categorical | e.g., `Turning`, `Milling`, `Grinding` |
| `weight` | float | Part weight in kg |
| `volume` | float | Part volume in cm³ |
| `machining_time` | float | Estimated machining time in minutes (optional) |
| `cost` | float | **Target**: manufacturing cost in ₹ |

### Point Cloud Files (`data/raw/<part_id>.txt`)

- One `.txt` or `.csv` file per part, named exactly `<part_id>.txt`
- Each file: **2048 rows × 6 columns**
- Columns: `x  y  z  nx  ny  nz` (3D coordinates + surface normals)
- Delimiter: space, comma, or tab

**Example file content** (`part_0001.txt`):
```
0.123456 -0.234567 0.456789 0.707107 0.000000 0.707107
-0.345678 0.456789 -0.123456 -0.577350 0.577350 0.577350
...
```

---

## Model Architecture

```
Input: Point Cloud (B, 6, 2048)
         │
    ┌────▼──────────────────────────────────────────────┐
    │  PointNet Encoder                                  │
    │   InputTNet (3×3)  →  Shared MLP (64, 64)         │
    │   FeatureTNet (64×64)  →  Shared MLP (64,128,1024)│
    │   Global Max Pooling                               │
    └────────────────────────────────┬──────────────────┘
                                     │ 1024-d
Input: Tabular Features (B, tab_dim) │
         │                           │
    ┌────▼──────────────────┐        │
    │  TabularEncoder        │        │
    │  FC(128) → FC(128)    │        │
    └────────────┬──────────┘        │
                 │ 128-d             │
                 └──────────┬────────┘
                          Concat
                        (B, 1152-d)
                             │
                    ┌────────▼────────┐
                    │  RegressionHead │
                    │  FC(512) → ReLU │
                    │  FC(256) → ReLU │
                    │  FC(64)  → ReLU │
                    │  FC(1)          │
                    └────────┬────────┘
                             │
                    Predicted Cost (B,)
```

---

## Configuration

All hyperparameters live in [`src/utils/config.py`](src/utils/config.py):

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 16 | Training batch size |
| `num_epochs` | 100 | Maximum training epochs |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `weight_decay` | 1e-4 | L2 regularization |
| `dropout_rate` | 0.3 | Dropout in regression head |
| `num_points` | 2048 | Points per point cloud |
| `use_normals` | True | Include nx,ny,nz channels |
| `early_stopping` | True | Stop if val loss plateaus |
| `patience` | 15 | Early stopping patience |

---

## Training Features

- ✅ **Early Stopping** — prevents overfitting
- ✅ **LR Scheduling** — StepLR decay every 20 epochs
- ✅ **Gradient Clipping** — stabilizes training
- ✅ **Orthogonality Regularization** — stabilizes T-Net learning
- ✅ **Data Augmentation** — random rotation (Z-axis), jitter, optional scaling
- ✅ **TensorBoard Logging** — loss + MAE curves per epoch
- ✅ **Best Checkpoint Saving** — saves model with lowest validation MSE

### View TensorBoard

```bash
tensorboard --logdir outputs/logs
```

---

## Outputs

After a training run:

| File | Description |
|---|---|
| `outputs/models/best_model.pth` | Best model checkpoint |
| `outputs/models/final_model.pth` | Final epoch checkpoint |
| `outputs/models/tabular_preprocessor.pkl` | Fitted sklearn preprocessor |
| `outputs/logs/training_history.json` | Per-epoch metrics |
| `outputs/predictions/predictions.csv` | Actual vs predicted + errors |
| `outputs/predictions/actual_vs_predicted.png` | Scatter plot |
| `outputs/predictions/residuals.png` | Residual plot |
| `outputs/predictions/residual_distribution.png` | Error histogram |
| `outputs/predictions/metrics.json` | Final test metrics |

---

## Evaluation Metrics

| Metric | Formula | Notes |
|---|---|---|
| **MAE** | mean\|actual − pred\| | Primary metric |
| **MSE** | mean(actual − pred)² | Training loss |
| **RMSE** | √MSE | Interpretable in ₹ |
| **MAPE** | mean\|(actual−pred)/actual\| × 100 | As % error |
| **R²** | 1 − SS_res/SS_tot | Variance explained |

---

## Dependencies

```
torch>=2.0.0          # Deep learning framework
torchvision>=0.15.0   # Utility transforms
numpy>=1.24.0         # Array operations
pandas>=2.0.0         # Tabular data
scikit-learn>=1.3.0   # Preprocessing + splits
matplotlib>=3.7.0     # Plots
seaborn>=0.12.0       # Statistical plots
tqdm>=4.65.0          # Progress bars
tensorboard>=2.13.0   # Training visualization
pyyaml>=6.0           # Config (optional extension)
scipy>=1.11.0         # Scientific utilities
```

---

## Citation

This project implements the PointNet architecture from:

> **Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017).**  
> PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.  
> *CVPR 2017*. [arxiv.org/abs/1612.00593](https://arxiv.org/abs/1612.00593)

---

## License

MIT License — free to use for academic and commercial projects.
