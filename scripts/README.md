# Training Scripts

This directory contains scripts for batch training and evaluation of all ML models.

## Quick Start

### Linux/macOS/Git Bash
```bash
./scripts/run_all_local.sh
```

### Windows (Command Prompt)
```batch
scripts\run_all_local.bat
```

## Scripts Overview

### 1. `run_all_local.sh` (Linux/macOS/Git Bash)
Bash script that:
- Creates necessary directories
- Trains all models sequentially
- Handles errors gracefully (continues on failure)
- Generates benchmark summary
- Reports success/failure counts

**Usage:**
```bash
chmod +x scripts/run_all_local.sh
./scripts/run_all_local.sh
```

### 2. `run_all_local.bat` (Windows)
Windows batch script with identical functionality:
- Creates necessary directories
- Trains all models sequentially
- Handles errors gracefully
- Generates benchmark summary
- Reports success/failure counts

**Usage:**
```batch
scripts\run_all_local.bat
```

### 3. `generate_summary.py` (Cross-platform)
Python script to generate benchmark summary report.

**Features:**
- Collects metrics from all model artifacts
- Generates markdown comparison table
- Identifies best performing models
- Handles missing/failed models gracefully

**Usage:**
```bash
python scripts/generate_summary.py
```

**Output:** `results/benchmark_summary.md`

## Models Trained

The scripts train the following models in sequence:

1. **XGBoost** - Gradient boosting with tree-based learners
2. **Random Forest** - Ensemble of decision trees
3. **MLP** - Multi-layer perceptron neural network
4. **Logistic Regression** - Linear classification
5. **SVM** - Support vector machine with RBF kernel

## Directory Structure

After execution, the following structure is created:

```
.
├── artifacts/
│   ├── xgboost/
│   │   ├── model.pkl
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix.csv
│   │   └── metrics.json
│   ├── random_forest/
│   ├── mlp/
│   ├── logistic_regression/
│   └── svm/
├── results/
│   └── benchmark_summary.md
└── scripts/
    ├── run_all_local.sh
    ├── run_all_local.bat
    └── generate_summary.py
```

## Output Files

### Per-Model Artifacts

Each model generates:
- `model.*` - Trained model file (`.pkl` or `.pt`)
- `confusion_matrix.png` - Visualization of predictions
- `confusion_matrix.csv` - Raw confusion matrix data
- `metrics.json` - Performance metrics (accuracy, precision, recall, F1, ROC-AUC)

### Benchmark Summary

`results/benchmark_summary.md` contains:
- Comparison table of all models
- Best performing model highlights
- Training time statistics
- Links to artifacts

## Configuration

All models use the unified configuration from `configs/local.yaml`:

```yaml
data:
  csv_path: "balanced_features.csv"
  target_column: "label"
  train_size: 600
  val_size: 200
  test_size: 200
  random_state: 42

training:
  n_jobs: -1
  verbose: 1
```

## Error Handling

The scripts are designed to be fault-tolerant:
- **Continue on failure**: If one model fails, others continue
- **Error reporting**: Failed models are tracked and reported
- **Missing metrics**: Summary handles missing/incomplete results
- **Exit codes**: Non-zero exit if any model fails

## Example Output

```
==================================================
NASA Exoplanet ML - Batch Training
Started: 2025-10-05 14:30:00
==================================================

Creating directories...

==================================================
Training: xgboost
==================================================
✓ xgboost completed successfully (12.5s)

==================================================
Training: random_forest
==================================================
✓ random_forest completed successfully (8.3s)

...

==================================================
Batch Training Complete
==================================================
Finished: 2025-10-05 14:32:15

Successful: 5
Failed: 0

Results saved to: results/benchmark_summary.md
==================================================
```

## Troubleshooting

### Permission Denied (Linux/macOS)
```bash
chmod +x scripts/run_all_local.sh
```

### Python Not Found
Ensure Python 3.8+ is installed and in PATH:
```bash
python --version
```

### Missing Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

### CUDA/GPU Issues
If GPU training fails, models will fall back to CPU automatically.

## Advanced Usage

### Run Individual Model
```bash
python -m src.models.xgboost.train --config configs/local.yaml
```

### Custom Configuration
```bash
python -m src.models.xgboost.train --config configs/custom.yaml
```

### Regenerate Summary Only
```bash
python scripts/generate_summary.py
```

## Next Steps

After successful execution:
1. Review `results/benchmark_summary.md`
2. Inspect confusion matrices in `artifacts/{model}/`
3. Compare model performance
4. Select best model for deployment
5. Run inference with selected model

## For Google Colab

See `scripts/run_all_colab.sh` for Colab-optimized execution (coming soon).
