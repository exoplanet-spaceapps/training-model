# NASA Exoplanet ML Training Pipeline

> Unified machine learning pipeline for exoplanet detection using NASA TESS data

[![Tests](https://img.shields.io/badge/tests-119%2F119%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This project implements a unified, production-ready machine learning pipeline for detecting exoplanets from NASA TESS light curve features. The pipeline supports 6 different ML algorithms with consistent data processing, evaluation metrics, and comprehensive benchmarking.

### Key Features

- **Unified Data Pipeline**: Consistent preprocessing across all models
- **6 ML Algorithms**: XGBoost, Random Forest, MLP, Logistic Regression, SVM, CNN1D
- **GPU Optimization**: CNN1D with automatic GPU detection and PyTorch acceleration
- **Fixed Data Split**: 600 train / 200 validation / 200 test (1000 total samples)
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Automated Benchmarking**: Compare all models with visualizations and reports
- **Test-Driven Development**: 119 tests with 100% pass rate
- **Production-Ready**: Artifact saving, confusion matrices, JSON metrics

## Quick Start

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Train a Single Model

```bash
# XGBoost
python -m src.models.xgboost.train

# Random Forest
python -m src.models.random_forest.train

# MLP (Multi-Layer Perceptron)
python -m src.models.mlp.train

# Logistic Regression
python -m src.models.logistic_regression.train

# SVM (Support Vector Machine)
python -m src.models.svm.train

# CNN1D (1D Convolutional Neural Network with GPU support)
python -m src.models.cnn1d.train
```

### Run Complete Benchmark

```bash
# Train all 6 models and generate comparison report
python scripts/benchmark_all_models.py
```

This will generate:
- `results/benchmark_summary.md` - Detailed comparison report
- `results/benchmark_results.json` - Structured metrics data
- `results/benchmark_*.png` - 3 visualization charts
- `artifacts/{model}/` - Model artifacts for each algorithm

### View Results

```bash
# Open benchmark report
cat results/benchmark_summary.md

# View JSON results
cat results/benchmark_results.json
```

## Dataset Information

- **Source**: `balanced_features.csv` (1000 samples)
- **Features**: 13 statistical features extracted from TESS light curves
- **Target**: Binary classification (0 = non-exoplanet, 1 = exoplanet)
- **Split**: 600 train / 200 validation / 200 test
- **Stratified**: Yes (maintains class balance)
- **Random Seed**: 42 (for reproducibility)

### Feature Columns

- `flux_mean`, `flux_std`, `flux_median`, `flux_mad`
- `flux_skew`, `flux_kurt`
- `bls_period`, `bls_duration`, `bls_depth`, `bls_power`, `bls_snr`
- `n_sectors`

## Project Structure

```
training-model/
├── src/                          # Source code
│   ├── data_loader.py            # Unified data loading (load_and_split_data)
│   ├── preprocess.py             # Feature preprocessing (standardization)
│   ├── metrics.py                # Evaluation metrics and artifacts
│   └── models/                   # Model implementations
│       ├── xgboost/
│       │   ├── model.py          # XGBClassifierWrapper
│       │   └── train.py          # train_xgboost()
│       ├── random_forest/
│       │   ├── model.py          # RandomForestWrapper
│       │   └── train.py          # train_random_forest()
│       ├── mlp/
│       │   ├── model.py          # MLPWrapper
│       │   └── train.py          # train_mlp()
│       ├── logistic_regression/
│       │   ├── model.py          # LogisticRegressionWrapper
│       │   └── train.py          # train_logistic_regression()
│       ├── svm/
│       │   ├── model.py          # SVMWrapper
│       │   └── train.py          # train_svm()
│       └── cnn1d/
│           ├── model.py          # CNN1DWrapper (PyTorch with GPU)
│           └── train.py          # train_cnn1d()
├── configs/                      # Configuration files
│   ├── base.yaml                 # Base configuration
│   ├── local.yaml                # Local optimization settings
│   └── colab.yaml                # Google Colab settings
├── scripts/                      # Utility scripts
│   ├── benchmark_all_models.py   # Complete benchmarking
│   ├── run_all_local.sh          # Batch training (bash)
│   ├── run_all_local.bat         # Batch training (Windows)
│   └── run_all_local.py          # Batch training (Python)
├── tests/                        # Test suite (119 tests)
│   ├── test_data_loader.py       # Data loading tests (17)
│   ├── test_preprocess.py        # Preprocessing tests (19)
│   ├── test_metrics.py           # Metrics tests (22)
│   └── test_models.py            # Model tests (61)
├── artifacts/                    # Model artifacts
│   ├── xgboost/
│   ├── random_forest/
│   ├── mlp/
│   ├── logistic_regression/
│   ├── svm/
│   └── cnn1d/
├── results/                      # Benchmark results
│   ├── benchmark_summary.md
│   ├── benchmark_results.json
│   └── benchmark_*.png
├── notebooks/                    # Jupyter notebooks
│   └── colab_runner.ipynb        # Google Colab runner
├── balanced_features.csv         # Dataset (1000 samples)
└── README.md                     # This file
```

## Usage Guide

### 1. Data Loading

All models use the unified data loading pipeline:

```python
from src.data_loader import load_and_split_data

# Load and split data with fixed configuration
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    csv_path='balanced_features.csv',
    target_col='label',
    train_size=600,
    val_size=200,
    test_size=200,
    random_state=42,
    stratify=True
)
```

### 2. Preprocessing

Apply standardization to features:

```python
from src.preprocess import standardize_train_test_split

# Standardize features (fit on train, transform val/test)
X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_train_test_split(
    X_train, X_val, X_test,
    method='standard'
)
```

### 3. Model Training

Each model follows the same interface:

```python
from src.models.xgboost.train import train_xgboost

# Train model with configuration
results = train_xgboost(
    config='configs/base.yaml',
    output_dir='artifacts/xgboost'
)

# Access results
model = results['model']              # Trained model
metrics = results['metrics']          # Evaluation metrics
data_split = results['data_split']    # Split information
```

### 4. Evaluation

Models are automatically evaluated with comprehensive metrics:

```python
from src.metrics import evaluate_model

# Evaluate model and generate artifacts
result = evaluate_model(
    y_true=y_test,
    y_pred=predictions,
    y_proba=probabilities,
    model_name='XGBoost',
    output_dir='artifacts/xgboost'
)

# Generates:
# - confusion_matrix.png (visualization)
# - confusion_matrix.csv (data)
# - metrics.json (all metrics)
```

### 5. Configuration

Customize model behavior via YAML configuration:

```yaml
# configs/base.yaml
data:
  csv_path: "balanced_features.csv"
  target_col: "label"
  train_size: 600
  val_size: 200
  test_size: 200
  random_state: 42

models:
  xgboost:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100

  random_forest:
    n_estimators: 100
    max_depth: null
```

## Model Performance

Latest benchmark results (all models trained on same data split):

| Rank | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Time (s) |
|------|-------|----------|-----------|--------|----|---------|----------|
| 1 🥇 | **Random Forest** | 0.6950 | 0.6857 | 0.7200 | 0.7024 | **0.7468** | 0.67 |
| 2 🥈 | **XGBoost** | 0.6400 | 0.6321 | 0.6700 | 0.6505 | 0.6897 | 0.55 |
| 3 🥉 | **SVM** | 0.6350 | 0.6901 | 0.4900 | 0.5731 | 0.6844 | 0.32 |
| 4 | **Logistic Regression** | 0.6400 | 0.7000 | 0.4900 | 0.5765 | 0.6718 | **0.28** ⚡ |
| 5 | **MLP** | 0.5800 | 0.5769 | 0.6000 | 0.5882 | 0.6039 | 21.92 |
| 6 | **CNN1D** | 0.5550 | 0.5455 | 0.6600 | 0.5973 | 0.5998 | 5.68 |

**Key Insights:**
- **Best Overall Performance**: Random Forest (ROC-AUC: 0.7468)
- **Fastest Training**: Logistic Regression (0.28s)
- **Average ROC-AUC**: 0.6661 across all 6 models
- **Total Benchmark Time**: 29.42 seconds

**Recommendations:**
- **For accuracy-critical applications**: Use **Random Forest** (ROC-AUC: 0.7468)
- **For speed-critical applications**: Use **Logistic Regression** (Training time: 0.28s)
- **For GPU acceleration**: Use **CNN1D** with PyTorch (automatic GPU detection)

## Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Quick test (quiet mode)
pytest tests/ -q
```

### Test Coverage

- **119 total tests** (100% passing ✅)
  - `test_data_loader.py`: 17 tests
  - `test_preprocess.py`: 19 tests
  - `test_metrics.py`: 22 tests
  - `test_models.py`: 61 tests (XGBoost: 10, Random Forest: 10, MLP: 10, Logistic Regression: 10, SVM: 10, CNN1D: 11)

## Advanced Usage

### Batch Training

Train all models in sequence:

```bash
# Linux/Mac
bash scripts/run_all_local.sh

# Windows
scripts\run_all_local.bat

# Python (cross-platform)
python scripts/run_all_local.py
```

### Google Colab

Use the optimized Colab notebook for GPU acceleration:

```python
# Upload to Google Colab
# Open: notebooks/colab_runner.ipynb
# Runtime > Change runtime type > GPU (T4/A100)
```

### Custom Configuration

Create custom configuration files:

```python
# Train with custom config
results = train_xgboost(config='configs/custom.yaml')
```

## Artifacts

Each trained model generates comprehensive artifacts:

```
artifacts/{model_name}/
├── model.pkl                 # Trained model (pickle format)
├── confusion_matrix.png      # Confusion matrix visualization
├── confusion_matrix.csv      # Confusion matrix data
└── metrics.json              # All evaluation metrics
```

### Example: Loading Saved Model

```python
import pickle
from pathlib import Path

# Load trained model
with open('artifacts/random_forest/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test_scaled)
```

## Benchmarking

The unified benchmarking tool provides comprehensive model comparison:

```bash
python scripts/benchmark_all_models.py
```

**Generates:**

1. **benchmark_summary.md** - Detailed analysis report
   - Performance rankings
   - Best performers by metric
   - Detailed model comparison
   - Key findings and recommendations

2. **benchmark_results.json** - Structured metrics data
   ```json
   {
     "models": {
       "XGBoost": {
         "accuracy": 0.64,
         "roc_auc": 0.6897,
         "training_time": 2.11
       }
     }
   }
   ```

3. **Visualizations**
   - `benchmark_all_metrics.png` - 2x3 grid of bar charts
   - `benchmark_ranking_table.png` - Formatted comparison table
   - `benchmark_radar_chart.png` - Multi-dimensional radar chart

## Development

### Adding a New Model

1. Create model directory:
   ```bash
   mkdir -p src/models/new_model
   ```

2. Implement model wrapper (`src/models/new_model/model.py`):
   ```python
   class NewModelWrapper:
       def __init__(self, config=None):
           self.model = None
           self.config = config

       def train(self, X_train, y_train, X_val, y_val):
           # Training logic
           pass

       def predict(self, X):
           return self.model.predict(X)

       def predict_proba(self, X):
           return self.model.predict_proba(X)
   ```

3. Create training script (`src/models/new_model/train.py`):
   ```python
   from src.data_loader import load_and_split_data
   from src.preprocess import standardize_train_test_split
   from src.metrics import evaluate_model

   def train_new_model(config='configs/base.yaml', output_dir=None):
       # Load data
       X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(...)

       # Preprocess
       X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_train_test_split(...)

       # Train
       model = NewModelWrapper(config)
       model.train(X_train_scaled, y_train, X_val_scaled, y_val)

       # Evaluate
       metrics = evaluate_model(y_test, predictions, probabilities, 'NewModel', output_dir)

       return {'model': model, 'metrics': metrics, 'data_split': {...}}
   ```

4. Add tests (`tests/test_models.py`):
   ```python
   def test_new_model_training():
       results = train_new_model()
       assert 'model' in results
       assert 'metrics' in results
   ```

## Troubleshooting

### Common Issues

**1. FileNotFoundError: CSV file not found**
```bash
# Ensure dataset is in correct location
ls balanced_features.csv

# Or update config
# configs/base.yaml: data.csv_path
```

**2. Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**3. Tests failing**
```bash
# Run tests with verbose output
pytest tests/ -v --tb=short

# Check specific test file
pytest tests/test_models.py::test_xgboost_training -v
```

**4. Memory issues (MLP training)**
```bash
# Reduce batch size in configs/base.yaml
models:
  mlp:
    max_iter: 100  # Reduce from 500
```

## Performance Optimization

### Local Optimization

```yaml
# configs/local.yaml
models:
  xgboost:
    tree_method: 'hist'  # CPU optimization
    n_jobs: -1           # Use all CPU cores

  random_forest:
    n_jobs: -1           # Parallel training
```

### Colab Optimization (GPU)

```yaml
# configs/colab.yaml
models:
  xgboost:
    tree_method: 'gpu_hist'  # GPU acceleration
    gpu_id: 0
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NASA TESS Mission for providing exoplanet data
- scikit-learn for ML algorithms
- XGBoost team for gradient boosting implementation
- All contributors to this project

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{nasa_exoplanet_ml_2025,
  title={NASA Exoplanet ML Training Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/training-model}
}
```

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review benchmark results in `results/`

---

**Version**: 2.0.0
**Last Updated**: 2025-10-05
**Status**: Production Ready ✅
