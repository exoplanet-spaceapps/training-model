# NASA Exoplanet Model Inventory & Refactoring Strategy

**Date:** 2025-10-05
**Status:** Phase 1 - Planning (TDD RED → GREEN phases completed for preprocessing & metrics)

---

## 📋 Executive Summary

This document provides a comprehensive inventory of all existing ML models in the NASA Exoplanet training project and outlines the refactoring strategy to achieve unified data pipeline integration.

**Refactoring Goals:**
1. ✅ Unified data source: `balanced_features.csv` (1000 samples)
2. ✅ Fixed split: train=600, val=200, test=200 (stratified, random_state=42)
3. ✅ Use `src/data_loader.py` for data loading
4. ✅ Use `src/preprocess.py` for preprocessing
5. ✅ Use `src/metrics.py` for evaluation
6. ✅ Every model outputs confusion matrix (PNG + CSV) to `artifacts/`
7. ✅ Load configuration from `configs/*.yaml`

---

## 🔍 Model Inventory

### **1. XGBoost (Gradient Boosting)**
**Status:** Multiple implementations found - NEEDS UNIFICATION

| File | Type | Notes |
|------|------|-------|
| `scripts/train_model_local.py` | Main training script | ✅ Best structure, uses balanced_features.csv |
| `scripts/train_xgboost_optuna.py` | Hyperparameter tuning | Uses Optuna for optimization |
| `ultraoptimized_cpu_models.py` | Class: `OptimizedXGBoostCPU` | CPU-optimized with MKL |
| `legacy/xgboost_koi.py` | Legacy version | ⚠️ Outdated data source |

**Features Used:** 11 features (flux_mean, flux_std, flux_median, flux_mad, flux_skew, flux_kurt, bls_period, bls_duration, bls_depth, bls_power, bls_snr)

**Current Split:** 80/20 train/test (NOT aligned with spec)

**Action Required:**
- [x] Identify best implementation (train_model_local.py)
- [ ] Refactor to use unified pipeline
- [ ] Implement 600/200/200 split
- [ ] Add confusion matrix output
- [ ] Load params from configs/base.yaml

---

### **2. Random Forest**
**Status:** Multiple implementations found - NEEDS UNIFICATION

| File | Type | Notes |
|------|------|-------|
| `ultraoptimized_cpu_models.py` | Class: `OptimizedRandomForest` | CPU-optimized (500 trees) |
| `legacy/train_rf_v1.py` | Legacy training script | ⚠️ Old data source |
| `final_model_comparison.py` | Comparison script | References TSFresh features |

**Current Configuration:**
- n_estimators=500
- max_depth=12
- n_jobs=physical_cores-1
- Features: Same 11 features as XGBoost

**Action Required:**
- [ ] Create `src/models/random_forest/` directory
- [ ] Implement unified training script
- [ ] Add confusion matrix output
- [ ] Load params from configs/base.yaml

---

### **3. MLP (Multi-Layer Perceptron)**
**Status:** Implementation found in ultraoptimized_cpu_models.py

| File | Type | Notes |
|------|------|-------|
| `ultraoptimized_cpu_models.py` | Class: `OptimizedMLPCPU` | LBFGS solver, 512-256-128-64 architecture |
| `legacy/koi_project_nn.py` | Legacy NN implementation | ⚠️ May use different architecture |

**Current Configuration:**
- Architecture: (512, 256, 128, 64)
- Solver: LBFGS (optimal for small datasets on CPU)
- Max iterations: 500
- Early stopping enabled

**Action Required:**
- [ ] Create `src/models/mlp/` directory
- [ ] Implement unified training script
- [ ] Add confusion matrix output
- [ ] Load params from configs/base.yaml

---

### **4. CNN1D (Convolutional Neural Network)**
**Status:** Multiple implementations found - GENESIS ensemble

| File | Type | Notes |
|------|------|-------|
| `cnn1d/cnn1d_trainer.py` | CNN1D trainer class | PyTorch/TensorFlow implementation |
| `scripts/train_genesis_cnn.py` | Genesis training script | 5-model ensemble |
| `Genesis/genesis_model.py` | Genesis model definition | Ensemble architecture |

**Genesis Ensemble Components:**
1. CNN1D model
2. XGBoost model
3. Random Forest model
4. Additional ensemble layers

**Data Type:** Raw light curves (NOT feature-based)

**Action Required:**
- [ ] Create `src/models/cnn1d/` directory
- [ ] Determine if Genesis should be separate model or ensemble
- [ ] Adapt for balanced_features.csv (currently uses light curves)
- [ ] Add confusion matrix output
- [ ] Load params from configs/colab.yaml (GPU-optimized)

---

### **5. SVM (Support Vector Machine) - TO BE LOCATED**
**Status:** NOT YET FOUND - Expected per project spec

**Expected Location:**
- Likely in `scripts/train_advanced_model.py`
- Or in `complete_gpcnn_benchmark.py`

**Action Required:**
- [ ] Search for SVM implementation
- [ ] If not found, create from scratch (TDD approach)
- [ ] Create `src/models/svm/` directory
- [ ] Implement training script

---

### **6. Logistic Regression - TO BE LOCATED**
**Status:** NOT YET FOUND - Expected per project spec

**Expected Location:**
- Likely in `scripts/train_advanced_model.py`

**Action Required:**
- [ ] Search for Logistic Regression implementation
- [ ] If not found, create from scratch (TDD approach)
- [ ] Create `src/models/logistic_regression/` directory
- [ ] Implement training script

---

## 📊 Current vs. Target Architecture

### **Current State:**
```
training-model/
├── scripts/
│   ├── train_model_local.py          # XGBoost
│   ├── train_xgboost_optuna.py      # XGBoost + Optuna
│   ├── train_genesis_cnn.py         # CNN ensemble
│   └── train_advanced_model.py      # Unknown models
├── ultraoptimized_cpu_models.py     # RF, XGBoost, MLP
├── cnn1d/cnn1d_trainer.py           # CNN1D
├── Genesis/genesis_model.py         # Genesis ensemble
└── legacy/                          # Old implementations
    ├── train_rf_v1.py
    ├── xgboost_koi.py
    └── koi_project_nn.py

ISSUES:
- ❌ Multiple scattered implementations per model
- ❌ Inconsistent data sources
- ❌ No unified split (600/200/200)
- ❌ No confusion matrix output
- ❌ No config-driven parameters
```

### **Target State:**
```
training-model/
├── src/
│   ├── data_loader.py               ✅ Implemented (19/19 tests pass)
│   ├── preprocess.py                ✅ Implemented (19/19 tests pass)
│   ├── metrics.py                   ✅ Implemented (22/22 tests pass)
│   └── models/
│       ├── xgboost/
│       │   ├── __init__.py
│       │   ├── model.py             # XGBClassifier wrapper
│       │   └── train.py             # Training script
│       ├── random_forest/
│       │   ├── __init__.py
│       │   ├── model.py             # RandomForestClassifier wrapper
│       │   └── train.py
│       ├── mlp/
│       │   ├── __init__.py
│       │   ├── model.py             # MLPClassifier wrapper
│       │   └── train.py
│       ├── svm/
│       │   ├── __init__.py
│       │   ├── model.py             # SVC wrapper
│       │   └── train.py
│       ├── logistic_regression/
│       │   ├── __init__.py
│       │   ├── model.py             # LogisticRegression wrapper
│       │   └── train.py
│       └── cnn1d/
│           ├── __init__.py
│           ├── model.py             # CNN architecture
│           └── train.py
├── configs/
│   ├── base.yaml                    ✅ Exists (shared settings)
│   ├── local.yaml                   ✅ Exists (CPU/GPU auto-detect)
│   └── colab.yaml                   ✅ Exists (T4/V100/A100 optimized)
├── artifacts/                       # Model outputs
│   ├── xgboost/
│   │   ├── model.pkl
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix.csv
│   │   └── metrics.json
│   ├── random_forest/
│   │   └── ... (same structure)
│   └── ... (for all 6 models)
├── scripts/
│   └── run_all_local.sh             # One-click execution
├── tests/
│   ├── test_data_loader.py          ✅ 19/19 pass
│   ├── test_preprocess.py           ✅ 19/19 pass
│   ├── test_metrics.py              ✅ 22/22 pass
│   └── test_models.py               ⏳ TO BE CREATED
└── results/
    └── benchmark_summary.md         ⏳ TO BE GENERATED

BENEFITS:
- ✅ Single data source (balanced_features.csv)
- ✅ Unified split (600/200/200, stratified)
- ✅ Config-driven parameters
- ✅ Consistent metrics output
- ✅ Easy comparison across models
- ✅ TDD coverage for all components
```

---

## 🎯 Refactoring Priority Order

### **Phase 1: Foundation (COMPLETED ✅)**
1. ✅ Create `src/data_loader.py` + tests (19/19 pass)
2. ✅ Create `src/preprocess.py` + tests (19/19 pass)
3. ✅ Create `src/metrics.py` + tests (22/22 pass)
4. ✅ Create `configs/base.yaml`, `configs/local.yaml`, `configs/colab.yaml`

### **Phase 2: Model Refactoring (IN PROGRESS ⏳)**
**Priority Order:**

1. **XGBoost** (FIRST - best existing implementation)
   - ✅ Existing code in `scripts/train_model_local.py`
   - ⏳ Create `src/models/xgboost/`
   - ⏳ Refactor to use unified pipeline
   - ⏳ Add tests in `tests/test_models.py`

2. **Random Forest** (SECOND - existing class in ultraoptimized_cpu_models.py)
   - ✅ Existing code in `ultraoptimized_cpu_models.py`
   - ⏳ Create `src/models/random_forest/`
   - ⏳ Refactor to use unified pipeline

3. **MLP** (THIRD - existing class in ultraoptimized_cpu_models.py)
   - ✅ Existing code in `ultraoptimized_cpu_models.py`
   - ⏳ Create `src/models/mlp/`
   - ⏳ Refactor to use unified pipeline

4. **Logistic Regression** (FOURTH - create from scratch if not found)
   - ⏳ Search in `scripts/train_advanced_model.py`
   - ⏳ Create `src/models/logistic_regression/`
   - ⏳ Implement with TDD

5. **SVM** (FIFTH - create from scratch if not found)
   - ⏳ Search in `scripts/train_advanced_model.py`
   - ⏳ Create `src/models/svm/`
   - ⏳ Implement with TDD

6. **CNN1D** (LAST - complex, GPU-dependent)
   - ✅ Existing code in `cnn1d/cnn1d_trainer.py`
   - ⏳ Create `src/models/cnn1d/`
   - ⏳ Adapt for feature-based input (currently uses light curves)
   - ⏳ Configure for Colab GPU execution

### **Phase 3: Integration & Testing (PENDING)**
1. ⏳ Create `scripts/run_all_local.sh`
2. ⏳ Create `notebooks/colab_runner.ipynb` (unified notebook)
3. ⏳ Run all 6 models + generate `results/benchmark_summary.md`
4. ⏳ Ensure all tests pass (target: 100+ tests total)

---

## 🔧 Refactoring Template (Per Model)

For each model, follow this TDD workflow:

### **1. Create Model Directory Structure**
```bash
mkdir -p src/models/{model_name}
touch src/models/{model_name}/__init__.py
touch src/models/{model_name}/model.py
touch src/models/{model_name}/train.py
```

### **2. Write Tests FIRST (TDD RED)**
```python
# tests/test_models.py

def test_{model_name}_loads_config():
    """Test model loads parameters from YAML config"""
    pass

def test_{model_name}_uses_unified_data_loader():
    """Test model uses load_and_split_data()"""
    pass

def test_{model_name}_uses_preprocessing():
    """Test model uses normalize_features()"""
    pass

def test_{model_name}_outputs_confusion_matrix():
    """Test model creates confusion_matrix.png and .csv"""
    pass

def test_{model_name}_outputs_metrics_json():
    """Test model creates metrics.json with required fields"""
    pass
```

### **3. Implement Model (TDD GREEN)**

**model.py:**
```python
from sklearn.ensemble import RandomForestClassifier  # Example
import yaml

class ModelWrapper:
    def __init__(self, config_path='configs/base.yaml'):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_params = config['models']['random_forest']
        self.model = RandomForestClassifier(**model_params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
```

**train.py:**
```python
from pathlib import Path
import yaml
from src.data_loader import load_and_split_data
from src.preprocess import standardize_train_test_split
from src.metrics import evaluate_model
from .model import ModelWrapper

def main():
    # 1. Load config
    with open('configs/base.yaml') as f:
        config = yaml.safe_load(f)

    # 2. Load data (unified pipeline)
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        csv_path=config['data']['csv_path'],
        target_col=config['data']['target_col'],
        train_size=config['data']['train_size'],
        val_size=config['data']['val_size'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=config['data']['stratify']
    )

    # 3. Preprocess (unified pipeline)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = \
        standardize_train_test_split(X_train, X_val, X_test, method='standard')

    # 4. Train model
    model = ModelWrapper()
    model.train(X_train_scaled, y_train)

    # 5. Evaluate (unified metrics)
    artifacts_dir = Path('artifacts/random_forest')
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_model(
        model=model.model,
        X_test=X_test_scaled,
        y_test=y_test,
        model_name='Random Forest',
        output_dir=artifacts_dir,
        save_confusion_matrix=True,
        save_metrics=True
    )

    print(f"\n✅ Random Forest training complete!")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   F1 Score: {results['f1']:.4f}")
    print(f"   Artifacts saved to: {artifacts_dir}")

if __name__ == '__main__':
    main()
```

### **4. Verify Tests Pass (TDD GREEN)**
```bash
pytest tests/test_models.py::test_random_forest_* -v
```

---

## 📝 Next Immediate Actions

1. **Search for SVM and Logistic Regression implementations** in:
   - `scripts/train_advanced_model.py`
   - `complete_gpcnn_benchmark.py`

2. **Start refactoring XGBoost** (highest priority):
   - Create `src/models/xgboost/` directory
   - Write tests in `tests/test_models.py`
   - Implement model using template above
   - Verify confusion matrix output

3. **Proceed systematically** through all 6 models using priority order

---

## 🎯 Success Criteria

**Per Model:**
- ✅ Uses `load_and_split_data()` from `src/data_loader.py`
- ✅ Uses `standardize_train_test_split()` from `src/preprocess.py`
- ✅ Uses `evaluate_model()` from `src/metrics.py`
- ✅ Loads parameters from `configs/base.yaml`
- ✅ Outputs to `artifacts/{model_name}/`:
  - `model.pkl` (or `.h5` for DL models)
  - `confusion_matrix.png`
  - `confusion_matrix.csv`
  - `metrics.json`
- ✅ Has passing tests in `tests/test_models.py`

**Overall Project:**
- ✅ All 6 models refactored and working
- ✅ `scripts/run_all_local.sh` executes all models
- ✅ `notebooks/colab_runner.ipynb` works on A100 GPU
- ✅ `results/benchmark_summary.md` generated with cross-model comparison
- ✅ All tests passing (target: 100+ total)

---

**Last Updated:** 2025-10-05
**Status:** Phase 1 Complete ✅ | Phase 2 In Progress ⏳
