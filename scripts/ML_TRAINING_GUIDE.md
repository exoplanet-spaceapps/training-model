# 🚀 機器學習訓練完整指南

**Exoplanet Detection ML Pipeline - 完整訓練與模型比較指南**

本文檔提供完整的機器學習訓練流程，包括所有模型的訓練方法、效能比較，以及如何在另一台機器上重現結果。

---

## 📋 目錄

1. [專案概述](#專案概述)
2. [環境設置](#環境設置)
3. [數據準備](#數據準備)
4. [模型訓練流程](#模型訓練流程)
5. [模型比較與評估](#模型比較與評估)
6. [在新機器上執行](#在新機器上執行)

---

## 專案概述

### 目標
使用 TESS 光變曲線數據檢測系外行星（Exoplanet Detection）

### 數據集
- **來源**: TESS (Transiting Exoplanet Survey Satellite) from MAST Archive
- **總樣本**: 11,979 筆 (5,944 True + 6,035 False)
- **訓練集**: 1,000 筆平衡數據 (500 True + 500 False)
- **格式**: HDF5 lightcurve files

### 已實作的模型

| 模型 | 方法 | ROC-AUC | 檔案 |
|------|------|---------|------|
| **XGBoost Baseline** | 基礎 BLS 特徵 (11個) | 75.23% | `train_model_local.py` |
| **XGBoost Optuna** | 超參數調優 (50 trials) | 75.50% | `train_xgboost_optuna.py` |
| **Genesis CNN** | 論文方法 (arXiv:2105.06292) | TBD | `train_genesis_cnn.py` |
| **Advanced XGBoost** | 時序 + Wavelet 特徵 (21個) | TBD | `train_advanced_model.py` |

---

## 環境設置

### 必要套件

```bash
# 基礎套件
pip install numpy pandas scikit-learn matplotlib seaborn tqdm h5py

# 天文學套件
pip install lightkurve astropy

# 機器學習
pip install xgboost optuna

# 深度學習 (Genesis CNN)
pip install torch torchvision

# 進階特徵工程
pip install pywt scipy
```

### 快速安裝（一鍵）

```bash
pip install -r requirements.txt
```

### 系統需求

- **Python**: 3.10+
- **RAM**: 最少 8GB
- **硬碟空間**:
  - Lightcurves: ~2-3 GB
  - Models: ~100 MB
- **GPU**: 可選（CUDA 11.8+ for PyTorch）

---

## 數據準備

### 方法 1: 從 GitHub Release 下載（推薦）

```bash
# 下載預提取特徵
wget https://github.com/exoplanet-spaceapps/exoplanet-starter/releases/download/v1.0-features/balanced_features.csv -O data/balanced_features.csv
```

### 方法 2: 從頭開始

#### Step 1: 下載 Lightcurves

```bash
# 下載全部 11,979 筆數據 (需時 2-4 小時)
python scripts/run_test_fixed.py

# 或下載測試集 (100 筆，約 5-10 分鐘)
python scripts/run_test_fixed.py --test-only
```

#### Step 2: 提取特徵

**基礎特徵 (11 個 BLS 特徵):**
```bash
python scripts/extract_balanced_features.py
```

**進階特徵 (21 個特徵: BLS + 時序 + Wavelet):**
```bash
python scripts/extract_advanced_features.py
```

---

## 模型訓練流程

### 1️⃣ XGBoost Baseline (基準模型)

**特徵**: 11 個 BLS 特徵
- flux_mean, flux_std, flux_median, flux_mad, flux_skew, flux_kurt
- bls_period, bls_duration, bls_depth, bls_power, bls_snr

**訓練**:
```bash
python scripts/train_model_local.py
```

**輸出**:
- 模型: `models/xgboost_model.json`
- 報告: `results/training_report.json`
- 視覺化: `results/training_visualization.png`

**效能**:
- Accuracy: 67.00%
- Precision: 64.91%
- Recall: 74.00%
- F1: 69.16%
- **ROC-AUC: 75.23%**

**問題**: 嚴重過擬合 (Train: 99.99% vs Test: 75.23%)

---

### 2️⃣ XGBoost Optuna (超參數調優)

**目標**: 透過自動化超參數搜索改善效能

**訓練**:
```bash
python scripts/train_xgboost_optuna.py
```

**Optuna 配置**:
- 優化演算法: TPE (Tree-structured Parzen Estimator)
- Trials: 50
- 優化目標: 最大化 5-fold CV ROC-AUC

**最佳參數**:
```python
{
    'max_depth': 3,
    'learning_rate': 0.036,
    'n_estimators': 150,
    'subsample': 0.689,
    'colsample_bytree': 0.846,
    'min_child_weight': 2,
    'reg_alpha': 1.087,
    'reg_lambda': 0.020
}
```

**輸出**:
- 模型: `models/xgboost_optimized.json`
- 報告: `results/xgboost_optuna_report.json`
- 視覺化: `results/optuna_history.png`, `results/optuna_param_importance.png`

**效能**:
- **ROC-AUC: 75.50%**
- 改進: +0.27% (有限改善)

**結論**: 超參數調優對基礎特徵改善有限，需要更好的特徵工程

---

### 3️⃣ Genesis CNN (深度學習方法)

**論文**: [A one-armed CNN for exoplanet detection from lightcurves (arXiv:2105.06292)](https://arxiv.org/abs/2105.06292)

**方法**:
- 簡化的單臂 CNN
- 直接處理時序數據（無需手動特徵提取）
- 4 層卷積 + 3 層全連接
- 總參數: ~500K (遠少於 Astronet 的 6M+)

**架構**:
```
Input (1, 2000)
  → Conv1D(16) + ReLU + MaxPool + Dropout
  → Conv1D(32) + ReLU + MaxPool + Dropout
  → Conv1D(64) + ReLU + MaxPool + Dropout
  → Conv1D(128) + ReLU + MaxPool + Dropout
  → Flatten
  → FC(256) + ReLU + Dropout
  → FC(64) + ReLU + Dropout
  → FC(1) + Sigmoid
```

**訓練**:
```bash
# 需要 PyTorch
pip install torch torchvision

# 訓練 (30 epochs, ~10-20 分鐘 on GPU)
python scripts/train_genesis_cnn.py
```

**輸出**:
- 模型: `models/genesis_cnn_best.pth`, `models/genesis_cnn_full.pth`
- 報告: `results/genesis_cnn_report.json`
- 訓練曲線: `results/genesis_training_curves.png`

**優點**:
- 端到端學習，無需手動特徵工程
- 自動提取時序模式
- 論文證實與複雜模型效能接近

---

### 4️⃣ Advanced XGBoost (進階特徵工程)

**新增特徵** (額外 10 個):

1. **時序特徵** (4個):
   - `autocorr_lag1`: 自相關係數 (lag=1)
   - `autocorr_lag5`: 自相關係數 (lag=5)
   - `trend_slope`: 線性趨勢斜率
   - `variability`: 變異係數 (CV)

2. **頻率域特徵 (FFT)** (3個):
   - `fft_peak_freq`: FFT 峰值頻率
   - `fft_peak_power`: FFT 峰值功率
   - `spectral_entropy`: 頻譜熵

3. **小波特徵 (Wavelet)** (3個):
   - `wavelet_energy`: 小波能量
   - `wavelet_entropy`: 小波熵
   - `wavelet_var`: 小波細節方差

**提取特徵**:
```bash
python scripts/extract_advanced_features.py
```

**訓練**:
```bash
python scripts/train_advanced_model.py
```

**輸出**:
- 特徵: `data/advanced_features.csv`
- 模型: `models/xgboost_advanced.json`
- 報告: `results/advanced_model_report.json`

**預期改善**: 透過更豐富的特徵表示，預期 ROC-AUC 可達 80%+

---

## 模型比較與評估

### 自動比較腳本

```bash
# 比較所有已訓練模型
python scripts/compare_models.py
```

### 手動比較

#### 效能總表

| 模型 | Features | Parameters | Train Time | ROC-AUC | Accuracy | F1 | 改善 |
|------|----------|------------|------------|---------|----------|-------|------|
| XGBoost Baseline | 11 | ~100 trees | ~1 min | 75.23% | 67.00% | 69.16% | - |
| XGBoost Optuna | 11 | ~150 trees | ~5 min | 75.50% | 67.00% | 68.57% | +0.27% |
| Genesis CNN | Raw data | ~500K | ~15 min | TBD | TBD | TBD | TBD |
| Advanced XGBoost | 21 | ~100 trees | ~2 min | TBD | TBD | TBD | TBD |

#### 評估指標說明

- **ROC-AUC**: 主要指標，衡量分類能力（目標 ≥ 80%）
- **Accuracy**: 整體準確率
- **Precision**: 預測為 exoplanet 的準確度
- **Recall**: 實際 exoplanet 被正確識別的比率
- **F1**: Precision 和 Recall 的調和平均

#### 混淆矩陣分析

```
                Predicted
                No    Yes
Actual: No      TN    FP
        Yes     FN    TP
```

- **TN (True Negative)**: 正確預測為無行星
- **FP (False Positive)**: 誤報（假警報）
- **FN (False Negative)**: 漏報（錯過真實行星）
- **TP (True Positive)**: 正確預測為有行星

---

## 在新機器上執行

### 快速開始（從 GitHub Release）

```bash
# 1. Clone repository
git clone https://github.com/exoplanet-spaceapps/exoplanet-starter.git
cd exoplanet-starter

# 2. 安裝環境
pip install -r requirements.txt

# 3. 下載預提取特徵
mkdir -p data
wget https://github.com/exoplanet-spaceapps/exoplanet-starter/releases/download/v1.0-features/balanced_features.csv -O data/balanced_features.csv

# 4. 訓練所有模型
python scripts/train_model_local.py          # XGBoost Baseline
python scripts/train_xgboost_optuna.py       # Optuna 調優
python scripts/train_genesis_cnn.py          # Genesis CNN
python scripts/extract_advanced_features.py  # 進階特徵
python scripts/train_advanced_model.py       # Advanced XGBoost

# 5. 比較結果
python scripts/compare_models.py
```

### 完整流程（從原始數據）

```bash
# 1. 下載 Lightcurves (需時 2-4 小時)
python scripts/run_test_fixed.py

# 2. 提取特徵
python scripts/extract_balanced_features.py      # 基礎特徵
python scripts/extract_advanced_features.py      # 進階特徵

# 3. 訓練模型（同上）
python scripts/train_model_local.py
python scripts/train_xgboost_optuna.py
python scripts/train_genesis_cnn.py
python scripts/train_advanced_model.py

# 4. 評估與比較
python scripts/compare_models.py
```

---

## 檔案結構說明

```
exoplanet-starter/
├── data/
│   ├── supervised_dataset.csv           # 原始標籤數據
│   ├── balanced_features.csv            # 基礎特徵 (11個)
│   ├── advanced_features.csv            # 進階特徵 (21個)
│   └── lightcurves/                     # HDF5 光變曲線檔案
│       └── SAMPLE_000001_TIC123456.h5
│
├── models/
│   ├── xgboost_model.json               # Baseline 模型
│   ├── xgboost_optimized.json           # Optuna 優化模型
│   ├── genesis_cnn_best.pth             # Genesis CNN 模型
│   ├── xgboost_advanced.json            # Advanced 模型
│   └── feature_names.json               # 特徵定義
│
├── results/
│   ├── training_report.json             # 訓練報告
│   ├── training_visualization.png       # 視覺化
│   ├── xgboost_optuna_report.json       # Optuna 報告
│   ├── genesis_cnn_report.json          # CNN 報告
│   └── model_comparison.json            # 模型比較
│
├── scripts/
│   ├── run_test_fixed.py                # 下載 lightcurves
│   ├── extract_balanced_features.py     # 基礎特徵提取
│   ├── extract_advanced_features.py     # 進階特徵提取
│   ├── train_model_local.py             # XGBoost Baseline
│   ├── train_xgboost_optuna.py          # Optuna 調優
│   ├── train_genesis_cnn.py             # Genesis CNN
│   ├── train_advanced_model.py          # Advanced XGBoost
│   └── compare_models.py                # 模型比較
│
└── docs/
    ├── ML_TRAINING_GUIDE.md             # 本文檔
    └── RELEASE_GUIDE.md                 # Release 指南
```

---

## 疑難排解

### 常見問題

**Q1: ImportError: No module named 'lightkurve'**
```bash
pip install lightkurve astropy
```

**Q2: CUDA out of memory (Genesis CNN)**
```bash
# 減小 batch size
# 在 train_genesis_cnn.py 中修改:
BATCH_SIZE = 16  # 從 32 降到 16
```

**Q3: XGBoost 訓練過慢**
```bash
# 啟用 GPU (需要 CUDA)
# XGBoost 會自動偵測 GPU
```

**Q4: 特徵提取失敗**
```bash
# 檢查 HDF5 檔案是否存在
ls data/lightcurves/*.h5 | wc -l

# 重新下載特定樣本
python scripts/run_test_fixed.py --retry-failed
```

---

## 下一步優化方向

### 模型改進

1. **Ensemble Methods**
   - 結合 XGBoost + CNN 預測
   - Stacking / Voting

2. **Data Augmentation**
   - 時序資料增強（Jittering, Scaling, Rotation）
   - Mixup / Cutmix

3. **更大數據集**
   - 使用全部 11,979 筆數據訓練
   - 可能突破 85% ROC-AUC

### 部署

1. **API 服務** (FastAPI)
2. **Web UI** (Streamlit)
3. **Batch Inference** 腳本

---

## 參考資料

- [TESS Mission](https://tess.mit.edu/)
- [Lightkurve Documentation](https://docs.lightkurve.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Genesis CNN Paper (arXiv:2105.06292)](https://arxiv.org/abs/2105.06292)
- [Optuna: Hyperparameter Optimization](https://optuna.org/)

---

**最後更新**: 2025-10-05
**版本**: 1.0
**作者**: Exoplanet Detection Team
