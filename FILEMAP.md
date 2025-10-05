# FILEMAP.md - 完整檔案清單與功能摘要

Generated: 2025-10-05
Repository: NASA Training Model Project

## 📋 專案概覽

本專案為 NASA 系外行星偵測機器學習專案，包含多種ML模型實作（XGBoost、Random Forest、PyTorch CNN、TensorFlow Genesis CNN）以及相關的資料處理、去雜訊、效能優化工具。

**當前主要問題**：
- ❌ 資料來源不一致（多個CSV檔案與HDF5檔案）
- ❌ 訓練/測試切分方式不統一
- ❌ 部分模型未輸出confusion matrix
- ❌ 缺乏統一的TDD測試架構
- ❌ 本機與Colab優化配置未分離

---

## 📁 目錄結構總覽

```
training-model/
├── data/                          # 資料檔案
├── scripts/                       # 訓練腳本
├── legacy/                        # 舊版程式碼
├── Genesis/                       # TensorFlow Genesis CNN實作
├── cnn1d/                        # PyTorch CNN1D實作
├── notebooks/                     # Jupyter notebooks
├── artifacts/                     # 模型輸出（待建立）
├── configs/                       # 配置檔案（待建立）
├── src/                          # 源碼模組（待建立）
├── tests/                        # 測試檔案（待建立）
└── 工具腳本                       # 比較、跑分、去雜訊工具
```

---

## 📄 檔案詳細清單

### 🔧 專案配置與文件

| 檔案路徑 | 語言 | 功能 | 相依 | 潛在風險 |
|---------|------|------|------|---------|
| `.gitignore` | Config | Git忽略規則，包含Python、venv、資料檔案 | - | 已修改，可能包含未追蹤的重要檔案 |
| `CLAUDE.md` | Markdown | **專案規格書**（中文），定義完整改造需求 | - | 這是主要需求文件 |
| `PROMPT.md` | Markdown | **專案規格書**（中文），與CLAUDE.md內容相同 | - | 重複文件，應整合 |
| `README.md` | Markdown | 專案說明，包含環境設定、訓練指令、模型架構 | - | 需更新以反映統一架構 |

### 📊 資料檔案

| 檔案路徑 | 格式 | 資料內容 | 筆數 | 欄位數 | 用途 |
|---------|------|---------|------|--------|------|
| `balanced_features.csv` | CSV | **主要資料來源**，包含1000筆樣本，17個欄位（sample_id, tic_id, label, n_sectors, 6個flux統計量, 5個BLS特徵, status, error） | 1000 | 17 | 所有模型應使用的統一資料來源 |
| `advanced_features.csv` | CSV | 進階特徵資料（21個特徵：6 basic + 5 BLS + 4 time series + 3 frequency + 3 wavelet） | 未讀取 | 21 | train_advanced_model.py使用 |
| `tsfresh_features.csv` | CSV | TSFresh自動提取的時序特徵 | 未知 | 未知 | legacy模型使用 |
| `Dataset/KOI_features_csv/` | CSV | Kepler Objects of Interest特徵資料（多個CSV檔案） | 未知 | 未知 | legacy/xgboost_koi.py使用 |
| `data/lightcurves/` | HDF5 | 光變曲線原始資料 | 未知 | - | train_genesis_cnn.py使用 |

### 🤖 機器學習模型實作

#### XGBoost 模型

| 檔案路徑 | 技術棧 | 資料來源 | 特徵數 | Train/Val/Test切分 | 輸出 | ROC-AUC | 備註 |
|---------|--------|---------|--------|-------------------|------|---------|------|
| `scripts/train_model_local.py` | XGBoost + sklearn | `balanced_features.csv` | 11 (6 flux + 5 BLS) | 80/20 (stratified, random_state=42) | model + metrics | 75.23% | **Baseline模型**，已有confusion matrix |
| `scripts/train_xgboost_optuna.py` | XGBoost + Optuna | `balanced_features.csv` | 11 | 80/20 + 5-fold CV | model + metrics | 75.50% | 使用Optuna超參優化 |
| `scripts/train_advanced_model.py` | XGBoost + sklearn | `advanced_features.csv` | 21 | 80/20 (stratified, random_state=42) | model + confusion matrix + 多種視覺化 | 未文件化 | 使用進階特徵，完整視覺化輸出 |
| `legacy/xgboost_koi.py` | XGBoost + sklearn | `Dataset/KOI_features_csv/` | 未知 | 自定義索引切分 (非stratified) | confusion matrix (seaborn heatmap) | 未文件化 | **需改用balanced_features.csv** |

#### Random Forest 模型

| 檔案路徑 | 技術棧 | 資料來源 | 特徵數 | Train/Val/Test切分 | 輸出 | 備註 |
|---------|--------|---------|--------|-------------------|------|------|
| `legacy/train_rf_v1.py` | sklearn RandomForest | `tsfresh_features.csv` | 未知 | 90/10 (random_state=4, 非stratified) | confusion matrix + GridSearchCV結果 | **需改用balanced_features.csv** |

#### 神經網路模型 (PyTorch)

| 檔案路徑 | 架構 | 技術棧 | 資料來源 | 輸入形狀 | Train/Val/Test切分 | 輸出 | 備註 |
|---------|------|--------|---------|---------|-------------------|------|------|
| `cnn1d/cnn1d.py` | Two-Branch CNN1D (global + local) | PyTorch | HDF5 lightcurves | (2000,1) global + (512,1) local | 未知 | model定義 | 模型定義檔，需配合訓練腳本 |
| `scripts/train_genesis_cnn.py` | Genesis CNN (4 conv layers + 2 fc) | PyTorch | `data/lightcurves/*.h5` | (2000,1) | 未知 | confusion matrix + training curves | **需改用balanced_features.csv**，需解決HDF5→CSV轉換 |
| `legacy/koi_project_nn.py` | 3-layer MLP (256→64→1) | PyTorch | `tsfresh_features.csv` | 特徵維度 | 未知 | confusion matrix | Colab版本，含early stopping，**需改用balanced_features.csv** |
| `03b_cnn_train_mps.ipynb` | Two-Branch CNN1D | PyTorch | 合成資料 | (2000,1) + (512,1) | 自定義 | model + calibrator | MPS (Apple Silicon)支援，含校準 |

#### 神經網路模型 (TensorFlow)

| 檔案路徑 | 架構 | 技術棧 | 資料來源 | 輸入形狀 | Ensemble | 輸出 | 備註 |
|---------|------|--------|---------|---------|----------|------|------|
| `Genesis/genesis_model.py` | Genesis CNN (4 conv + 2 dense) | TensorFlow/Keras | 未知 | (2001,1) | 10個模型 | model定義 | arXiv:2105.06292實作 |
| `Genesis/train.py` | Genesis CNN訓練腳本 | TensorFlow | 未知 | (2001,1) | 10個模型 | 10個模型權重 | 完整訓練管線 |
| `Genesis/data_loader.py` | 資料載入器 | TensorFlow | FITS檔案 | - | - | Dataset | 需配合FITS資料 |

### 🔬 推論與評估

| 檔案路徑 | 支援模型 | 功能 | 輸出 |
|---------|---------|------|------|
| `04_newdata_inference.ipynb` | CNN (cnn1d.pt) / XGBoost (.pkl) / sklearn (.pkl) | 自動偵測模型類型，執行推論，輸出confusion matrix, ROC curve, PR curve | 完整評估報告 |
| `Genesis/predict.py` | Genesis CNN ensemble | Ensemble預測，投票機制 | 預測結果 |

### 🛠️ 工具與輔助腳本

#### 比較與跑分工具

| 檔案路徵 | 語言 | 功能 | 相依 | 可整合性 |
|---------|------|------|------|---------|
| `final_model_comparison.py` | Python (matplotlib, pandas) | **模型比較工具**：比較Genesis Ensemble vs XGBoost vs Random Forest，產生效能圖表、ROC-AUC比較、詳細表格 | matplotlib, pandas, seaborn | ✅ 可直接整合到`results/benchmark_summary.md`生成流程 |
| `complete_gpcnn_benchmark.py` | Python (PyTorch) | **GP+CNN跑分工具**：Gaussian Process + CNN架構，混合精度訓練，GPU優化 | torch, matplotlib | ✅ 可整合為benchmark模組 |

#### 去雜訊工具

| 檔案路徑 | 語言 | 功能 | 演算法 | 可整合性 |
|---------|------|------|--------|---------|
| `gp.py` | Python | **Gaussian Process去雜訊**：celerite2 GP detrending + Savitzky-Golay filter fallback | celerite2, scipy | ✅ 可整合到`src/preprocess.py` |
| `fold.py` | Python (NumPy) | **相位折疊與重採樣**：phase folding, robust normalization, equal resampling, 產生global/local views | numpy | ✅ 可整合到資料前處理管線 |

#### CPU/GPU優化工具

| 檔案路徑 | 語言 | 功能 | 優化技術 | 可整合性 |
|---------|------|------|---------|---------|
| `ultraoptimized_cpu_models.py` | Python (XGBoost, sklearn) | **CPU優化模型**：Intel MKL配置、OpenMP多執行緒、向量化、記憶體對齊、硬體偵測 | Intel MKL, OpenMP, AVX512 | ✅ 模式應用於`configs/local.yaml` |

#### 其他工具

| 檔案路徑 | 語言 | 功能 | 用途 |
|---------|------|------|------|
| `check_artifacts.py` | Python (pandas) | **輸出驗證工具**：檢查candidates_*.csv是否包含必要欄位、驗證provenance.yaml存在 | 確保輸出格式正確性 |
| `tls_runner.py` | Python | **Transit Least Squares**：執行TLS分析，返回period, duration, SDE等參數 | 特徵工程或驗證 |

### 📚 文件與指南

| 檔案路徑 | 內容 | 狀態 |
|---------|------|------|
| `scripts/ML_TRAINING_GUIDE.md` | ML訓練指南：環境設定、訓練步驟、模型說明、效能基準 | 需更新以反映統一架構 |
| `scripts/README.md` | Scripts資料夾說明 | 需更新 |
| `Genesis/README.md` | Genesis CNN專案說明 | 需更新 |
| `cnn1d/README.md` | CNN1D專案說明 | 需要建立 |

### 📓 Jupyter Notebooks

| 檔案路徑 | 內容 | 用途 | 平台 |
|---------|------|------|------|
| `03b_cnn_train_mps.ipynb` | Two-Branch CNN訓練，MPS支援 | 訓練與評估 | Apple Silicon Mac |
| `04_newdata_inference.ipynb` | 多模型推論，自動偵測 | 推論與評估 | 通用 |

---

## 🔍 機器學習模型總覽表

| # | 模型名稱 | 技術棧 | 檔案路徑 | 資料來源 | 特徵數 | Train/Test切分 | 輸出格式 | ROC-AUC | 問題 |
|---|---------|--------|---------|---------|--------|----------------|---------|---------|------|
| 1 | XGBoost Baseline | XGBoost + sklearn | `scripts/train_model_local.py` | `balanced_features.csv` | 11 | 80/20 (stratified, seed=42) | ✅ confusion matrix | 75.23% | - |
| 2 | XGBoost Optuna | XGBoost + Optuna | `scripts/train_xgboost_optuna.py` | `balanced_features.csv` | 11 | 80/20 + 5-fold CV | ✅ confusion matrix | 75.50% | - |
| 3 | XGBoost Advanced | XGBoost + sklearn | `scripts/train_advanced_model.py` | `advanced_features.csv` | 21 | 80/20 (stratified, seed=42) | ✅ confusion matrix + 多種視覺化 | ? | ❌ 非統一資料來源 |
| 4 | Genesis CNN (PyTorch) | PyTorch | `scripts/train_genesis_cnn.py` | `data/lightcurves/*.h5` | 2000 | ? | ✅ confusion matrix | ? | ❌ 使用HDF5非CSV |
| 5 | Genesis CNN (TF) | TensorFlow | `Genesis/train.py` | FITS檔案 | 2001 | ? | 10個模型權重 | ? | ❌ 使用FITS非CSV，❌ 無confusion matrix |
| 6 | Two-Branch CNN1D | PyTorch | `cnn1d/cnn1d.py` | 合成資料 | 2000+512 | ? | model定義 | ? | ❌ 缺乏訓練腳本 |
| 7 | MLP (legacy) | PyTorch | `legacy/koi_project_nn.py` | `tsfresh_features.csv` | ? | ? | ✅ confusion matrix | ? | ❌ 非統一資料來源 |
| 8 | Random Forest (legacy) | sklearn | `legacy/train_rf_v1.py` | `tsfresh_features.csv` | ? | 90/10 (seed=4) | ✅ confusion matrix | ? | ❌ 非stratified，❌ 非統一資料來源 |
| 9 | XGBoost KOI (legacy) | XGBoost | `legacy/xgboost_koi.py` | `Dataset/KOI_features_csv/` | ? | 自定義索引 | ✅ confusion matrix | ? | ❌ 非stratified，❌ 非統一資料來源 |

**圖例**：
- ✅ 已實作
- ❌ 需修正
- ? 未文件化

---

## 🧰 工具與輔助功能總覽

| 類別 | 工具名稱 | 檔案路徑 | 主要功能 | 整合方式建議 |
|------|---------|---------|---------|-------------|
| **比較** | Model Comparison | `final_model_comparison.py` | Genesis vs XGBoost vs RF效能比較 | 整合到`scripts/run_all_*.sh`產生`results/benchmark_summary.md` |
| **跑分** | GP+CNN Benchmark | `complete_gpcnn_benchmark.py` | GP+CNN混合精度跑分 | 整合到benchmark模組 |
| **去雜訊** | Gaussian Process | `gp.py` | celerite2 GP detrending | 整合到`src/preprocess.py` |
| **去雜訊** | Phase Folding | `fold.py` | 相位折疊與視圖生成 | 整合到`src/preprocess.py` |
| **CPU優化** | Ultra Optimized | `ultraoptimized_cpu_models.py` | Intel MKL + OpenMP配置 | 應用模式到`configs/local.yaml` |
| **驗證** | Artifacts Check | `check_artifacts.py` | 輸出格式驗證 | 整合到CI/pytest |
| **特徵** | TLS Runner | `tls_runner.py` | Transit Least Squares | 可選的特徵工程工具 |

---

## 📦 相依套件分析

### Python 核心套件
- `numpy`, `pandas`, `scipy`：資料處理與科學計算
- `matplotlib`, `seaborn`：視覺化
- `scikit-learn`：傳統機器學習與評估指標
- `joblib`：模型序列化

### 機器學習框架
- `xgboost`：梯度提升樹
- `torch` (PyTorch)：深度學習（CNN模型）
- `tensorflow` / `keras`：深度學習（Genesis CNN）
- `optuna`：超參數優化

### 專業套件
- `celerite2`：Gaussian Process快速計算
- `transitleastsquares`：Transit Least Squares演算法
- `tsfresh`：時序特徵自動提取（legacy使用）
- `h5py`：HDF5檔案讀取
- `astropy`：FITS檔案處理

---

## ⚠️ 發現的主要問題與風險

### 1️⃣ 資料來源不一致
- **問題**：9個模型使用5種不同資料來源
  - `balanced_features.csv` (3個模型) ✅
  - `advanced_features.csv` (1個模型) ❌
  - `tsfresh_features.csv` (2個模型) ❌
  - `Dataset/KOI_features_csv/` (1個模型) ❌
  - HDF5/FITS檔案 (2個模型) ❌
- **影響**：無法公平比較模型效能
- **風險等級**：🔴 高

### 2️⃣ Train/Test切分不統一
- **問題**：不同模型使用不同切分方式
  - 80/20 stratified seed=42 (2個模型) ✅
  - 80/20 stratified seed=42 + CV (1個模型) ✅
  - 90/10 seed=4 非stratified (1個模型) ❌
  - 自定義索引非stratified (1個模型) ❌
  - 未知切分方式 (4個模型) ❌
- **影響**：無法確保結果可重現，無法公平比較
- **風險等級**：🔴 高

### 3️⃣ 缺少統一的Validation Set
- **問題**：規格要求600/200/200切分，但現有模型多為80/20
- **影響**：需重新設計資料切分策略
- **風險等級**：🟡 中

### 4️⃣ Confusion Matrix輸出不一致
- **問題**：
  - ✅ 有輸出：7個模型
  - ❌ 無輸出：2個模型（Genesis TF, CNN1D定義檔）
  - ❓ 未知：部分模型未檢查
- **影響**：無法統一評估流程
- **風險等級**：🟡 中

### 5️⃣ 缺乏TDD測試架構
- **問題**：整個專案沒有`tests/`資料夾，無pytest測試
- **影響**：無法確保重構品質，無法自動化驗證
- **風險等級**：🔴 高

### 6️⃣ 本機與Colab配置混雜
- **問題**：
  - CPU優化在獨立檔案中（`ultraoptimized_cpu_models.py`）
  - GPU優化散佈在各訓練腳本中
  - MPS支援僅在notebook中
  - 無統一配置檔案
- **影響**：難以維護，無法快速切換平台
- **風險等級**：🟡 中

### 7️⃣ HDF5/FITS資料轉換問題
- **問題**：CNN模型需要光變曲線時序資料，但`balanced_features.csv`僅包含統計特徵
- **影響**：需要設計資料轉換策略或保留雙軌資料來源
- **風險等級**：🟡 中

### 8️⃣ 文件與實作不同步
- **問題**：
  - `ML_TRAINING_GUIDE.md`記錄的效能可能過時
  - README描述與實際檔案結構不符
  - 缺少部分模型的文件
- **影響**：增加理解與維護成本
- **風險等級**：🟢 低

---

## 🎯 改造優先級建議

### 🔴 Priority 1 (必須立即處理)
1. ✅ 建立`FILEMAP.md` (本檔案)
2. 📝 制定統一資料切分策略（600/200/200, stratified, seed=42）
3. 🧪 建立`tests/`資料夾與基礎測試架構
4. 📊 實作`src/data_loader.py`統一資料載入
5. 🔧 建立`configs/local.yaml`與`configs/colab.yaml`

### 🟡 Priority 2 (第二階段)
6. 🤖 逐一改造XGBoost/RF/MLP模型使用統一資料來源
7. 📈 確保所有模型輸出confusion matrix (PNG + CSV)
8. 📊 實作`src/metrics.py`統一評估指標
9. 🔗 整合比較/跑分工具到`scripts/run_all_*.sh`

### 🟢 Priority 3 (最後階段)
10. 🧠 解決CNN模型的資料轉換問題（HDF5→CSV或保留雙軌）
11. 📚 更新所有文件以反映新架構
12. 🚀 Colab環境配置優化（2025/10規格）
13. 📋 產生`results/benchmark_summary.md`

---

## 📝 下一步行動

根據規格書要求，接下來應：

1. ✅ **已完成**：輸出`FILEMAP.md`（本檔案）
2. 📋 **待執行**：提出改造計畫與PR拆分策略
3. 🧪 **待執行**：建立`tests/`最小測試集（預期紅燈）
4. 💻 **待執行**：落實`src/data_loader.py`、切分與`configs/`
5. 🔄 **待執行**：逐一改造模型...

---

**文件版本**：v1.0
**最後更新**：2025-10-05
**產生方式**：自動掃描 + 人工分析
