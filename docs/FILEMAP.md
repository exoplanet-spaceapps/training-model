# FILEMAP.md - NASA Exoplanet ML Training Project
# 完整檔案清單與功能摘要索引

**掃描日期**: 2025-01-05
**專案版本**: 2.0.0 (改造前)
**總筆數**: 1000 (balanced_features.csv)

---

## 📋 目錄

1. [資料檔案](#1-資料檔案)
2. [ML 模型與訓練腳本](#2-ml-模型與訓練腳本)
3. [工具腳本](#3-工具腳本)
4. [核心模組](#4-核心模組-src)
5. [配置檔案](#5-配置檔案-configs)
6. [測試檔案](#6-測試檔案-tests)
7. [文檔](#7-文檔)
8. [Notebooks](#8-notebooks)
9. [遺留代碼](#9-遺留代碼-legacy)
10. [專案管理](#10-專案管理)
11. [CI/CD](#11-cicd)

---

## 1. 資料檔案

### balanced_features.csv
- **路徑**: `./balanced_features.csv`
- **語言**: CSV
- **筆數**: 1000
- **功能**: 主資料集，包含平衡的特徵與標籤
- **欄位**:
  - `sample_id`: 樣本ID
  - `tic_id`: TIC目標ID
  - `label`: 目標標籤 (0/1 分類)
  - `n_sectors`: 觀測區段數
  - 特徵欄位: `flux_mean`, `flux_std`, `flux_median`, `flux_mad`, `flux_skew`, `flux_kurt`
  - BLS特徵: `bls_period`, `bls_duration`, `bls_depth`, `bls_power`, `bls_snr`
  - 狀態欄位: `status`, `error`
- **相依**: 所有模型的資料來源
- **風險**: ⚠️ **當前問題**: 不同模型使用不同資料切分方式，需統一

---

## 2. ML 模型與訓練腳本

### 2.1 Deep Learning 模型

#### Genesis Model (CNN Ensemble)
- **路徑**: `./Genesis/genesis_model.py`
- **語言**: Python (TensorFlow/Keras)
- **功能**: 實作論文 "A one-armed CNN for exoplanet detection" 的深度學習模型
- **架構**:
  - 5個CNN模型集成
  - 參數量: ~487K
  - 輸入: 光變曲線 (2001個時間點)
- **訓練**: `./scripts/train_genesis_cnn.py`
- **資料來源**: ⚠️ 使用原始光變曲線，**與主資料集不同**
- **輸出**:
  - 模型權重: 未統一路徑
  - 指標: JSON格式
  - 混淆矩陣: ❌ **缺少**
- **優化**: GPU (TF混合精度FP16, XLA JIT)
- **相依**: TensorFlow, Keras
- **風險**:
  - 資料來源與其他模型不一致
  - 缺少統一的輸出格式
  - 訓練時間長 (~16.7分鐘)

#### CNN1D Model
- **路徑**: `./cnn1d/cnn1d.py`, `./cnn1d/cnn1d_trainer.py`
- **語言**: Python (PyTorch)
- **功能**: 1D CNN用於光變曲線分類
- **架構**: Conv1D + BatchNorm + Pooling
- **訓練**: 內建 trainer
- **資料來源**: ⚠️ 自訂資料載入
- **輸出**:
  - 模型: `artifacts/cnn1d.pt`
  - 指標: `reports/metrics_cnn.json`
  - 混淆矩陣: ❌ **缺少**
- **優化**: PyTorch GPU
- **相依**: PyTorch
- **風險**: 資料載入不統一

#### GP+CNN Benchmark
- **路徑**: `./complete_gpcnn_benchmark.py`
- **語言**: Python (PyTorch)
- **功能**: GP去雜訊 + CNN pipeline
- **架構**:
  - Phase 1: GP simulator (Linear layers)
  - Phase 2: Multi-layer CNN
  - Phase 3: Classifier
- **訓練**: 一體化腳本
- **資料來源**: `tsfresh_features.csv` ⚠️ **不是主資料集**
- **輸出**: JSON指標
- **優化**: GPU (CUDA), AMP混合精度
- **相依**: PyTorch, TSFresh
- **風險**:
  - 使用錯誤的資料集
  - 資料切分不一致
  - 缺少混淆矩陣

### 2.2 傳統 ML 模型

#### XGBoost Models
- **路徑**:
  - `./scripts/train_xgboost_optuna.py` (Optuna優化版)
  - `./ultraoptimized_cpu_models.py` (CPU優化版)
  - `./final_model_comparison.py` (比較腳本)
- **語言**: Python (XGBoost)
- **功能**: 梯度提升樹分類
- **訓練配置**:
  - n_estimators: 300-500
  - max_depth: 6-8
  - tree_method: 'hist' (CPU) / 'gpu_hist' (GPU)
- **資料來源**: ⚠️ 混合使用 `tsfresh_features.csv` 和其他
- **輸出**: JSON指標
- **優化**:
  - CPU: Intel MKL, OpenMP, 多線程
  - GPU: CUDA histogram method
- **相依**: XGBoost, Optuna (可選)
- **風險**:
  - 多個版本存在
  - 資料來源不統一
  - 缺少混淆矩陣輸出

#### Random Forest
- **路徑**:
  - `./ultraoptimized_cpu_models.py` (CPU優化版)
  - `./final_model_comparison.py` (比較腳本)
  - `./legacy/train_rf_v1.py` (舊版)
- **語言**: Python (Scikit-learn)
- **功能**: 隨機森林分類器
- **訓練配置**:
  - n_estimators: 200-500
  - max_depth: 8-12
  - n_jobs: -1 (多線程)
- **資料來源**: ⚠️ 不統一
- **輸出**: JSON指標
- **優化**: Intel Extension for Scikit-learn, 多線程
- **相依**: Scikit-learn, scikit-learn-intelex (可選)
- **風險**: 多版本存在，資料不統一

#### MLP (Neural Network)
- **路徑**:
  - `./ultraoptimized_cpu_models.py`
  - `./legacy/koi_project_nn.py`
- **語言**: Python (Scikit-learn)
- **功能**: 多層感知機分類器
- **架構**: (512, 256, 128, 64)
- **Solver**: LBFGS (最適合CPU小資料集)
- **資料來源**: ⚠️ 不統一
- **優化**: Intel MKL
- **風險**: 性能較低，可能需要移除或改進

---

## 3. 工具腳本

### 3.1 比較與跑分工具

#### final_model_comparison.py
- **路徑**: `./final_model_comparison.py`
- **語言**: Python
- **功能**:
  - 比較 Genesis vs XGBoost vs Random Forest
  - 生成綜合比較報告（PDF + 圖表）
  - 輸出排名表
- **輸入**:
  - `reports/results/genesis_final_results.json`
  - `reports/results/quick_complete_comparison.json`
- **輸出**:
  - PDF: `reports/Final_Model_Comparison_Report.pdf`
  - CSV: `reports/results/final_comparison_table.csv`
  - JSON: `reports/results/final_comparison_summary.json`
  - 圖表: `reports/figures/final_*.png`
- **相依**: Matplotlib, Seaborn, Pandas
- **風險**: ⚠️ **需整合**: 輸入來源不統一，需改為統一格式

### 3.2 CPU/GPU 優化工具

#### ultraoptimized_cpu_models.py
- **路徑**: `./ultraoptimized_cpu_models.py`
- **語言**: Python
- **功能**:
  - CPU極致優化的訓練流程
  - 包含 RF, XGBoost, MLP
  - Intel MKL + OpenMP + 多線程
- **優化項目**:
  - 物理核心偵測
  - Intel MKL threading
  - OpenMP thread binding
  - CPU affinity設定
  - Memory-aligned arrays (64-byte)
  - Cache-optimized dtypes (float32)
- **輸出**: `ultraoptimized_cpu_results.json`
- **相依**: NumPy (MKL), psutil
- **風險**: ⚠️ **需整合**: 改為使用統一資料管線

#### complete_gpcnn_benchmark.py
- **路徑**: `./complete_gpcnn_benchmark.py`
- **語言**: Python
- **功能**: 完整benchmark，包含GP+CNN
- **優化**: GPU (CUDA benchmark, TF32)
- **風險**: 資料來源錯誤，需修正

### 3.3 其他工具

#### check_artifacts.py
- **路徑**: `./check_artifacts.py`
- **功能**: 檢查 artifacts 目錄輸出
- **風險**: ⚠️ **需更新**: 配合新的統一輸出格式

#### fold.py
- **路徑**: `./fold.py`
- **功能**: 未明確，可能是交叉驗證相關
- **風險**: ⚠️ **需檢視**: 是否保留或移除

#### gp.py
- **路徑**: `./gp.py`
- **功能**: Gaussian Process相關，可能用於去雜訊
- **風險**: ⚠️ **需整合**: 作為前處理選項

#### tls_runner.py
- **路徑**: `./tls_runner.py`
- **功能**: 未明確
- **風險**: ⚠️ **需檢視**

---

## 4. 核心模組 (src/)

### 4.1 已實作模組

#### src/data_loader.py ✅
- **狀態**: ✅ **已實作** (符合規格)
- **功能**:
  - 統一資料載入
  - 固定切分: 600/200/200
  - 分層抽樣 (stratified)
  - 資料完整性驗證
- **API**:
  - `load_and_split_data()` - 主要函數
  - `load_csv_data()` - CSV載入
  - `get_feature_columns()` - 自動偵測特徵欄
  - `validate_data_integrity()` - 資料驗證
- **測試**: ✅ `tests/test_data_loader.py` (已有測試)
- **相依**: Pandas, NumPy, Scikit-learn
- **優點**:
  - 符合規格要求
  - 良好的錯誤處理
  - 完整的文檔註釋

#### src/__init__.py
- **狀態**: ✅ 存在
- **功能**: Python package初始化
- **內容**: 空檔案

### 4.2 待實作模組

#### src/preprocess.py ❌
- **狀態**: ❌ **尚未實作**
- **功能 (規劃)**:
  - 統一資料前處理
  - 標準化/正規化
  - 缺失值處理
  - 特徵工程 (可選)
  - GP去雜訊 (可選)
- **需求**:
  - 與 data_loader 整合
  - 支援不同模型的需求
  - Pipeline化

#### src/metrics.py ❌
- **狀態**: ❌ **尚未實作**
- **功能 (規劃)**:
  - 統一指標計算
  - Confusion matrix (PNG + CSV)
  - Classification report
  - ROC-AUC, PR-AUC
  - 可視化工具
- **需求**:
  - 所有模型使用相同函數
  - 輸出格式統一到 `artifacts/<model>/`

#### src/models/ ❌
- **狀態**: ❌ **目錄不存在**
- **功能 (規劃)**:
  - 每個模型一個子目錄
  - 統一訓練接口
  - `src/models/xgboost/`
  - `src/models/random_forest/`
  - `src/models/genesis/`
  - `src/models/cnn1d/`
  - `src/models/gpcnn/`
- **需求**: 重構現有模型代碼

#### src/trainers/ ❌
- **狀態**: ❌ **目錄不存在**
- **功能 (規劃)**:
  - 統一Trainer類別
  - `<model_name>_trainer.py`
  - 標準化訓練流程
  - 自動保存 artifacts
- **需求**: 抽象化訓練邏輯

---

## 5. 配置檔案 (configs/)

### configs/base.yaml ✅
- **狀態**: ✅ **已實作**
- **功能**: 基礎配置，所有平台共用
- **內容**:
  - 專案資訊
  - 資料切分設定 (600/200/200)
  - 輸出格式設定
  - 模型訓練參數
  - 日誌配置
- **優點**: 結構清晰，符合規格

### configs/local.yaml ✅
- **狀態**: ✅ **已實作**
- **功能**: 本機執行配置
- **特色**:
  - 硬體自動偵測 (CPU/GPU)
  - Intel MKL優化
  - OpenMP設定
  - Apple Silicon MPS支援
  - 自適應batch size
  - 資源限制
- **優點**: 完善的本機優化設定

### configs/colab.yaml ✅
- **狀態**: ✅ **檔案存在** (需驗證內容)
- **功能**: Google Colab執行配置
- **需求**:
  - ⚠️ **需驗證**: 2025/10 Colab環境相容性
  - GPU配置 (T4/V100)
  - 套件版本鎖定
  - Drive掛載設定
  - 避免timeout

---

## 6. 測試檔案 (tests/)

### 6.1 已實作測試

#### tests/test_data_loader.py ✅
- **狀態**: ✅ **已實作**
- **功能**: 測試 data_loader.py
- **測試覆蓋**:
  - CSV載入
  - 資料切分
  - 分層抽樣
  - 資料驗證
  - 錯誤處理
- **測試框架**: pytest
- **覆蓋率**: 待確認

#### tests/conftest.py ✅
- **狀態**: ✅ **已實作**
- **功能**: Pytest fixtures和共用設定

#### tests/__init__.py ✅
- **狀態**: ✅ 存在

### 6.2 待實作測試

#### tests/test_preprocess.py ❌
- **狀態**: ❌ **尚未實作**
- **需求**: 測試前處理模組

#### tests/test_metrics.py ❌
- **狀態**: ❌ **尚未實作**
- **需求**: 測試指標計算

#### tests/test_models/ ❌
- **狀態**: ❌ **目錄不存在**
- **需求**: 每個模型的煙霧測試

#### tests/test_integration.py ❌
- **狀態**: ❌ **尚未實作**
- **需求**: 端到端整合測試

### pytest.ini ✅
- **狀態**: ✅ **已實作**
- **功能**: Pytest配置檔

---

## 7. 文檔

### PROMPT.md ✅
- **狀態**: ✅ **存在**
- **功能**: 專案改造需求文檔
- **內容**: 完整的任務規格與執行步驟

### CLAUDE.md ✅
- **狀態**: ✅ **存在**
- **功能**: Claude Code配置文檔

### FILEMAP.md 🚧
- **狀態**: 🚧 **本檔案**
- **功能**: 完整檔案清單與摘要

### README.md 📝
- **狀態**: 📝 **需更新**
- **功能**: 專案主要說明文檔
- **需求**:
  - 更新為新架構
  - 加入執行指引
  - 資料欄位說明
  - 如何加新模型

### docs/REFACTORING_PLAN.md ✅
- **狀態**: ✅ **存在**
- **功能**: 重構計畫文檔

### Genesis/GENESIS_MODEL_README.md
- **功能**: Genesis模型說明

### scripts/ML_TRAINING_GUIDE.md
- **功能**: ML訓練指南

---

## 8. Notebooks

### 03b_cnn_train_mps.ipynb
- **路徑**: `./03b_cnn_train_mps.ipynb`
- **功能**: CNN訓練 (Apple Silicon MPS)
- **風險**: ⚠️ **需整合**: 整合到統一流程

### 04_newdata_inference.ipynb
- **路徑**: `./04_newdata_inference.ipynb`
- **功能**: 新資料推論
- **風險**: ⚠️ **需整合**

### notebooks/colab_runner.ipynb ❌
- **狀態**: ❌ **尚未實作**
- **需求**:
  - Colab專用入口
  - 環境自動檢測
  - 套件安裝
  - Drive掛載
  - 一鍵執行所有模型

---

## 9. 遺留代碼 (legacy/)

### legacy/xgboost_koi.py
- **功能**: 舊版XGBoost訓練
- **狀態**: ⚠️ **待評估**: 保留或移除

### legacy/train_rf_v1.py
- **功能**: 舊版Random Forest訓練
- **狀態**: ⚠️ **待評估**: 保留或移除

### legacy/koi_project_nn.py
- **功能**: 舊版神經網路
- **狀態**: ⚠️ **待評估**: 保留或移除

---

## 10. 專案管理

### .gitignore
- **功能**: Git忽略檔案規則
- **需求**: ⚠️ **需更新**: 加入新的 artifacts/, results/ 等

### LICENSE
- **功能**: 專案授權

### requirements.txt ❌
- **狀態**: ❌ **尚未實作**
- **需求**:
  - 列出所有套件依賴
  - 固定版本號
  - 區分本機/Colab需求

### environment.yml ❌
- **狀態**: ❌ **尚未實作**
- **功能**: Conda環境配置 (可選)

---

## 11. CI/CD

### .github/workflows/ ❌
- **狀態**: ❌ **尚未實作**
- **需求**:
  - GitHub Actions CI
  - 自動執行 pytest
  - 代碼品質檢查

---

## 📊 統計摘要

### 檔案統計
- **總檔案數**: ~50+ (不含 `.claude/`, `htmlcov/`)
- **Python檔案**: ~23
- **Notebook檔案**: 2
- **配置檔案**: 3 (YAML)
- **文檔檔案**: ~7 (Markdown)
- **資料檔案**: 1 (CSV, 1000筆)

### 模型統計
- **Deep Learning**: 3 (Genesis, CNN1D, GP+CNN)
- **傳統ML**: 3 (XGBoost, Random Forest, MLP)
- **總計**: 6個模型

### 代碼健康度
- ✅ **已完成**:
  - 資料載入模組 (`src/data_loader.py`)
  - 基礎配置檔 (`configs/*.yaml`)
  - 基礎測試 (`tests/test_data_loader.py`)
- 🚧 **部分完成**:
  - 模型訓練腳本 (分散，不統一)
- ❌ **待實作**:
  - 前處理模組 (`src/preprocess.py`)
  - 指標模組 (`src/metrics.py`)
  - 統一訓練接口 (`src/trainers/`)
  - 模型重構 (`src/models/`)
  - 批次執行腳本 (`scripts/run_all_*.sh`)
  - Colab Notebook (`notebooks/colab_runner.ipynb`)
  - 完整測試套件
  - CI/CD pipeline

---

## ⚠️ 關鍵問題與風險

### 🔴 高風險問題 (必須解決)

1. **資料來源不一致** ❌
   - 不同模型使用不同資料集:
     - Genesis: 原始光變曲線
     - GP+CNN: `tsfresh_features.csv`
     - XGBoost/RF: 混合來源
   - **影響**: 無法公平比較模型
   - **解決**: 統一使用 `balanced_features.csv`

2. **資料切分不一致** ❌
   - 不同模型使用不同的 train/val/test 切分
   - **影響**: 結果無法重現和比較
   - **解決**: 統一使用 `src/data_loader.py`

3. **缺少 Confusion Matrix** ❌
   - 所有模型都缺少 confusion matrix 輸出
   - **影響**: 不符合規格要求
   - **解決**: 實作 `src/metrics.py` 並整合

4. **輸出格式不統一** ❌
   - Artifacts 存放位置混亂
   - **影響**: 難以整合和比較
   - **解決**: 統一到 `artifacts/<model_name>/`

### 🟡 中風險問題 (需要處理)

5. **模型代碼重複** ⚠️
   - XGBoost, Random Forest有多個版本
   - **影響**: 維護困難
   - **解決**: 重構到 `src/models/`

6. **缺少統一訓練接口** ⚠️
   - 每個模型訓練方式不同
   - **影響**: 難以批次執行
   - **解決**: 實作 `src/trainers/`

7. **測試覆蓋不足** ⚠️
   - 只有 data_loader 有測試
   - **影響**: 代碼品質無保障
   - **解決**: 完整測試套件

### 🟢 低風險問題 (建議改進)

8. **文檔不完整**
   - README需更新
   - 缺少 API 文檔

9. **缺少 CI/CD**
   - 沒有自動化測試

10. **Colab 支援不完整**
    - 缺少 Colab 專用 Notebook

---

## 🎯 改造優先順序

### Phase 1: 基礎建設 (Week 1)
1. ✅ `src/data_loader.py` (已完成)
2. ❌ `src/preprocess.py`
3. ❌ `src/metrics.py`
4. ❌ 完整測試套件

### Phase 2: 模型重構 (Week 2-3)
5. ❌ `src/models/<model_name>/`
6. ❌ `src/trainers/<model_name>_trainer.py`
7. ❌ 統一訓練接口

### Phase 3: 整合與自動化 (Week 4)
8. ❌ `scripts/run_all_local.sh`
9. ❌ `scripts/run_all_colab.sh`
10. ❌ `notebooks/colab_runner.ipynb`
11. ❌ 跨模型比較報告

### Phase 4: 驗證與文檔 (Week 5)
12. ❌ 完整測試通過
13. ❌ README 更新
14. ❌ CI/CD 設置

---

## 📝 備註

- 本文檔將隨改造進度持續更新
- 所有 ✅/❌/⚠️/🚧 標記表示實作狀態
- 建議每週review並更新此文檔

---

**文檔維護**: Claude Code
**最後更新**: 2025-01-05
