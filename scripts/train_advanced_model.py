#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced XGBoost Model Training with Time Series + Wavelet Features
Uses 21 features (11 basic + 10 advanced) for improved performance
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

warnings.filterwarnings('ignore')

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

print("="*70)
print("Advanced XGBoost Model Training (21 Features)")
print("="*70)

# Check dependencies
print("\n[1/9] Checking dependencies...")
try:
    import xgboost as xgb
    print(f"  OK: XGBoost {xgb.__version__}")
except ImportError as e:
    print(f"  ERROR: {e}")
    print("  Install: pip install xgboost")
    sys.exit(1)

# Paths
FEATURES_PATH = PROJECT_ROOT / 'data' / 'advanced_features.csv'
MODEL_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load advanced features
print(f"\n[2/9] Loading advanced features...")
if not FEATURES_PATH.exists():
    print(f"  ERROR: Advanced features not found!")
    print(f"  Please run: python scripts/extract_advanced_features.py")
    sys.exit(1)

df = pd.read_csv(FEATURES_PATH)
df_success = df[df['status'] == 'success'].copy()
print(f"  Samples: {len(df_success)}")

# Feature columns (21 features total)
feature_columns = [
    # Basic statistics (6)
    'flux_mean', 'flux_std', 'flux_median', 'flux_mad',
    'flux_skew', 'flux_kurt',

    # BLS features (5)
    'bls_period', 'bls_duration', 'bls_depth', 'bls_power', 'bls_snr',

    # Time series features (4)
    'autocorr_lag1', 'autocorr_lag5', 'trend_slope', 'variability',

    # Frequency domain features (3)
    'fft_peak_freq', 'fft_peak_power', 'spectral_entropy',

    # Wavelet features (3)
    'wavelet_energy', 'wavelet_entropy', 'wavelet_var'
]

print(f"\n[3/9] Feature summary:")
print(f"  Total features: {len(feature_columns)}")
print(f"  - Basic: 6")
print(f"  - BLS: 5")
print(f"  - Time series: 4")
print(f"  - Frequency: 3")
print(f"  - Wavelet: 3")

# Prepare data
X = df_success[feature_columns].copy()
y = df_success['label'].copy()

# Handle NaN values
print(f"\n[4/9] Handling missing values...")
nan_counts = X.isnull().sum()
total_nan = nan_counts.sum()
if total_nan > 0:
    print(f"  Found NaN values in {(nan_counts > 0).sum()} features")
    for col in feature_columns:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"    {col}: filled {nan_counts[col]} NaN with median {median_val:.4f}")
else:
    print(f"  No missing values found")

# Split data
print(f"\n[5/9] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} (Positive: {y_train.sum()}, Negative: {len(y_train) - y_train.sum()})")
print(f"  Test:  {len(X_test)} (Positive: {y_test.sum()}, Negative: {len(y_test) - y_test.sum()})")

# Train XGBoost model with optimized hyperparameters
print(f"\n[6/9] Training Advanced XGBoost model...")

# Use best parameters from Optuna tuning
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'tree_method': 'hist',

    # Optimized parameters (from Optuna)
    'max_depth': 3,
    'learning_rate': 0.036,
    'n_estimators': 150,
    'subsample': 0.689,
    'colsample_bytree': 0.846,
    'min_child_weight': 2,
    'reg_alpha': 1.087,
    'reg_lambda': 0.020
}

# GPU support
try:
    if hasattr(xgb, 'device') and hasattr(xgb.device, 'is_cuda_available'):
        if xgb.device.is_cuda_available():
            params['device'] = 'cuda'
            print(f"  Using GPU acceleration")
except:
    pass

model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# Evaluate
print(f"\n[7/9] Evaluating model...")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_proba_train = model.predict_proba(X_train)[:, 1]
y_pred_proba_test = model.predict_proba(X_test)[:, 1]

# Metrics
metrics = {
    'Train': {
        'Accuracy': accuracy_score(y_train, y_pred_train),
        'Precision': precision_score(y_train, y_pred_train),
        'Recall': recall_score(y_train, y_pred_train),
        'F1': f1_score(y_train, y_pred_train),
        'ROC-AUC': roc_auc_score(y_train, y_pred_proba_train)
    },
    'Test': {
        'Accuracy': accuracy_score(y_test, y_pred_test),
        'Precision': precision_score(y_test, y_pred_test),
        'Recall': recall_score(y_test, y_pred_test),
        'F1': f1_score(y_test, y_pred_test),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba_test)
    }
}

print(f"\n  Train Set:")
print(f"    Accuracy:  {metrics['Train']['Accuracy']:.2%}")
print(f"    Precision: {metrics['Train']['Precision']:.2%}")
print(f"    Recall:    {metrics['Train']['Recall']:.2%}")
print(f"    F1:        {metrics['Train']['F1']:.2%}")
print(f"    ROC-AUC:   {metrics['Train']['ROC-AUC']:.2%}")

print(f"\n  Test Set:")
print(f"    Accuracy:  {metrics['Test']['Accuracy']:.2%}")
print(f"    Precision: {metrics['Test']['Precision']:.2%}")
print(f"    Recall:    {metrics['Test']['Recall']:.2%}")
print(f"    F1:        {metrics['Test']['F1']:.2%}")
print(f"    ROC-AUC:   {metrics['Test']['ROC-AUC']:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print(f"\n  Confusion Matrix:")
print(f"    TN: {cm[0, 0]:4d}  FP: {cm[0, 1]:4d}")
print(f"    FN: {cm[1, 0]:4d}  TP: {cm[1, 1]:4d}")

# Overfitting check
train_test_gap = metrics['Train']['ROC-AUC'] - metrics['Test']['ROC-AUC']
print(f"\n  Overfitting Analysis:")
print(f"    Train-Test Gap: {train_test_gap:.2%}")
if train_test_gap > 0.10:
    print(f"    WARNING: Significant overfitting detected!")
elif train_test_gap > 0.05:
    print(f"    Mild overfitting detected")
else:
    print(f"    OK: Good generalization")

# Feature Importance
print(f"\n  Top 10 Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"    {row['feature']:20s}: {row['importance']:.4f}")

# Save model
print(f"\n[8/9] Saving model...")
model_path = MODEL_DIR / 'xgboost_advanced.json'
model.save_model(model_path)
print(f"  Model: {model_path}")

# Save feature names
feature_names_path = MODEL_DIR / 'advanced_feature_names.json'
with open(feature_names_path, 'w') as f:
    json.dump(feature_columns, f, indent=2)
print(f"  Features: {feature_names_path}")

# Save report
report = {
    'timestamp': datetime.now().isoformat(),
    'model': 'XGBoost Advanced (21 features)',
    'features': {
        'total': len(feature_columns),
        'basic': 6,
        'bls': 5,
        'time_series': 4,
        'frequency': 3,
        'wavelet': 3
    },
    'hyperparameters': params,
    'train_metrics': {k: float(v) for k, v in metrics['Train'].items()},
    'test_metrics': {k: float(v) for k, v in metrics['Test'].items()},
    'confusion_matrix': {
        'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]), 'TP': int(cm[1, 1])
    },
    'overfitting_gap': float(train_test_gap),
    'feature_importance': feature_importance.to_dict('records')
}

report_path = RESULTS_DIR / 'advanced_model_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"  Report: {report_path}")

# Visualization
print(f"\n[9/9] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. ROC Curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)

axes[0, 0].plot(fpr_train, tpr_train, label=f"Train (AUC = {metrics['Train']['ROC-AUC']:.3f})", linewidth=2)
axes[0, 0].plot(fpr_test, tpr_test, label=f"Test (AUC = {metrics['Test']['ROC-AUC']:.3f})", linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.3)
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve - Advanced XGBoost (21 Features)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')

# 3. Feature Importance (Top 15)
top_features = feature_importance.head(15)
axes[1, 0].barh(range(len(top_features)), top_features['importance'].values)
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'].values)
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 15 Feature Importance')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(alpha=0.3, axis='x')

# 4. Model Comparison
comparison_data = {
    'Model': ['Baseline\n(11 features)', 'Optuna\n(11 features)', 'Advanced\n(21 features)'],
    'ROC-AUC': [0.7523, 0.7550, metrics['Test']['ROC-AUC']]
}
comparison_df = pd.DataFrame(comparison_data)

bars = axes[1, 1].bar(comparison_df['Model'], comparison_df['ROC-AUC'],
                       color=['#3498db', '#e74c3c', '#2ecc71'])
axes[1, 1].set_ylabel('ROC-AUC Score')
axes[1, 1].set_title('Model Performance Comparison')
axes[1, 1].set_ylim([0.7, max(comparison_df['ROC-AUC']) + 0.05])
axes[1, 1].axhline(y=0.80, color='red', linestyle='--', alpha=0.5, label='Target (0.80)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
viz_path = RESULTS_DIR / 'advanced_model_visualization.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"  Visualization: {viz_path}")

# Final comparison
print("="*70)
print("\nModel Performance Comparison:")
print(f"  XGBoost Baseline  (11 features): ROC-AUC = 75.23%")
print(f"  XGBoost Optuna    (11 features): ROC-AUC = 75.50%")
print(f"  XGBoost Advanced  (21 features): ROC-AUC = {metrics['Test']['ROC-AUC']:.2%}")

improvement_from_baseline = (metrics['Test']['ROC-AUC'] - 0.7523) * 100
improvement_from_optuna = (metrics['Test']['ROC-AUC'] - 0.7550) * 100

print(f"\nImprovement:")
print(f"  vs Baseline: {improvement_from_baseline:+.2f}%")
print(f"  vs Optuna:   {improvement_from_optuna:+.2f}%")

if metrics['Test']['ROC-AUC'] >= 0.80:
    print(f"\nSUCCESS! Advanced model meets target (ROC-AUC >= 0.80)")
else:
    gap_to_target = (0.80 - metrics['Test']['ROC-AUC']) * 100
    print(f"\nGap to target (0.80): {gap_to_target:.2f}%")
    print(f"Consider: Genesis CNN or ensemble methods")

print("="*70)
