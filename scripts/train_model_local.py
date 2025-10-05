#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train XGBoost model locally - Enhanced version with visualization"""

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
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings('ignore')

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

print("="*70)
print("Exoplanet Detection - XGBoost Training (Local)")
print("="*70)

# Check dependencies
print("\n[1/9] Checking dependencies...")
try:
    import xgboost as xgb
    print(f"  OK: XGBoost {xgb.__version__}")
except ImportError as e:
    print(f"  ERROR: XGBoost not installed: {e}")
    print("  Install: pip install xgboost")
    sys.exit(1)

# Setup paths
FEATURES_PATH = PROJECT_ROOT / 'data' / 'balanced_features.csv'
MODEL_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n[2/9] Paths configured")
print(f"  Features: {FEATURES_PATH}")
print(f"  Models: {MODEL_DIR}")
print(f"  Results: {RESULTS_DIR}")

# Load features
print(f"\n[3/9] Loading features...")
if not FEATURES_PATH.exists():
    print(f"  ERROR: Features file not found: {FEATURES_PATH}")
    print("  Run: python scripts/extract_balanced_features.py")
    sys.exit(1)

df = pd.read_csv(FEATURES_PATH)
print(f"  Total samples: {len(df)}")

# Filter successful extractions
df_success = df[df['status'] == 'success'].copy()
print(f"  Successful extractions: {len(df_success)} ({len(df_success)/len(df)*100:.1f}%)")

if len(df_success) < 100:
    print("  ERROR: Not enough successful samples (minimum 100)")
    sys.exit(1)

# Label distribution
label_counts = df_success['label'].value_counts()
print(f"\n  Label distribution:")
print(f"    True (label=1): {label_counts.get(1, 0)}")
print(f"    False (label=0): {label_counts.get(0, 0)}")

# Prepare features
print(f"\n[4/9] Preparing features...")
feature_columns = [
    'flux_mean', 'flux_std', 'flux_median', 'flux_mad',
    'flux_skew', 'flux_kurt',
    'bls_period', 'bls_duration', 'bls_depth', 'bls_power', 'bls_snr'
]

X = df_success[feature_columns].copy()
y = df_success['label'].copy()

# Handle NaN values
nan_counts = X.isnull().sum()
if nan_counts.sum() > 0:
    print("   NaN values detected, filling with median:")
    for col in feature_columns:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"    {col}: {nan_counts[col]} NaNs filled with {median_val:.4f}")

print(f"\n   Features shape: {X.shape}")
print(f"   Labels shape: {y.shape}")

# Train-test split
print(f"\n[5/9] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train set: {len(X_train)} samples")
print(f"  Test set: {len(X_test)} samples")
print(f"  Train labels: {dict(y_train.value_counts())}")
print(f"  Test labels: {dict(y_test.value_counts())}")

# XGBoost configuration (aligned with Colab notebook)
print(f"\n[6/9] Configuring XGBoost...")

# Detect GPU availability
try:
    if hasattr(xgb, 'device') and hasattr(xgb.device, 'is_cuda_available'):
        cuda_available = xgb.device.is_cuda_available()
    else:
        # Fallback check
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        cuda_available = result.returncode == 0
except:
    cuda_available = False

device = 'cuda' if cuda_available else 'cpu'

xgb_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'tree_method': 'hist',
}

# Add device parameter (XGBoost 2.1+)
try:
    xgb_params['device'] = device
    model = xgb.XGBClassifier(**xgb_params)
    print(f"   Device: {device}")
except TypeError:
    # Fallback for older XGBoost versions
    del xgb_params['device']
    if device == 'cuda':
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['gpu_id'] = 0
    model = xgb.XGBClassifier(**xgb_params)
    print(f"   Device: {device} (legacy mode)")

print(f"\n  Parameters:")
for key, val in xgb_params.items():
    print(f"    {key}: {val}")

# Train model
print(f"\n[7/9] Training XGBoost model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=10
)
print(f"\n   Training complete!")

# Predictions
print(f"\n[8/9] Evaluating model...")
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

# Display metrics
metrics_df = pd.DataFrame(metrics).T
print(f"\n   Model Performance:")
print(metrics_df.round(4).to_string())

print(f"\n   Test Set Performance:")
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

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  Feature Importance:")
for idx, row in feature_importance.iterrows():
    print(f"    {row['feature']:20s}: {row['importance']:.4f}")

# Visualization
print(f"\n[9/9] Creating visualizations...")

# Setup matplotlib for better display
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(18, 6))

# 1. Confusion Matrix
ax1 = plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Exoplanet', 'Exoplanet'],
            yticklabels=['No Exoplanet', 'Exoplanet'])
plt.title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 2. ROC Curve
ax2 = plt.subplot(1, 3, 2)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
roc_auc = roc_auc_score(y_test, y_pred_proba_test)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)

# 3. Feature Importance
ax3 = plt.subplot(1, 3, 3)
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance', fontsize=12)
plt.title('Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

plt.tight_layout()

# Save visualization
viz_path = RESULTS_DIR / 'training_visualization.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"   Visualization saved: {viz_path}")
plt.close()

# Save model
model_path = MODEL_DIR / 'xgboost_model.json'
model.save_model(model_path)
print(f"   Model saved: {model_path}")

# Save training report
report = {
    'timestamp': datetime.now().isoformat(),
    'environment': 'Local',
    'xgboost_version': xgb.__version__,
    'dataset': {
        'total_samples': len(df_success),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': feature_columns
    },
    'model': {
        'type': 'XGBClassifier',
        'parameters': xgb_params,
        'device': device
    },
    'metrics': {
        'train': {k: float(v) for k, v in metrics['Train'].items()},
        'test': {k: float(v) for k, v in metrics['Test'].items()}
    },
    'confusion_matrix': {
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1])
    },
    'feature_importance': feature_importance.to_dict('records')
}

report_path = RESULTS_DIR / 'training_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"   Report saved: {report_path}")

# Save feature names
features_meta_path = MODEL_DIR / 'feature_names.json'
with open(features_meta_path, 'w') as f:
    json.dump({'features': feature_columns}, f, indent=2)
print(f"   Feature metadata saved: {features_meta_path}")

print("="*70)

if metrics['Test']['ROC-AUC'] >= 0.80:
    print("SUCCESS! Model training successful! (ROC-AUC >= 0.80)")
    print("\nOutput files:")
    print(f"  - Model: {model_path}")
    print(f"  - Report: {report_path}")
    print(f"  - Visualization: {viz_path}")
    print("\nNext steps:")
    print("  1. Review visualization: open results/training_visualization.png")
    print("  2. Review report: cat results/training_report.json")
    print("  3. Test inference: python scripts/predict.py (if available)")
else:
    print(f"WARNING: Model performance below target (ROC-AUC = {metrics['Test']['ROC-AUC']:.4f})")
    print("\nSuggestions:")
    print("  1. Collect more training data")
    print("  2. Feature engineering (add more BLS parameters)")
    print("  3. Hyperparameter tuning")

print("="*70)
