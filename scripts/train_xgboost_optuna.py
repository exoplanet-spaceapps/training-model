#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XGBoost Hyperparameter Tuning with Optuna"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

warnings.filterwarnings('ignore')

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

print("="*70)
print("XGBoost Hyperparameter Tuning with Optuna")
print("="*70)

# Check dependencies
print("\n[1/8] Checking dependencies...")
try:
    import xgboost as xgb
    import optuna
    print(f"  OK: XGBoost {xgb.__version__}")
    print(f"  OK: Optuna {optuna.__version__}")
except ImportError as e:
    print(f"  ERROR: {e}")
    print("  Install: pip install xgboost optuna")
    sys.exit(1)

# Paths
FEATURES_PATH = PROJECT_ROOT / 'data' / 'balanced_features.csv'
MODEL_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load and prepare data
print(f"\n[2/8] Loading features...")
df = pd.read_csv(FEATURES_PATH)
df_success = df[df['status'] == 'success'].copy()
print(f"  Samples: {len(df_success)}")

feature_columns = [
    'flux_mean', 'flux_std', 'flux_median', 'flux_mad',
    'flux_skew', 'flux_kurt',
    'bls_period', 'bls_duration', 'bls_depth', 'bls_power', 'bls_snr'
]

X = df_success[feature_columns].copy()
y = df_success['label'].copy()

# Handle NaN
nan_counts = X.isnull().sum()
if nan_counts.sum() > 0:
    for col in feature_columns:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)

# Split data
print(f"\n[3/8] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# Optuna objective function
print(f"\n[4/8] Setting up Optuna optimization...")

def objective(trial):
    """Optuna objective function for XGBoost hyperparameter tuning"""

    # Suggest hyperparameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'tree_method': 'hist',

        # Tunable parameters
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # GPU support
    try:
        if hasattr(xgb, 'device') and hasattr(xgb.device, 'is_cuda_available'):
            if xgb.device.is_cuda_available():
                params['device'] = 'cuda'
    except:
        pass

    # Train model
    model = xgb.XGBClassifier(**params)

    # Cross-validation to prevent overfitting
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='roc_auc', n_jobs=-1
    )

    return cv_scores.mean()

# Run Optuna study
print(f"\n[5/8] Running Optuna optimization (50 trials)...")
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n  Best ROC-AUC (CV): {study.best_value:.4f}")
print(f"  Best parameters:")
for key, val in study.best_params.items():
    print(f"    {key}: {val}")

# Train final model with best parameters
print(f"\n[6/8] Training final model with best parameters...")
best_params = study.best_params.copy()
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'tree_method': 'hist',
})

# GPU support
try:
    if hasattr(xgb, 'device') and hasattr(xgb.device, 'is_cuda_available'):
        if xgb.device.is_cuda_available():
            best_params['device'] = 'cuda'
except:
    pass

best_model = xgb.XGBClassifier(**best_params)
best_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# Evaluate
print(f"\n[7/8] Evaluating optimized model...")
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]

metrics = {
    'Train': {
        'Accuracy': accuracy_score(y_train, y_pred_train),
        'ROC-AUC': roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
    },
    'Test': {
        'Accuracy': accuracy_score(y_test, y_pred_test),
        'Precision': precision_score(y_test, y_pred_test),
        'Recall': recall_score(y_test, y_pred_test),
        'F1': f1_score(y_test, y_pred_test),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba_test)
    }
}

print(f"\n  Test Set Performance:")
print(f"    Accuracy:  {metrics['Test']['Accuracy']:.2%}")
print(f"    Precision: {metrics['Test']['Precision']:.2%}")
print(f"    Recall:    {metrics['Test']['Recall']:.2%}")
print(f"    F1:        {metrics['Test']['F1']:.2%}")
print(f"    ROC-AUC:   {metrics['Test']['ROC-AUC']:.2%}")

cm = confusion_matrix(y_test, y_pred_test)
print(f"\n  Confusion Matrix:")
print(f"    TN: {cm[0, 0]:4d}  FP: {cm[0, 1]:4d}")
print(f"    FN: {cm[1, 0]:4d}  TP: {cm[1, 1]:4d}")

# Comparison with baseline
baseline_roc = 0.7523  # From previous training
improvement = (metrics['Test']['ROC-AUC'] - baseline_roc) * 100
print(f"\n  Improvement over baseline:")
print(f"    Baseline ROC-AUC: {baseline_roc:.2%}")
print(f"    Optimized ROC-AUC: {metrics['Test']['ROC-AUC']:.2%}")
print(f"    Delta: {improvement:+.2f}%")

# Save results
print(f"\n[8/8] Saving results...")

# Save model
model_path = MODEL_DIR / 'xgboost_optimized.json'
best_model.save_model(model_path)
print(f"  Model: {model_path}")

# Save report
report = {
    'timestamp': datetime.now().isoformat(),
    'method': 'Optuna TPE Sampler',
    'n_trials': 50,
    'best_cv_roc_auc': float(study.best_value),
    'best_params': study.best_params,
    'test_metrics': {k: float(v) for k, v in metrics['Test'].items()},
    'train_metrics': {k: float(v) for k, v in metrics['Train'].items()},
    'confusion_matrix': {
        'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]), 'TP': int(cm[1, 1])
    },
    'improvement_over_baseline': float(improvement)
}

report_path = RESULTS_DIR / 'xgboost_optuna_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"  Report: {report_path}")

# Optuna visualization
print(f"\n  Creating Optuna visualizations...")

try:
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(RESULTS_DIR / 'optuna_history.png')

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(RESULTS_DIR / 'optuna_param_importance.png')

    print(f"    Visualizations: results/optuna_*.png")
except:
    print(f"    Skipping plotly visualizations (install: pip install kaleido)")

print("="*70)

if metrics['Test']['ROC-AUC'] >= 0.80:
    print("SUCCESS! Optimized model meets target (ROC-AUC >= 0.80)")
else:
    print(f"Model ROC-AUC: {metrics['Test']['ROC-AUC']:.2%}")

print("\nNext: Train Genesis CNN model (arXiv:2105.06292)")
print("  Run: python scripts/train_genesis_cnn.py")

print("="*70)
