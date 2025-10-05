"""Training script for CNN1D model.

This script follows the unified training pipeline pattern.
"""

import os
import sys
import pickle
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import data_loader
from src import preprocess
from src import metrics
from src.models.cnn1d.model import CNN1DWrapper


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_cnn1d(
    config: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Train CNN1D model with unified pipeline.

    Args:
        config: Path to config file or dict with config values
        output_dir: Directory to save model artifacts

    Returns:
        Dictionary containing:
            - model: Trained model wrapper
            - metrics: Evaluation metrics dict
            - data_split: Information about data split
    """
    # Load configuration
    if config is None:
        config = 'configs/base.yaml'

    if isinstance(config, str):
        cfg = load_config(config)
    elif isinstance(config, Path):
        cfg = load_config(str(config))
    else:
        cfg = config

    # Set output directory
    if output_dir is None:
        artifacts_base = cfg.get('output', {}).get('artifacts_dir', 'artifacts')
        output_dir = os.path.join(artifacts_base, 'cnn1d')

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Training CNN1D Model")
    print("="*60)

    # 1. Load and split data
    print("\n[1/5] Loading and splitting data...")
    data_cfg = cfg['data']
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_split_data(
        csv_path=data_cfg['csv_path'],
        target_col=data_cfg['target_col'],
        train_size=data_cfg['train_size'],
        val_size=data_cfg['val_size'],
        test_size=data_cfg['test_size'],
        random_state=data_cfg['random_state'],
        stratify=data_cfg.get('stratify', True)
    )

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # 2. Preprocess features
    print("\n[2/5] Preprocessing features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess.standardize_train_test_split(
        X_train, X_val, X_test, method='standard'
    )

    # 3. Initialize and train model
    print("\n[3/5] Training CNN1D model...")

    # Get CNN1D specific config (with defaults)
    model_cfg = cfg.get('models', {}).get('cnn1d', {})
    random_state = cfg.get('models', {}).get('random_state', 42)

    # Prepare model configuration
    model_config = {
        'n_channels': model_cfg.get('n_channels', 32),
        'dropout': model_cfg.get('dropout', 0.3),
        'learning_rate': model_cfg.get('learning_rate', 0.001),
        'weight_decay': model_cfg.get('weight_decay', 0.0001),
        'batch_size': model_cfg.get('batch_size', 32),
        'max_epochs': model_cfg.get('max_epochs', 100),
        'patience': model_cfg.get('patience', 10),
        'device': model_cfg.get('device', 'auto'),
        'random_state': random_state
    }

    # Create and train model
    model = CNN1DWrapper(config=model_config)
    model.train(X_train_scaled, y_train, X_val_scaled, y_val)

    # 4. Make predictions
    print("\n[4/5] Making predictions...")
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)

    # 5. Evaluate model
    print("\n[5/5] Evaluating model...")
    eval_result = metrics.evaluate_model(
        y_true=y_test,
        y_pred=y_test_pred,
        y_proba=y_test_proba,
        model_name='CNN1D',
        output_dir=output_dir
    )

    # Extract metrics from evaluation result
    model_metrics = eval_result['metrics']

    print("\nTest Set Performance:")
    print(f"  Accuracy:  {model_metrics['accuracy']:.4f}")
    print(f"  Precision: {model_metrics['precision']:.4f}")
    print(f"  Recall:    {model_metrics['recall']:.4f}")
    print(f"  F1-Score:  {model_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {model_metrics['roc_auc']:.4f}")

    # Save model
    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")

    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print("\nArtifacts saved to:", output_dir)
    print("  - model.pkl (trained model)")
    print("  - scaler.pkl (feature scaler)")
    print("  - confusion_matrix.png")
    print("  - confusion_matrix.csv")
    print("  - metrics.json")

    return {
        'model': model,
        'metrics': eval_result,
        'data_split': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
    }


if __name__ == '__main__':
    # Train model with default config
    results = train_cnn1d()
