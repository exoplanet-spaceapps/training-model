"""
SVM Training Script

This script implements the complete SVM training pipeline using
the unified data loading, preprocessing, and evaluation components.
"""

import yaml
from pathlib import Path
from typing import Union, Dict, Any

from src.data_loader import load_and_split_data
from src.preprocess import standardize_train_test_split
from src.metrics import evaluate_model
from src.models.svm.model import SVMWrapper


def train_svm(
    config: Union[str, Path, Dict[str, Any]] = 'configs/base.yaml',
    output_dir: Union[str, Path] = None
) -> Dict[str, Any]:
    """
    Train SVM model using the unified pipeline.

    This function implements the complete training workflow:
    1. Load configuration
    2. Load and split data (600/200/200)
    3. Preprocess features (standardization)
    4. Train SVM model
    5. Generate predictions
    6. Evaluate on test set
    7. Save artifacts (model, confusion matrix, metrics)

    Args:
        config: Path to YAML config file or config dictionary
        output_dir: Directory to save model artifacts (overrides config)

    Returns:
        Dictionary containing:
            - model: Trained SVMWrapper instance
            - metrics: Dictionary of evaluation metrics
            - data_split: Information about data split sizes
    """
    # Load configuration
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = config

    # Get data configuration
    data_config = config_dict.get('data', {})
    csv_path = data_config.get('csv_path', 'data/balanced_features.csv')
    target_col = data_config.get('target_col', 'label')
    train_size = data_config.get('train_size', 600)
    val_size = data_config.get('val_size', 200)
    test_size = data_config.get('test_size', 200)
    random_state = data_config.get('random_state', 42)
    stratify = data_config.get('stratify', True)

    # Get output configuration
    output_config = config_dict.get('output', {})
    if output_dir is None:
        artifacts_dir = Path(output_config.get('artifacts_dir', 'artifacts'))
        output_dir = artifacts_dir / 'svm'
    else:
        output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and split data using unified pipeline
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        csv_path=csv_path,
        target_col=target_col,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # Step 2: Preprocess features using unified pipeline
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_train_test_split(
        X_train, X_val, X_test,
        method='standard'
    )

    # Step 3: Create and train model
    model = SVMWrapper(config=config_dict)
    model.train(X_train_scaled, y_train, X_val_scaled, y_val)

    # Step 4: Generate predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Step 5: Evaluate model using unified metrics
    metrics = evaluate_model(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        model_name='SVM',
        output_dir=output_dir
    )

    # Step 6: Save model
    if output_config.get('save_model', True):
        model_path = output_dir / 'model.pkl'
        model.save(model_path)

    # Return results
    results = {
        'model': model,
        'metrics': metrics,
        'data_split': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
    }

    return results


if __name__ == '__main__':
    """
    Example usage:
        python -m src.models.svm.train
    """
    import argparse

    parser = argparse.ArgumentParser(description='Train SVM model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for artifacts'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SVM Training - Unified Pipeline")
    print("=" * 70)

    results = train_svm(
        config=args.config,
        output_dir=args.output_dir
    )

    print("\nTraining Complete!")
    print(f"Metrics: {results['metrics']}")
    print(f"Data split: {results['data_split']}")
    print("=" * 70)
