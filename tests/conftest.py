"""
Pytest configuration and fixtures for test suite

This module provides shared fixtures and configuration for all tests.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add src/ to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Test constants
TRAIN_SIZE = 600
VAL_SIZE = 200
TEST_SIZE = 200
TOTAL_SIZE = 1000
RANDOM_STATE = 42


@pytest.fixture
def data_path():
    """Fixture providing path to test data CSV"""
    return PROJECT_ROOT / "data" / "balanced_features.csv"


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing"""
    np.random.seed(RANDOM_STATE)

    # Create synthetic data matching balanced_features.csv structure
    n_samples = TOTAL_SIZE
    data = {
        'sample_id': [f'SAMPLE_{i:06d}' for i in range(n_samples)],
        'tic_id': np.random.randint(1000000, 9999999, n_samples),
        'label': np.random.randint(0, 2, n_samples),  # Binary classification
        'n_sectors': np.random.randint(1, 20, n_samples),
        'flux_mean': np.random.normal(1.0, 0.01, n_samples),
        'flux_std': np.random.normal(0.01, 0.005, n_samples),
        'flux_median': np.random.normal(1.0, 0.01, n_samples),
        'flux_mad': np.random.normal(0.005, 0.002, n_samples),
        'flux_skew': np.random.normal(0.0, 0.5, n_samples),
        'flux_kurt': np.random.normal(0.0, 1.0, n_samples),
        'bls_period': np.random.uniform(1.0, 15.0, n_samples),
        'bls_duration': np.random.uniform(0.05, 0.3, n_samples),
        'bls_depth': np.random.uniform(0.001, 0.1, n_samples),
        'bls_power': np.random.uniform(0.0, 1.0, n_samples),
        'bls_snr': np.random.uniform(1.0, 20.0, n_samples),
        'status': ['success'] * n_samples,
        'error': [''] * n_samples,
    }

    return pd.DataFrame(data)


@pytest.fixture
def expected_split_sizes():
    """Fixture providing expected data split sizes"""
    return {
        'train': TRAIN_SIZE,
        'val': VAL_SIZE,
        'test': TEST_SIZE,
        'total': TOTAL_SIZE
    }


@pytest.fixture
def mock_csv_file(tmp_path, sample_data):
    """
    Create temporary CSV file for testing load_csv_data()

    This fixture enables tests that were previously skipping due to
    missing balanced_features.csv, fixing the coverage gap from 57% to ≥80%.

    Args:
        tmp_path: pytest built-in fixture providing temporary directory
        sample_data: Fixture providing synthetic 1000-sample DataFrame

    Returns:
        Path to temporary CSV file containing sample_data
    """
    csv_path = tmp_path / "balanced_features.csv"
    sample_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def artifacts_dir(tmp_path):
    """Fixture providing temporary artifacts directory for testing"""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return artifacts


@pytest.fixture
def config_base():
    """Fixture providing base configuration dictionary"""
    return {
        'data': {
            'csv_path': 'data/balanced_features.csv',
            'target_col': 'label',
            'train_size': TRAIN_SIZE,
            'val_size': VAL_SIZE,
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'stratify': True
        },
        'models': {
            'random_state': RANDOM_STATE,
            'xgboost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'tree_method': 'hist',
                'random_state': 42,
                'early_stopping_rounds': 10
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            'mlp': {
                'hidden_layer_sizes': [512, 256, 128, 64],
                'activation': 'relu',
                'solver': 'lbfgs',
                'alpha': 0.001,
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20
            },
            'logistic_regression': {
                'C': 1.0,
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': 42
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            },
            'cnn1d': {
                'n_channels': 32,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'max_epochs': 100,
                'patience': 10,
                'device': 'auto',
                'random_state': 42
            }
        },
        'output': {
            'artifacts_dir': 'artifacts',
            'save_confusion_matrix': True,
            'save_metrics': True,
            'save_model': True
        }
    }
