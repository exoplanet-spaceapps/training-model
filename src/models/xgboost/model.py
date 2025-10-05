"""
XGBoost Model Wrapper

This module provides a wrapper for XGBoost classifier that integrates with
the unified data pipeline and configuration system.
"""

import yaml
from pathlib import Path
from typing import Union, Dict, Any
import numpy as np


class XGBoostWrapper:
    """
    Wrapper for XGBoost classifier that loads configuration from YAML
    and provides a consistent interface for training and prediction.

    Args:
        config: Either a path to YAML config file or a config dictionary
    """

    def __init__(self, config: Union[str, Path, Dict[str, Any]] = 'configs/base.yaml'):
        """Initialize XGBoost wrapper with configuration."""
        # Load config if path provided
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        # Extract XGBoost parameters from config
        model_params = self.config.get('models', {}).get('xgboost', {})

        # Import XGBoost
        try:
            import xgboost as xgb
            self._xgb = xgb
        except ImportError:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )

        # Create model with config parameters
        self.model = self._xgb.XGBClassifier(**model_params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> 'XGBoostWrapper':
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            self: Returns self for method chaining
        """
        # Prepare eval_set if validation data provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        # Get training config
        training_config = self.config.get('training', {})
        verbose = training_config.get('verbose', 1)

        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X: Input features

        Returns:
            Predicted labels (0 or 1)
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        return self.model.predict_proba(X)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save model (supports .json, .pkl, .ubj)
        """
        filepath = Path(filepath)

        # Use XGBoost's native save for .json/.ubj, pickle for .pkl
        if filepath.suffix in ['.json', '.ubj']:
            self.model.save_model(str(filepath))
        else:
            # Use pickle for .pkl
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)

    def load(self, filepath: Union[str, Path]) -> 'XGBoostWrapper':
        """
        Load model from file.

        Args:
            filepath: Path to load model from

        Returns:
            self: Returns self for method chaining
        """
        filepath = Path(filepath)

        if filepath.suffix in ['.json', '.ubj']:
            self.model.load_model(str(filepath))
        else:
            import pickle
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)

        return self
