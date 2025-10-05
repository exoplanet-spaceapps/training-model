"""
MLP Model Wrapper

This module provides a wrapper for Multi-Layer Perceptron (MLP) classifier that integrates
with the unified data pipeline and configuration system.
"""

import yaml
from pathlib import Path
from typing import Union, Dict, Any
import numpy as np


class MLPWrapper:
    """
    Wrapper for MLP classifier that loads configuration from YAML
    and provides a consistent interface for training and prediction.

    Args:
        config: Either a path to YAML config file or a config dictionary
    """

    def __init__(self, config: Union[str, Path, Dict[str, Any]] = 'configs/base.yaml'):
        """Initialize MLP wrapper with configuration."""
        # Load config if path provided
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        # Extract MLP parameters from config
        model_params = self.config.get('models', {}).get('mlp', {})

        # Get random_state from models section if not in mlp
        if 'random_state' not in model_params:
            model_params['random_state'] = self.config.get('models', {}).get('random_state', 42)

        # Import sklearn
        try:
            from sklearn.neural_network import MLPClassifier
            self._mlp = MLPClassifier
        except ImportError:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )

        # Create model with config parameters
        self.model = self._mlp(**model_params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> 'MLPWrapper':
        """
        Train the MLP model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, not used for sklearn MLP)
            y_val: Validation labels (optional, not used for sklearn MLP)

        Returns:
            self: Returns self for method chaining
        """
        # Train model
        self.model.fit(X_train, y_train)

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
            filepath: Path to save model (typically .pkl)
        """
        filepath = Path(filepath)

        # Use pickle for MLP
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filepath: Union[str, Path]) -> 'MLPWrapper':
        """
        Load model from file.

        Args:
            filepath: Path to load model from

        Returns:
            self: Returns self for method chaining
        """
        filepath = Path(filepath)

        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

        return self
