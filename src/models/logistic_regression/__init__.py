"""
Logistic Regression Model Module

This module provides a Logistic Regression implementation for exoplanet detection
using the unified data pipeline and configuration system.
"""

from src.models.logistic_regression.model import LogisticRegressionWrapper
from src.models.logistic_regression.train import train_logistic_regression

__all__ = [
    'LogisticRegressionWrapper',
    'train_logistic_regression',
]
