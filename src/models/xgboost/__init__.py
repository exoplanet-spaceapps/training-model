"""
XGBoost Model Module

This module provides a unified interface for XGBoost model training and inference
integrated with the project's data pipeline.
"""

from src.models.xgboost.model import XGBoostWrapper
from src.models.xgboost.train import train_xgboost

__all__ = ['XGBoostWrapper', 'train_xgboost']
