"""
Random Forest Model Module

This module provides a unified interface for Random Forest model training and inference
integrated with the project's data pipeline.
"""

from src.models.random_forest.model import RandomForestWrapper
from src.models.random_forest.train import train_random_forest

__all__ = ['RandomForestWrapper', 'train_random_forest']
