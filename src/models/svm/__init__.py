"""
SVM Model Module

This module provides a unified interface for SVM model training and inference
integrated with the project's data pipeline.
"""

from src.models.svm.model import SVMWrapper
from src.models.svm.train import train_svm

__all__ = ['SVMWrapper', 'train_svm']
