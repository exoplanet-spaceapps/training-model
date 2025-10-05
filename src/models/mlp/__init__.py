"""
MLP Model Module

This module provides a unified interface for MLP model training and inference
integrated with the project's data pipeline.
"""

from src.models.mlp.model import MLPWrapper
from src.models.mlp.train import train_mlp

__all__ = ['MLPWrapper', 'train_mlp']
