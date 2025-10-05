"""CNN1D model for exoplanet detection.

This module provides a 1D Convolutional Neural Network for tabular feature data.
"""

from .model import CNN1DWrapper
from .train import train_cnn1d

__all__ = ['CNN1DWrapper', 'train_cnn1d']
