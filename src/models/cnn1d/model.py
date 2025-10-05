"""CNN1D model implementation for tabular data.

This module provides a 1D Convolutional Neural Network wrapper that treats
tabular features as a sequence for feature learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and pooling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        return x


class CNN1DModel(nn.Module):
    """1D CNN for tabular feature classification.

    This model treats tabular features as a 1D sequence and applies
    convolutional layers to learn feature interactions.
    """

    def __init__(self, n_features: int = 13, n_channels: int = 32, dropout: float = 0.3):
        super().__init__()

        self.n_features = n_features

        # Input layer: reshape features to (batch, 1, n_features)
        # Conv blocks with decreasing sequence length
        self.conv1 = ConvBlock(1, n_channels, kernel_size=3, pool_size=1)
        self.conv2 = ConvBlock(n_channels, n_channels*2, kernel_size=3, pool_size=1)
        self.conv3 = ConvBlock(n_channels*2, n_channels*4, kernel_size=3, pool_size=1)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(n_channels*4, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, n_features)
        # Reshape to (batch, 1, n_features) for Conv1d
        x = x.unsqueeze(1)

        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Global average pooling
        x = self.gap(x)
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class CNN1DWrapper:
    """Wrapper for CNN1D model following unified pipeline pattern.

    This wrapper provides a consistent interface for training and prediction,
    compatible with the unified benchmarking system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CNN1D wrapper.

        Args:
            config: Configuration dictionary containing:
                - n_channels: Number of channels (default: 32)
                - dropout: Dropout rate (default: 0.3)
                - learning_rate: Learning rate (default: 0.001)
                - weight_decay: L2 regularization (default: 0.0001)
                - batch_size: Training batch size (default: 32)
                - max_epochs: Maximum training epochs (default: 100)
                - patience: Early stopping patience (default: 10)
                - device: Device for training ('auto', 'cuda', 'mps', 'cpu')
        """
        self.config = config or {}
        self.model = None
        self.device = self._get_device()
        self.n_features = None

    def _get_device(self):
        """Get the appropriate device for training.

        Priority: CUDA > MPS > CPU
        """
        device_str = self.config.get('device', 'auto')

        if device_str == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device_str)

    def _seed_everything(self, seed: int = 42):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self, X_train, y_train, X_val, y_val):
        """Train the CNN1D model.

        Args:
            X_train: Training features (numpy array or pandas DataFrame)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        # Set random seed
        random_state = self.config.get('random_state', 42)
        self._seed_everything(random_state)

        # Convert to numpy arrays if needed
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        if hasattr(X_val, 'values'):
            X_val = X_val.values
        if hasattr(y_val, 'values'):
            y_val = y_val.values

        # Store number of features
        self.n_features = X_train.shape[1]

        # Create model
        n_channels = self.config.get('n_channels', 32)
        dropout = self.config.get('dropout', 0.3)
        self.model = CNN1DModel(n_features=self.n_features, n_channels=n_channels, dropout=dropout)
        self.model.to(self.device)

        # Training parameters
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0001)
        batch_size = self.config.get('batch_size', 32)
        max_epochs = self.config.get('max_epochs', 100)
        patience = self.config.get('patience', 10)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training loop with early stopping
        best_val_loss = float('inf')
        wait = 0

        for epoch in range(max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze(1)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = self.model(X_batch).squeeze(1)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item()
                    n_val_batches += 1

            val_loss /= n_val_batches

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def predict(self, X):
        """Predict class labels.

        Args:
            X: Features to predict on

        Returns:
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities.

        Args:
            X: Features to predict on

        Returns:
            Predicted probabilities for the positive class
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values

        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).squeeze(1)
            proba = torch.sigmoid(outputs).cpu().numpy()

        return proba
