"""
Data Loader Module for NASA Exoplanet Detection

This module provides unified data loading and splitting functionality for all ML models.
All models must use these functions to ensure consistent train/val/test splits.

Key Requirements:
- Fixed split: 600 train / 200 val / 200 test (from 1000 total samples)
- Stratified splitting to maintain class balance
- Reproducible with random_state=42
- No data leakage between splits

Author: NASA Exoplanet ML Team
Date: 2025-01-05
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split


# Constants from specification
RANDOM_STATE = 42
TOTAL_SIZE = 1000
TRAIN_SIZE = 600
VAL_SIZE = 200
TEST_SIZE = 200

# Default columns to exclude from features
DEFAULT_EXCLUDE_COLS = ['sample_id', 'tic_id', 'label', 'status', 'error']


def load_csv_data(csv_path: Path) -> pd.DataFrame:
    """
    Load data from CSV file.

    Args:
        csv_path: Path to CSV file containing balanced features

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If CSV file does not exist
        pd.errors.EmptyDataError: If CSV is empty

    Example:
        >>> df = load_csv_data(Path('data/balanced_features.csv'))
        >>> len(df)
        1000
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise pd.errors.EmptyDataError("CSV file is empty")

    return df


def get_feature_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None
) -> List[str]:
    """
    Automatically detect feature columns by excluding metadata columns.

    Args:
        df: Input DataFrame
        exclude_cols: List of column names to exclude from features
                     (default: ['sample_id', 'tic_id', 'label', 'status', 'error'])

    Returns:
        List of feature column names

    Example:
        >>> feature_cols = get_feature_columns(df)
        >>> 'label' in feature_cols
        False
        >>> 'flux_mean' in feature_cols
        True
    """
    if exclude_cols is None:
        exclude_cols = DEFAULT_EXCLUDE_COLS

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols


def validate_data_integrity(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate data integrity for feature columns.

    Checks:
    - No missing values in feature columns
    - All feature columns are numeric types

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names to validate

    Returns:
        Tuple of (is_valid: bool, errors: List[str])
        - is_valid: True if all checks pass, False otherwise
        - errors: List of error messages (empty if is_valid=True)

    Example:
        >>> is_valid, errors = validate_data_integrity(df, feature_cols)
        >>> if not is_valid:
        ...     print(f"Data integrity issues: {errors}")
    """
    errors = []

    # Check for missing values
    missing_counts = df[feature_cols].isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    if len(cols_with_missing) > 0:
        for col, count in cols_with_missing.items():
            errors.append(f"Column '{col}' has {count} missing values")

    # Check that all features are numeric
    for col in feature_cols:
        if col not in df.columns:
            errors.append(f"Column '{col}' not found in DataFrame")
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' is not numeric (dtype: {df[col].dtype})")

    is_valid = len(errors) == 0

    return is_valid, errors


def load_and_split_data(
    csv_path: Optional[Path] = None,
    df: Optional[pd.DataFrame] = None,
    target_col: str = 'label',
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load data and split into train/val/test sets with stratification.

    This is the main function that all models should use for data loading.
    Ensures consistent 600/200/200 split across all models.

    Args:
        csv_path: Path to CSV file (required if df is None)
        df: DataFrame to split (required if csv_path is None)
        target_col: Name of target column (default: 'label')
        train_size: Number of training samples (default: 600)
        val_size: Number of validation samples (default: 200)
        test_size: Number of test samples (default: 200)
        random_state: Random seed for reproducibility (default: 42)
        stratify: Whether to use stratified splitting (default: True)
        exclude_cols: Columns to exclude from features

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        - X_train: Training features (600 samples)
        - X_val: Validation features (200 samples)
        - X_test: Test features (200 samples)
        - y_train: Training labels (600 samples)
        - y_val: Validation labels (200 samples)
        - y_test: Test labels (200 samples)

    Raises:
        ValueError: If split sizes don't sum to total data size
        ValueError: If neither csv_path nor df is provided
        KeyError: If target_col not found in DataFrame

    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        ...     csv_path=Path('data/balanced_features.csv'),
        ...     target_col='label',
        ...     random_state=42
        ... )
        >>> len(X_train), len(X_val), len(X_test)
        (600, 200, 200)
    """
    # Load data if not provided
    if df is None:
        if csv_path is None:
            raise ValueError("Either csv_path or df must be provided")
        df = load_csv_data(csv_path)

    # Validate target column exists
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame. "
                      f"Available columns: {list(df.columns)}")

    # Validate split sizes
    total_split_size = train_size + val_size + test_size
    if total_split_size != len(df):
        raise ValueError(
            f"Split sizes ({train_size} + {val_size} + {test_size} = {total_split_size}) "
            f"do not match data size ({len(df)})"
        )

    # Get feature columns
    if exclude_cols is None:
        exclude_cols = DEFAULT_EXCLUDE_COLS

    feature_cols = get_feature_columns(df, exclude_cols)

    # Validate data integrity
    is_valid, errors = validate_data_integrity(df, feature_cols)
    if not is_valid:
        raise ValueError(f"Data integrity validation failed: {errors}")

    # Prepare features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # First split: separate test set (200 samples)
    # Remaining: 800 samples (600 train + 200 val)
    test_ratio = test_size / len(df)

    stratify_split = y if stratify else None

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify_split
    )

    # Second split: separate train (600) and val (200) from remaining 800
    # val_size / (train_size + val_size) = 200 / 800 = 0.25
    val_ratio = val_size / (train_size + val_size)

    stratify_split = y_temp if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_split
    )

    # Verify split sizes (sanity check)
    assert len(X_train) == train_size, f"Train size mismatch: {len(X_train)} != {train_size}"
    assert len(X_val) == val_size, f"Val size mismatch: {len(X_val)} != {val_size}"
    assert len(X_test) == test_size, f"Test size mismatch: {len(X_test)} != {test_size}"

    # Verify no data leakage (indices should not overlap)
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)

    assert len(train_indices & val_indices) == 0, "Data leakage: train/val overlap"
    assert len(train_indices & test_indices) == 0, "Data leakage: train/test overlap"
    assert len(val_indices & test_indices) == 0, "Data leakage: val/test overlap"

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    """
    Example usage and verification
    """
    import sys

    # Example: Load and split data
    csv_path = Path("data/balanced_features.csv")

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        print("Please ensure balanced_features.csv exists in data/ directory")
        sys.exit(1)

    print("Loading and splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        csv_path=csv_path,
        target_col='label',
        random_state=42,
        stratify=True
    )

    print(f"\nData split successful!")
    print(f"Train set: {len(X_train)} samples")
    print(f"Val set:   {len(X_val)} samples")
    print(f"Test set:  {len(X_test)} samples")
    print(f"\nFeature columns: {X_train.shape[1]}")
    print(f"Features: {list(X_train.columns)}")

    # Check class distribution
    print(f"\nClass distribution:")
    print(f"Train: {y_train.value_counts().to_dict()}")
    print(f"Val:   {y_val.value_counts().to_dict()}")
    print(f"Test:  {y_test.value_counts().to_dict()}")

    # Check stratification
    train_ratio = y_train.mean()
    val_ratio = y_val.mean()
    test_ratio = y_test.mean()

    print(f"\nClass balance (positive class ratio):")
    print(f"Train: {train_ratio:.4f}")
    print(f"Val:   {val_ratio:.4f}")
    print(f"Test:  {test_ratio:.4f}")
    print(f"Max difference: {max(abs(train_ratio - val_ratio), abs(train_ratio - test_ratio), abs(val_ratio - test_ratio)):.4f}")
