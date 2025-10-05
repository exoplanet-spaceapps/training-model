"""
Preprocessing Module for NASA Exoplanet Detection

This module provides feature preprocessing, normalization, and engineering utilities.
All preprocessing operations maintain consistency across train/val/test splits.

Key Features:
- Feature normalization (StandardScaler, MinMaxScaler)
- Missing value detection and handling
- Feature engineering (polynomial features)
- Feature statistics calculation

Author: NASA Exoplanet ML Team
Date: 2025-01-05
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer


def normalize_features(
    X: pd.DataFrame,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]:
    """
    Normalize features using specified scaling method.

    Args:
        X: Input features (DataFrame)
        method: Normalization method ('standard' or 'minmax')
            - 'standard': StandardScaler (mean=0, std=1)
            - 'minmax': MinMaxScaler (range [0, 1])

    Returns:
        Tuple of (normalized_features, fitted_scaler)
        - normalized_features: DataFrame with normalized values
        - fitted_scaler: Fitted scaler object for applying to new data

    Raises:
        ValueError: If DataFrame is empty or method is invalid

    Example:
        >>> X_train_normalized, scaler = normalize_features(X_train, method='standard')
        >>> X_test_normalized = apply_scaler(X_test, scaler)
    """
    if X.empty:
        raise ValueError("Cannot normalize empty DataFrame")

    if method not in ['standard', 'minmax']:
        raise ValueError(f"Invalid normalization method: '{method}'. "
                        "Must be 'standard' or 'minmax'")

    # Create and fit scaler
    scaler = create_feature_scaler(X, method=method)

    # Apply scaler
    X_normalized = apply_scaler(X, scaler)

    return X_normalized, scaler


def create_feature_scaler(
    X: pd.DataFrame,
    method: str = 'standard'
) -> Union[StandardScaler, MinMaxScaler]:
    """
    Create and fit a feature scaler.

    This function should be used on training data only, then the fitted
    scaler can be applied to validation/test data using apply_scaler().

    Args:
        X: Training features (DataFrame)
        method: Scaling method ('standard' or 'minmax')

    Returns:
        Fitted scaler object

    Raises:
        ValueError: If method is invalid

    Example:
        >>> scaler = create_feature_scaler(X_train, method='standard')
        >>> X_val_scaled = apply_scaler(X_val, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Invalid normalization method: '{method}'")

    # Fit scaler on training data
    scaler.fit(X)

    return scaler


def apply_scaler(
    X: pd.DataFrame,
    scaler: Union[StandardScaler, MinMaxScaler]
) -> pd.DataFrame:
    """
    Apply a fitted scaler to features.

    Args:
        X: Input features (DataFrame)
        scaler: Fitted scaler object from create_feature_scaler()

    Returns:
        DataFrame with scaled features (preserves column names)

    Example:
        >>> scaler = create_feature_scaler(X_train)
        >>> X_test_scaled = apply_scaler(X_test, scaler)
    """
    # Transform data
    X_scaled = scaler.transform(X)

    # Convert back to DataFrame with original column names
    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index
    )

    return X_scaled_df


def validate_no_missing_values(
    X: pd.DataFrame
) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has no missing values.

    Args:
        X: Input features (DataFrame)

    Returns:
        Tuple of (is_valid, missing_info)
        - is_valid: True if no missing values, False otherwise
        - missing_info: List of strings describing missing values per column

    Example:
        >>> is_valid, missing_info = validate_no_missing_values(X_train)
        >>> if not is_valid:
        ...     print(f"Missing values found: {missing_info}")
    """
    missing_info = []

    # Check for missing values per column
    missing_counts = X.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    if len(cols_with_missing) > 0:
        for col, count in cols_with_missing.items():
            missing_info.append(
                f"Column '{col}' has {count} missing values "
                f"({count/len(X)*100:.2f}%)"
            )

    is_valid = len(missing_info) == 0

    return is_valid, missing_info


def handle_missing_values(
    X: pd.DataFrame,
    strategy: str = 'mean'
) -> pd.DataFrame:
    """
    Handle missing values using specified strategy.

    Args:
        X: Input features (DataFrame) with missing values
        strategy: Imputation strategy
            - 'mean': Replace with column mean
            - 'median': Replace with column median
            - 'drop': Drop rows with missing values

    Returns:
        DataFrame with missing values handled

    Raises:
        ValueError: If strategy is invalid

    Example:
        >>> X_clean = handle_missing_values(X, strategy='mean')
    """
    if strategy not in ['mean', 'median', 'drop']:
        raise ValueError(f"Invalid strategy: '{strategy}'. "
                        "Must be 'mean', 'median', or 'drop'")

    if strategy == 'drop':
        # Drop rows with any missing values
        X_cleaned = X.dropna()
        return X_cleaned

    # Use sklearn's SimpleImputer for mean/median
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X)

    # Convert back to DataFrame with original column names
    X_imputed_df = pd.DataFrame(
        X_imputed,
        columns=X.columns,
        index=X.index
    )

    return X_imputed_df


def create_polynomial_features(
    X: pd.DataFrame,
    degree: int = 2,
    include_bias: bool = False,
    interaction_only: bool = False
) -> pd.DataFrame:
    """
    Create polynomial features for capturing non-linear relationships.

    Args:
        X: Input features (DataFrame)
        degree: Polynomial degree (default: 2)
            - degree=2 includes: x, x^2, x*y (for features x, y)
        include_bias: If True, include bias column (all ones)
        interaction_only: If True, only include interaction terms (no powers)

    Returns:
        DataFrame with polynomial features
        - Column names are auto-generated: 'x1 x2', 'x1^2', etc.

    Example:
        >>> X_poly = create_polynomial_features(X, degree=2)
        >>> # For 3 features, degree=2 produces 9 features
    """
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only
    )

    # Generate polynomial features
    X_poly = poly.fit_transform(X)

    # Get feature names
    feature_names = poly.get_feature_names_out(X.columns)

    # Convert to DataFrame
    X_poly_df = pd.DataFrame(
        X_poly,
        columns=feature_names,
        index=X.index
    )

    return X_poly_df


def get_feature_statistics(
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate comprehensive statistics for features.

    Args:
        X: Input features (DataFrame)

    Returns:
        DataFrame with statistics (columns are statistics, rows are features)
        - Includes: count, mean, std, min, 25%, 50%, 75%, max

    Example:
        >>> stats = get_feature_statistics(X_train)
        >>> print(stats.loc['flux_mean'])  # Stats for specific feature
    """
    # Use pandas describe() which provides comprehensive statistics
    stats = X.describe().T  # Transpose so features are rows

    return stats


def detect_outliers_iqr(
    X: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> pd.Series:
    """
    Detect outliers using Interquartile Range (IQR) method.

    Args:
        X: Input features (DataFrame)
        column: Column name to check for outliers
        multiplier: IQR multiplier (default: 1.5)
            - Values outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR] are outliers

    Returns:
        Boolean Series indicating outliers (True = outlier)

    Example:
        >>> outliers = detect_outliers_iqr(X, 'flux_mean')
        >>> X_clean = X[~outliers]  # Remove outliers
    """
    Q1 = X[column].quantile(0.25)
    Q3 = X[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = (X[column] < lower_bound) | (X[column] > upper_bound)

    return outliers


def clip_outliers(
    X: pd.DataFrame,
    column: str,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> pd.DataFrame:
    """
    Clip outliers to specified percentiles.

    Args:
        X: Input features (DataFrame)
        column: Column name to clip
        lower_percentile: Lower percentile threshold (default: 1.0)
        upper_percentile: Upper percentile threshold (default: 99.0)

    Returns:
        DataFrame with clipped values (copy, original unchanged)

    Example:
        >>> X_clipped = clip_outliers(X, 'flux_mean', lower_percentile=1, upper_percentile=99)
    """
    X_clipped = X.copy()

    lower_bound = X[column].quantile(lower_percentile / 100)
    upper_bound = X[column].quantile(upper_percentile / 100)

    X_clipped[column] = X_clipped[column].clip(lower=lower_bound, upper=upper_bound)

    return X_clipped


def create_interaction_features(
    X: pd.DataFrame,
    feature_pairs: List[Tuple[str, str]],
    operations: List[str] = ['multiply']
) -> pd.DataFrame:
    """
    Create interaction features from specified feature pairs.

    Args:
        X: Input features (DataFrame)
        feature_pairs: List of (feature1, feature2) tuples
        operations: List of operations to apply
            - 'multiply': f1 * f2
            - 'add': f1 + f2
            - 'subtract': f1 - f2
            - 'divide': f1 / f2 (with zero handling)

    Returns:
        DataFrame with original features + interaction features

    Example:
        >>> X_interact = create_interaction_features(
        ...     X,
        ...     [('flux_mean', 'flux_std')],
        ...     operations=['multiply', 'divide']
        ... )
    """
    X_new = X.copy()

    for feat1, feat2 in feature_pairs:
        for op in operations:
            if op == 'multiply':
                new_col = f'{feat1}_x_{feat2}'
                X_new[new_col] = X[feat1] * X[feat2]

            elif op == 'add':
                new_col = f'{feat1}_plus_{feat2}'
                X_new[new_col] = X[feat1] + X[feat2]

            elif op == 'subtract':
                new_col = f'{feat1}_minus_{feat2}'
                X_new[new_col] = X[feat1] - X[feat2]

            elif op == 'divide':
                new_col = f'{feat1}_div_{feat2}'
                # Handle division by zero
                X_new[new_col] = X[feat1] / (X[feat2] + 1e-10)

    return X_new


def log_transform_features(
    X: pd.DataFrame,
    columns: List[str],
    add_constant: float = 1.0
) -> pd.DataFrame:
    """
    Apply log transformation to specified columns.

    Useful for handling skewed distributions.

    Args:
        X: Input features (DataFrame)
        columns: List of column names to transform
        add_constant: Constant to add before log (handles zeros/negatives)
            - log(x + add_constant)

    Returns:
        DataFrame with transformed columns (copy, original unchanged)

    Example:
        >>> X_log = log_transform_features(X, ['flux_mean', 'flux_std'])
    """
    X_transformed = X.copy()

    for col in columns:
        X_transformed[col] = np.log(X[col] + add_constant)

    return X_transformed


def standardize_train_test_split(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """
    Standardize features across train/val/test splits consistently.

    Fits scaler on training data only, then applies to all splits.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        method: Scaling method ('standard' or 'minmax')

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)

    Example:
        >>> X_train_s, X_val_s, X_test_s, scaler = standardize_train_test_split(
        ...     X_train, X_val, X_test, method='standard'
        ... )
    """
    # Fit scaler on training data ONLY
    scaler = create_feature_scaler(X_train, method=method)

    # Apply scaler to all splits
    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)
    X_test_scaled = apply_scaler(X_test, scaler)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    """
    Example usage and verification
    """
    import sys
    from pathlib import Path

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_loader import load_and_split_data

    # Example: Load data and preprocess
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
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Example 1: Validate no missing values
    print("\n--- Validation ---")
    is_valid, missing_info = validate_no_missing_values(X_train)
    if is_valid:
        print("✓ No missing values detected")
    else:
        print(f"✗ Missing values found: {missing_info}")

    # Example 2: Get feature statistics
    print("\n--- Feature Statistics ---")
    stats = get_feature_statistics(X_train)
    print(stats.head())

    # Example 3: Normalize features
    print("\n--- Normalization ---")
    X_train_norm, scaler = normalize_features(X_train, method='standard')
    X_val_norm = apply_scaler(X_val, scaler)
    X_test_norm = apply_scaler(X_test, scaler)

    print(f"✓ Normalized train/val/test using StandardScaler")
    print(f"Train mean: {X_train_norm.mean().mean():.6f} (should be ≈0)")
    print(f"Train std: {X_train_norm.std().mean():.6f} (should be ≈1)")

    # Example 4: Polynomial features
    print("\n--- Polynomial Features ---")
    # Use only first 3 features for demonstration
    X_train_subset = X_train[['flux_mean', 'flux_std', 'flux_median']]
    X_poly = create_polynomial_features(X_train_subset, degree=2)
    print(f"Original features: {X_train_subset.shape[1]}")
    print(f"Polynomial features: {X_poly.shape[1]}")
    print(f"Feature names: {list(X_poly.columns)[:5]}...")

    print("\n✓ Preprocessing module verification complete!")
