"""
Test suite for preprocess module

NOTE: These tests are written FIRST (TDD approach) before src/preprocess.py exists.
They are EXPECTED TO FAIL initially (red lights 🔴).
After implementing src/preprocess.py, tests should pass (green lights 🟢).

Test priorities:
- P0: Critical preprocessing functionality
- P1: Important feature engineering
- P2: Edge cases and advanced features
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# This import will FAIL until we create src/preprocess.py
# This is EXPECTED and CORRECT for TDD (Red → Green → Refactor)
try:
    from src.preprocess import (
        normalize_features,
        create_feature_scaler,
        apply_scaler,
        validate_no_missing_values,
        handle_missing_values,
        create_polynomial_features,
        get_feature_statistics
    )
    PREPROCESS_EXISTS = True
except ImportError:
    PREPROCESS_EXISTS = False
    # Create placeholder functions for test discovery
    def normalize_features(*args, **kwargs):
        raise NotImplementedError("preprocess.py not implemented yet (TDD red phase)")

    def create_feature_scaler(*args, **kwargs):
        raise NotImplementedError("preprocess.py not implemented yet (TDD red phase)")

    def apply_scaler(*args, **kwargs):
        raise NotImplementedError("preprocess.py not implemented yet (TDD red phase)")

    def validate_no_missing_values(*args, **kwargs):
        raise NotImplementedError("preprocess.py not implemented yet (TDD red phase)")

    def handle_missing_values(*args, **kwargs):
        raise NotImplementedError("preprocess.py not implemented yet (TDD red phase)")

    def create_polynomial_features(*args, **kwargs):
        raise NotImplementedError("preprocess.py not implemented yet (TDD red phase)")

    def get_feature_statistics(*args, **kwargs):
        raise NotImplementedError("preprocess.py not implemented yet (TDD red phase)")


# Mark all tests to skip if preprocess.py doesn't exist yet
pytestmark = pytest.mark.skipif(
    not PREPROCESS_EXISTS,
    reason="src/preprocess.py not implemented yet (TDD red phase - this is expected)"
)


class TestFeatureNormalization:
    """Test feature normalization functionality (P0 - Critical)"""

    @pytest.mark.p0
    def test_normalize_features_returns_correct_shape(self, sample_data):
        """
        Test that normalization preserves data shape
        This is a HARD REQUIREMENT for model compatibility
        """
        feature_cols = [col for col in sample_data.columns
                       if col not in ['sample_id', 'tic_id', 'label', 'status', 'error']]

        X_normalized, scaler = normalize_features(
            sample_data[feature_cols],
            method='standard'
        )

        # Shape must be preserved
        assert X_normalized.shape == sample_data[feature_cols].shape, \
            "Normalization must preserve data shape"

        # Must return scaler object
        assert scaler is not None, "Must return fitted scaler object"

    @pytest.mark.p0
    def test_normalize_features_standard_scaler(self, sample_data):
        """
        Test that StandardScaler produces mean≈0 and std≈1
        This is fundamental requirement for neural networks
        """
        feature_cols = [col for col in sample_data.columns
                       if col not in ['sample_id', 'tic_id', 'label', 'status', 'error']]

        X_normalized, scaler = normalize_features(
            sample_data[feature_cols],
            method='standard'
        )

        # Check mean ≈ 0 (within tolerance)
        means = X_normalized.mean(axis=0)
        assert np.allclose(means, 0, atol=1e-7), \
            f"StandardScaler mean should be ≈0, got {means.max()}"

        # Check std ≈ 1 (within tolerance)
        # Note: pandas .std() uses ddof=1, resulting in slight deviation from 1.0
        stds = X_normalized.std(axis=0)
        assert np.allclose(stds, 1, atol=0.01), \
            f"StandardScaler std should be ≈1, got {stds.max()}"

    @pytest.mark.p0
    def test_normalize_features_minmax_scaler(self, sample_data):
        """
        Test that MinMaxScaler produces values in [0, 1] range
        Important for some model architectures
        """
        feature_cols = [col for col in sample_data.columns
                       if col not in ['sample_id', 'tic_id', 'label', 'status', 'error']]

        X_normalized, scaler = normalize_features(
            sample_data[feature_cols],
            method='minmax'
        )

        # All values must be in [0, 1]
        assert X_normalized.min().min() >= 0, \
            "MinMaxScaler minimum should be ≥0"
        assert X_normalized.max().max() <= 1, \
            "MinMaxScaler maximum should be ≤1"

    @pytest.mark.p0
    def test_scaler_reproducibility(self, sample_data):
        """
        Test that scaler can be fitted once and applied to new data
        Critical for train/val/test pipeline consistency
        """
        feature_cols = [col for col in sample_data.columns
                       if col not in ['sample_id', 'tic_id', 'label', 'status', 'error']]

        # Fit scaler on first 800 samples (train + val)
        train_val_data = sample_data.iloc[:800][feature_cols]
        scaler = create_feature_scaler(train_val_data, method='standard')

        # Apply to test data (last 200 samples)
        test_data = sample_data.iloc[800:][feature_cols]
        X_test_normalized = apply_scaler(test_data, scaler)

        # Must preserve shape
        assert X_test_normalized.shape == test_data.shape, \
            "Applied scaler must preserve shape"

        # Apply scaler twice should give same result (idempotent)
        X_train_normalized_1 = apply_scaler(train_val_data, scaler)
        X_train_normalized_2 = apply_scaler(train_val_data, scaler)

        pd.testing.assert_frame_equal(
            X_train_normalized_1,
            X_train_normalized_2,
            check_exact=False,
            rtol=1e-10
        )

    @pytest.mark.p1
    def test_normalize_with_dataframe_returns_dataframe(self, sample_data):
        """
        Test that DataFrame input returns DataFrame output
        Maintains column names for interpretability
        """
        feature_cols = [col for col in sample_data.columns
                       if col not in ['sample_id', 'tic_id', 'label', 'status', 'error']]

        X_normalized, scaler = normalize_features(
            sample_data[feature_cols],
            method='standard'
        )

        # Must return DataFrame
        assert isinstance(X_normalized, pd.DataFrame), \
            "normalize_features should return DataFrame when given DataFrame"

        # Column names must be preserved
        assert list(X_normalized.columns) == feature_cols, \
            "Column names must be preserved after normalization"

    @pytest.mark.p2
    def test_normalize_with_invalid_method_raises_error(self, sample_data):
        """Test that invalid normalization method raises ValueError"""
        feature_cols = [col for col in sample_data.columns
                       if col not in ['sample_id', 'tic_id', 'label', 'status', 'error']]

        with pytest.raises(ValueError, match="Invalid normalization method"):
            normalize_features(sample_data[feature_cols], method='invalid_method')


class TestMissingValueHandling:
    """Test missing value detection and handling (P1 - Important)"""

    @pytest.mark.p1
    def test_validate_no_missing_values_clean_data(self, sample_data):
        """
        Test that validation passes for clean data
        """
        feature_cols = [col for col in sample_data.columns
                       if col not in ['sample_id', 'tic_id', 'label', 'status', 'error']]

        is_valid, missing_info = validate_no_missing_values(sample_data[feature_cols])

        assert is_valid is True, "Clean data should pass validation"
        assert len(missing_info) == 0, "Clean data should have no missing value info"

    @pytest.mark.p1
    def test_validate_detects_missing_values(self):
        """
        Test that validation detects missing values correctly
        """
        # Create data with missing values
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'feature2': [10.0, np.nan, 30.0, 40.0, 50.0],
            'feature3': [100.0, 200.0, 300.0, 400.0, 500.0]
        })

        is_valid, missing_info = validate_no_missing_values(df)

        assert is_valid is False, "Should detect missing values"
        assert len(missing_info) >= 2, "Should report both columns with missing values"
        assert 'feature1' in str(missing_info), "Should report feature1 has missing values"
        assert 'feature2' in str(missing_info), "Should report feature2 has missing values"

    @pytest.mark.p1
    def test_handle_missing_values_mean_imputation(self):
        """
        Test mean imputation for missing values
        """
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0],
            'feature2': [10.0, 20.0, 30.0, np.nan]
        })

        df_imputed = handle_missing_values(df, strategy='mean')

        # No missing values after imputation
        assert df_imputed.isnull().sum().sum() == 0, \
            "Should have no missing values after imputation"

        # Check mean imputation for feature1 (should be (1+2+4)/3 = 2.333...)
        expected_mean = (1.0 + 2.0 + 4.0) / 3
        assert abs(df_imputed.loc[2, 'feature1'] - expected_mean) < 1e-6, \
            "Mean imputation should use mean of non-missing values"

    @pytest.mark.p1
    def test_handle_missing_values_median_imputation(self):
        """
        Test median imputation for missing values
        """
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0, 100.0]  # Median = 3.0
        })

        df_imputed = handle_missing_values(df, strategy='median')

        # Check median imputation
        expected_median = 3.0  # median of [1, 2, 4, 100]
        assert df_imputed.loc[2, 'feature1'] == expected_median, \
            "Median imputation should use median of non-missing values"

    @pytest.mark.p2
    def test_handle_missing_values_drop_strategy(self):
        """
        Test dropping rows with missing values
        """
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0],
            'feature2': [10.0, 20.0, 30.0, 40.0]
        })

        df_cleaned = handle_missing_values(df, strategy='drop')

        # Should have 3 rows (1 dropped)
        assert len(df_cleaned) == 3, \
            "Should drop rows with missing values"

        # No missing values
        assert df_cleaned.isnull().sum().sum() == 0, \
            "Should have no missing values after dropping"


class TestFeatureEngineering:
    """Test feature engineering utilities (P1 - Important)"""

    @pytest.mark.p1
    def test_create_polynomial_features(self, sample_data):
        """
        Test polynomial feature creation
        Useful for capturing non-linear relationships
        """
        # Use only 3 features for testing (to keep polynomial expansion manageable)
        features = sample_data[['flux_mean', 'flux_std', 'flux_median']].head(100)

        poly_features = create_polynomial_features(
            features,
            degree=2,
            include_bias=False
        )

        # Should have more features than original
        # For 3 features with degree=2: 3 + 3 + 3 (interactions) = 9 features
        assert poly_features.shape[1] > features.shape[1], \
            "Polynomial features should increase feature count"

        # Row count should be preserved
        assert poly_features.shape[0] == features.shape[0], \
            "Polynomial features should preserve row count"

    @pytest.mark.p1
    def test_get_feature_statistics(self, sample_data):
        """
        Test feature statistics calculation
        Important for data understanding and debugging
        """
        feature_cols = [col for col in sample_data.columns
                       if col not in ['sample_id', 'tic_id', 'label', 'status', 'error']]

        stats = get_feature_statistics(sample_data[feature_cols])

        # Should return dictionary or DataFrame
        assert isinstance(stats, (dict, pd.DataFrame)), \
            "Statistics should be dict or DataFrame"

        # Should include basic statistics
        if isinstance(stats, pd.DataFrame):
            assert 'mean' in stats.index or 'mean' in stats.columns, \
                "Statistics should include mean"
            assert 'std' in stats.index or 'std' in stats.columns, \
                "Statistics should include std"
            assert 'min' in stats.index or 'min' in stats.columns, \
                "Statistics should include min"
            assert 'max' in stats.index or 'max' in stats.columns, \
                "Statistics should include max"

    @pytest.mark.p2
    def test_feature_statistics_handles_edge_cases(self):
        """
        Test that feature statistics handles edge cases properly
        """
        # Single value column
        df = pd.DataFrame({
            'constant': [5.0, 5.0, 5.0, 5.0],
            'normal': [1.0, 2.0, 3.0, 4.0]
        })

        stats = get_feature_statistics(df)

        # Should not raise error on constant column
        assert stats is not None, "Should handle constant columns"


class TestEdgeCases:
    """Test edge cases and error handling (P2 - Nice to have)"""

    @pytest.mark.p2
    def test_normalize_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises appropriate error"""
        df_empty = pd.DataFrame()

        with pytest.raises(ValueError, match="empty|Empty"):
            normalize_features(df_empty, method='standard')

    @pytest.mark.p2
    def test_normalize_single_row_handles_gracefully(self):
        """Test that single row is handled gracefully"""
        df_single = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0]
        })

        # Should not raise error
        X_normalized, scaler = normalize_features(df_single, method='standard')

        # Shape should be preserved
        assert X_normalized.shape == (1, 2), \
            "Single row should be handled gracefully"

    @pytest.mark.p2
    def test_normalize_single_column_handles_gracefully(self):
        """Test that single column is handled gracefully"""
        df_single_col = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0]
        })

        # Should not raise error
        X_normalized, scaler = normalize_features(df_single_col, method='standard')

        # Shape should be preserved
        assert X_normalized.shape == (4, 1), \
            "Single column should be handled gracefully"

    @pytest.mark.p2
    def test_normalize_constant_feature_handles_gracefully(self):
        """
        Test that constant feature (zero variance) is handled gracefully
        StandardScaler will produce NaN for zero-variance features
        """
        df = pd.DataFrame({
            'constant': [5.0, 5.0, 5.0, 5.0],
            'variable': [1.0, 2.0, 3.0, 4.0]
        })

        # Should handle constant feature
        X_normalized, scaler = normalize_features(df, method='standard')

        # Variable column should be normalized properly
        assert not X_normalized['variable'].isnull().any(), \
            "Variable feature should normalize without NaN"


# Integration test combining multiple preprocessing steps
@pytest.mark.p2
def test_end_to_end_preprocessing_pipeline(sample_data):
    """
    Integration test: Load → Validate → Normalize → Check
    This simulates the complete preprocessing pipeline
    """
    feature_cols = [col for col in sample_data.columns
                   if col not in ['sample_id', 'tic_id', 'label', 'status', 'error']]

    # Step 1: Validate no missing values
    is_valid, missing_info = validate_no_missing_values(sample_data[feature_cols])
    assert is_valid, f"Data should have no missing values: {missing_info}"

    # Step 2: Get statistics before normalization
    stats_before = get_feature_statistics(sample_data[feature_cols])
    assert stats_before is not None

    # Step 3: Normalize features
    X_normalized, scaler = normalize_features(
        sample_data[feature_cols],
        method='standard'
    )

    # Step 4: Verify normalization
    means = X_normalized.mean(axis=0)
    stds = X_normalized.std(axis=0)

    assert np.allclose(means, 0, atol=1e-7), "Mean should be ≈0"
    # Note: pandas .std() uses ddof=1, resulting in slight deviation from 1.0
    assert np.allclose(stds, 1, atol=0.01), "Std should be ≈1"

    # Step 5: Apply same scaler to new data (simulating test set)
    test_sample = sample_data.iloc[800:][feature_cols]
    X_test_normalized = apply_scaler(test_sample, scaler)

    assert X_test_normalized.shape == test_sample.shape, \
        "Test data shape should be preserved"
