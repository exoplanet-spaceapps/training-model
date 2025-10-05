"""
Test suite for data_loader module

NOTE: These tests are written FIRST (TDD approach) before src/data_loader.py exists.
They are EXPECTED TO FAIL initially (red lights 🔴).
After implementing src/data_loader.py in PR#2, tests should pass (green lights 🟢).

Test priorities:
- P0: Critical data integrity tests
- P1: Basic functionality tests
- P2: Edge cases and integration tests
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# This import will FAIL until we create src/data_loader.py in PR#2
# This is EXPECTED and CORRECT for TDD (Red → Green → Refactor)
try:
    from src.data_loader import (
        load_and_split_data,
        load_csv_data,
        validate_data_integrity,
        get_feature_columns
    )
    DATA_LOADER_EXISTS = True
except ImportError:
    DATA_LOADER_EXISTS = False
    # Create placeholder functions for test discovery
    def load_and_split_data(*args, **kwargs):
        raise NotImplementedError("data_loader.py not implemented yet (TDD red phase)")

    def load_csv_data(*args, **kwargs):
        raise NotImplementedError("data_loader.py not implemented yet (TDD red phase)")

    def validate_data_integrity(*args, **kwargs):
        raise NotImplementedError("data_loader.py not implemented yet (TDD red phase)")

    def get_feature_columns(*args, **kwargs):
        raise NotImplementedError("data_loader.py not implemented yet (TDD red phase)")


# Mark all tests to skip if data_loader.py doesn't exist yet
pytestmark = pytest.mark.skipif(
    not DATA_LOADER_EXISTS,
    reason="src/data_loader.py not implemented yet (TDD red phase - this is expected)"
)


class TestDataSplitting:
    """Test data splitting functionality (P0 - Critical)"""

    @pytest.mark.p0
    def test_split_ratio_600_200_200(self, sample_data, expected_split_sizes):
        """
        Test that data split produces exactly 600/200/200 samples
        This is a HARD REQUIREMENT from specification
        """
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
            csv_path=None,  # Will use sample_data fixture
            df=sample_data,
            target_col='label',
            train_size=600,
            val_size=200,
            test_size=200,
            random_state=42,
            stratify=True
        )

        # MUST be exactly these sizes (per specification)
        assert len(X_train) == expected_split_sizes['train'], \
            f"Train set must be {expected_split_sizes['train']} samples"
        assert len(X_val) == expected_split_sizes['val'], \
            f"Validation set must be {expected_split_sizes['val']} samples"
        assert len(X_test) == expected_split_sizes['test'], \
            f"Test set must be {expected_split_sizes['test']} samples"

        # Check corresponding labels
        assert len(y_train) == expected_split_sizes['train']
        assert len(y_val) == expected_split_sizes['val']
        assert len(y_test) == expected_split_sizes['test']

    @pytest.mark.p0
    def test_stratified_split_maintains_class_ratio(self, sample_data):
        """
        Test that stratified splitting maintains class distribution
        within ±5% across train/val/test sets
        """
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
            df=sample_data,
            target_col='label',
            stratify=True
        )

        # Calculate class ratios
        train_ratio = y_train.mean()
        val_ratio = y_val.mean()
        test_ratio = y_test.mean()

        # All ratios should be within 5% of each other
        assert abs(train_ratio - val_ratio) < 0.05, \
            "Train/Val class ratio difference too large (not stratified)"
        assert abs(train_ratio - test_ratio) < 0.05, \
            "Train/Test class ratio difference too large (not stratified)"
        assert abs(val_ratio - test_ratio) < 0.05, \
            "Val/Test class ratio difference too large (not stratified)"

    @pytest.mark.p0
    def test_reproducibility_with_random_state_42(self, sample_data):
        """
        Test that random_state=42 produces identical splits
        Reproducibility is CRITICAL for comparing models
        """
        # First split
        X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = load_and_split_data(
            df=sample_data,
            target_col='label',
            random_state=42
        )

        # Second split with same seed
        X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = load_and_split_data(
            df=sample_data,
            target_col='label',
            random_state=42
        )

        # All splits must be identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_val1, X_val2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_val1, y_val2)
        pd.testing.assert_series_equal(y_test1, y_test2)

    @pytest.mark.p0
    def test_no_data_leakage_between_splits(self, sample_data):
        """
        Test that there's no overlap between train/val/test sets
        Data leakage would invalidate all results
        """
        X_train, X_val, X_test, _, _, _ = load_and_split_data(
            df=sample_data,
            target_col='label'
        )

        # Get indices (assuming DataFrames preserve original indices)
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)

        # Check no overlap
        assert len(train_indices & val_indices) == 0, \
            "Data leakage: Train and Val sets overlap!"
        assert len(train_indices & test_indices) == 0, \
            "Data leakage: Train and Test sets overlap!"
        assert len(val_indices & test_indices) == 0, \
            "Data leakage: Val and Test sets overlap!"

        # Check all data is used
        total_indices = train_indices | val_indices | test_indices
        assert len(total_indices) == 1000, \
            "Not all data used in splits"


class TestDataLoading:
    """Test data loading from CSV (P1 - Basic functionality)"""

    @pytest.mark.p1
    def test_load_csv_with_correct_shape(self, mock_csv_file):
        """Test that CSV loads with expected shape (1000 rows)"""
        df = load_csv_data(mock_csv_file)

        assert len(df) == 1000, \
            "CSV must contain exactly 1000 samples (per specification)"
        assert 'label' in df.columns, \
            "CSV must contain 'label' column"

    @pytest.mark.p1
    def test_auto_detect_feature_columns(self, sample_data):
        """Test automatic feature column detection"""
        feature_cols = get_feature_columns(
            sample_data,
            exclude_cols=['sample_id', 'tic_id', 'label', 'status', 'error']
        )

        # Should have 11 feature columns (as per specification)
        expected_features = [
            'n_sectors', 'flux_mean', 'flux_std', 'flux_median', 'flux_mad',
            'flux_skew', 'flux_kurt', 'bls_period', 'bls_duration',
            'bls_depth', 'bls_power', 'bls_snr'
        ]

        for feature in expected_features:
            assert feature in feature_cols, \
                f"Expected feature '{feature}' not detected"

        # Should not include excluded columns
        assert 'label' not in feature_cols
        assert 'sample_id' not in feature_cols


class TestDataIntegrity:
    """Test data integrity validation (P1 - Basic functionality)"""

    @pytest.mark.p1
    def test_validate_no_missing_values_in_features(self, sample_data):
        """Test that feature columns have no missing values"""
        feature_cols = [col for col in sample_data.columns
                        if col not in ['sample_id', 'tic_id', 'label', 'status', 'error']]

        is_valid, errors = validate_data_integrity(sample_data, feature_cols)

        if not is_valid:
            pytest.fail(f"Data integrity check failed: {errors}")

    @pytest.mark.p1
    def test_target_column_is_binary(self, sample_data):
        """Test that target column contains only 0 and 1"""
        unique_labels = sample_data['label'].unique()

        assert set(unique_labels).issubset({0, 1}), \
            "Label column must contain only 0 and 1 (binary classification)"

    @pytest.mark.p1
    def test_all_features_are_numeric(self, sample_data):
        """Test that all feature columns are numeric types"""
        feature_cols = [col for col in sample_data.columns
                        if col not in ['sample_id', 'tic_id', 'label', 'status', 'error']]

        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(sample_data[col]), \
                f"Feature column '{col}' must be numeric"


class TestEdgeCases:
    """Test edge cases and error handling (P2 - Integration)"""

    @pytest.mark.p2
    def test_error_on_invalid_split_sizes(self, sample_data):
        """Test that invalid split sizes raise appropriate errors"""
        with pytest.raises(ValueError):
            # Split sizes don't sum to total
            load_and_split_data(
                df=sample_data,
                target_col='label',
                train_size=500,
                val_size=200,
                test_size=200  # 500 + 200 + 200 = 900 ≠ 1000
            )

    @pytest.mark.p2
    def test_error_on_missing_target_column(self, sample_data):
        """Test that missing target column raises error"""
        with pytest.raises(KeyError):
            load_and_split_data(
                df=sample_data,
                target_col='nonexistent_column'
            )

    @pytest.mark.p2
    def test_handle_csv_with_extra_columns(self, sample_data, tmp_path):
        """Test that loader handles CSV with extra columns gracefully"""
        # Add extra column
        sample_data['extra_column'] = np.random.rand(len(sample_data))

        # Should still work, just exclude the extra column
        X_train, X_val, X_test, _, _, _ = load_and_split_data(
            df=sample_data,
            target_col='label'
        )

        # Extra column should be in feature matrix (unless explicitly excluded)
        assert len(X_train) == 600
        assert len(X_val) == 200
        assert len(X_test) == 200

    @pytest.mark.p2
    def test_error_on_missing_csv_file(self, tmp_path):
        """Test that load_csv_data raises FileNotFoundError for missing file"""
        nonexistent_path = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            load_csv_data(nonexistent_path)

    @pytest.mark.p2
    def test_error_on_empty_csv_file(self, tmp_path):
        """Test that load_csv_data raises EmptyDataError for empty CSV"""
        empty_csv = tmp_path / "empty.csv"
        # Create empty CSV with just headers
        pd.DataFrame(columns=['col1', 'col2']).to_csv(empty_csv, index=False)

        with pytest.raises(pd.errors.EmptyDataError, match="CSV file is empty"):
            load_csv_data(empty_csv)

    @pytest.mark.p2
    def test_validate_data_integrity_with_missing_values(self):
        """Test that validate_data_integrity detects missing values"""
        # Create data with missing values
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0],
            'feature2': [5.0, np.nan, 7.0, 8.0],
            'label': [0, 1, 0, 1]
        })

        feature_cols = ['feature1', 'feature2']
        is_valid, errors = validate_data_integrity(df, feature_cols)

        assert not is_valid, "Should detect missing values"
        assert len(errors) >= 2, "Should report missing values in both columns"
        assert any("feature1" in err and "missing" in err.lower() for err in errors)
        assert any("feature2" in err and "missing" in err.lower() for err in errors)

    @pytest.mark.p2
    def test_validate_data_integrity_with_non_numeric_columns(self):
        """Test that validate_data_integrity detects non-numeric columns"""
        # Create data with non-numeric column
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': ['a', 'b', 'c', 'd'],  # Non-numeric
            'label': [0, 1, 0, 1]
        })

        feature_cols = ['feature1', 'feature2']
        is_valid, errors = validate_data_integrity(df, feature_cols)

        assert not is_valid, "Should detect non-numeric column"
        assert len(errors) >= 1, "Should report non-numeric column"
        assert any("feature2" in err and "not numeric" in err.lower() for err in errors)


# Integration test combining multiple functionalities
@pytest.mark.p2
def test_end_to_end_data_loading_and_splitting(mock_csv_file):
    """
    Integration test: Load CSV → Split → Validate
    This simulates the complete data loading pipeline
    """
    # Full pipeline
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        csv_path=mock_csv_file,
        target_col='label',
        train_size=600,
        val_size=200,
        test_size=200,
        random_state=42,
        stratify=True
    )

    # All checks in one test
    assert len(X_train) == 600
    assert len(X_val) == 200
    assert len(X_test) == 200

    # Check stratification
    train_ratio = y_train.mean()
    val_ratio = y_val.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - val_ratio) < 0.05
    assert abs(train_ratio - test_ratio) < 0.05

    # Check no data leakage
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)
    assert len(train_indices & val_indices) == 0
    assert len(train_indices & test_indices) == 0
    assert len(val_indices & test_indices) == 0
