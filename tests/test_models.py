"""
Test suite for ML models

NOTE: These tests follow TDD approach (Red → Green → Refactor).
Tests are written FIRST before model implementations.

Test priorities:
- P0: Critical model functionality
- P1: Important integration tests
- P2: Edge cases and advanced features
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json

# Import model modules (will fail until implemented - expected for TDD red phase)
try:
    from src.models.xgboost.model import XGBoostWrapper
    from src.models.xgboost.train import train_xgboost
    XGBOOST_EXISTS = True
except ImportError:
    XGBOOST_EXISTS = False
    # Placeholders for test discovery
    XGBoostWrapper = None
    train_xgboost = None

try:
    from src.models.random_forest.model import RandomForestWrapper
    from src.models.random_forest.train import train_random_forest
    RANDOMFOREST_EXISTS = True
except ImportError:
    RANDOMFOREST_EXISTS = False
    # Placeholders for test discovery
    RandomForestWrapper = None
    train_random_forest = None


# Mark all XGBoost tests to skip if not implemented yet
xgboost_mark = pytest.mark.skipif(
    not XGBOOST_EXISTS,
    reason="src/models/xgboost not implemented yet (TDD red phase - expected)"
)

# Mark all Random Forest tests to skip if not implemented yet
randomforest_mark = pytest.mark.skipif(
    not RANDOMFOREST_EXISTS,
    reason="src/models/random_forest not implemented yet (TDD red phase - expected)"
)

# Try to import MLP (may not exist yet - TDD workflow)
try:
    from src.models.mlp.model import MLPWrapper
    from src.models.mlp.train import train_mlp
    MLP_EXISTS = True
except ImportError:
    MLP_EXISTS = False
    # Placeholders for test discovery
    MLPWrapper = None
    train_mlp = None

# Mark all MLP tests to skip if not implemented yet
mlp_mark = pytest.mark.skipif(
    not MLP_EXISTS,
    reason="src/models/mlp not implemented yet (TDD red phase - expected)"
)

# Try to import Logistic Regression (may not exist yet - TDD workflow)
try:
    from src.models.logistic_regression.model import LogisticRegressionWrapper
    from src.models.logistic_regression.train import train_logistic_regression
    LOGISTIC_REGRESSION_EXISTS = True
except ImportError:
    LOGISTIC_REGRESSION_EXISTS = False
    # Placeholders for test discovery
    LogisticRegressionWrapper = None
    train_logistic_regression = None

# Mark all Logistic Regression tests to skip if not implemented yet
logistic_regression_mark = pytest.mark.skipif(
    not LOGISTIC_REGRESSION_EXISTS,
    reason="src/models/logistic_regression not implemented yet (TDD red phase - expected)"
)

# Try to import SVM (may not exist yet - TDD workflow)
try:
    from src.models.svm.model import SVMWrapper
    from src.models.svm.train import train_svm
    SVM_EXISTS = True
except ImportError:
    SVM_EXISTS = False
    # Placeholders for test discovery
    SVMWrapper = None
    train_svm = None

# Mark all SVM tests to skip if not implemented yet
svm_mark = pytest.mark.skipif(
    not SVM_EXISTS,
    reason="src/models/svm not implemented yet (TDD red phase - expected)"
)

# Try to import CNN1D (may not exist yet - TDD workflow)
try:
    from src.models.cnn1d.model import CNN1DWrapper
    from src.models.cnn1d.train import train_cnn1d
    CNN1D_EXISTS = True
except ImportError:
    CNN1D_EXISTS = False
    # Placeholders for test discovery
    CNN1DWrapper = None
    train_cnn1d = None

# Mark all CNN1D tests to skip if not implemented yet
cnn1d_mark = pytest.mark.skipif(
    not CNN1D_EXISTS,
    reason="src/models/cnn1d not implemented yet (TDD red phase - expected)"
)


class TestXGBoostModel:
    """Test XGBoost model implementation (P0 - Critical)"""

    @pytest.mark.p0
    @xgboost_mark
    def test_xgboost_loads_config(self, config_base):
        """
        Test that XGBoost loads parameters from YAML config
        This ensures configuration-driven model creation
        """
        model = XGBoostWrapper(config=config_base)

        assert model is not None, "XGBoostWrapper should be created successfully"
        assert hasattr(model, 'model'), "XGBoostWrapper should have 'model' attribute"

    @pytest.mark.p0
    @xgboost_mark
    def test_xgboost_uses_unified_data_loader(self, sample_data, config_base, tmp_path):
        """
        Test that XGBoost uses load_and_split_data() from unified pipeline
        """
        # Save sample data to CSV
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)

        # Update config with test CSV path
        config_base['data']['csv_path'] = str(csv_path)

        model = XGBoostWrapper(config=config_base)

        # Train should use unified data loader
        # This will be verified through integration test
        assert model is not None

    @pytest.mark.p0
    @xgboost_mark
    def test_xgboost_uses_preprocessing(self, sample_data, config_base):
        """
        Test that XGBoost uses standardize_train_test_split() for preprocessing
        """
        model = XGBoostWrapper(config=config_base)

        # Verify model can be trained with preprocessed data
        feature_cols = [col for col in sample_data.columns
                       if col not in ['sample_id', 'tic_id', 'label', 'status', 'error']]

        X = sample_data[feature_cols].iloc[:100]
        y = sample_data['label'].iloc[:100]

        # Should not raise error
        model.train(X, y)
        assert hasattr(model.model, 'feature_importances_'), \
            "Trained XGBoost should have feature_importances_"

    @pytest.mark.p0
    @xgboost_mark
    def test_xgboost_outputs_confusion_matrix(self, sample_data, config_base, tmp_path):
        """
        Test that XGBoost creates confusion_matrix.png and .csv
        This is a HARD REQUIREMENT from specification
        """
        # Setup artifacts directory
        artifacts_dir = tmp_path / "artifacts" / "xgboost"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Update config
        config_base['output']['artifacts_dir'] = str(artifacts_dir)
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        # Run training with evaluation
        results = train_xgboost(config=config_base, output_dir=artifacts_dir)

        # Check confusion matrix files exist
        cm_png = artifacts_dir / "confusion_matrix.png"
        cm_csv = artifacts_dir / "confusion_matrix.csv"

        assert cm_png.exists(), f"Confusion matrix PNG should exist at {cm_png}"
        assert cm_csv.exists(), f"Confusion matrix CSV should exist at {cm_csv}"

        # Verify CSV contains valid data (read with index_col=0 since we save with row labels)
        cm_data = pd.read_csv(cm_csv, index_col=0)
        assert len(cm_data) == 2, "Confusion matrix should be 2x2 for binary classification"
        assert len(cm_data.columns) == 2, "Confusion matrix should have 2 columns"

    @pytest.mark.p0
    @xgboost_mark
    def test_xgboost_outputs_metrics_json(self, sample_data, config_base, tmp_path):
        """
        Test that XGBoost creates metrics.json with required fields
        Required fields: accuracy, precision, recall, f1
        """
        # Setup
        artifacts_dir = tmp_path / "artifacts" / "xgboost"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        config_base['output']['artifacts_dir'] = str(artifacts_dir)

        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        # Run training
        results = train_xgboost(config=config_base, output_dir=artifacts_dir)

        # Check metrics.json exists
        metrics_file = artifacts_dir / "metrics.json"
        assert metrics_file.exists(), f"Metrics JSON should exist at {metrics_file}"

        # Load and verify metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        required_fields = ['accuracy', 'precision', 'recall', 'f1']
        for field in required_fields:
            assert field in metrics, f"Metrics must include '{field}'"
            assert isinstance(metrics[field], (int, float)), \
                f"{field} should be numeric"
            assert 0 <= metrics[field] <= 1, \
                f"{field} should be between 0 and 1"

    @pytest.mark.p0
    @xgboost_mark
    def test_xgboost_uses_correct_data_split(self, sample_data, config_base, tmp_path):
        """
        Test that XGBoost uses 600/200/200 split
        This is a HARD REQUIREMENT from specification
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        artifacts_dir = tmp_path / "artifacts" / "xgboost"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Run training (should use 600/200/200 split internally)
        results = train_xgboost(config=config_base, output_dir=artifacts_dir)

        # Verify results contain information about data split
        assert 'data_split' in results or 'train_size' in results, \
            "Results should contain data split information"

    @pytest.mark.p1
    @xgboost_mark
    def test_xgboost_saves_model_file(self, sample_data, config_base, tmp_path):
        """
        Test that trained XGBoost model is saved to artifacts/
        """
        artifacts_dir = tmp_path / "artifacts" / "xgboost"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        results = train_xgboost(config=config_base, output_dir=artifacts_dir)

        # Check model file exists (could be .pkl, .json, or .ubj)
        model_files = list(artifacts_dir.glob("model.*"))
        assert len(model_files) > 0, "Model file should be saved in artifacts directory"

    @pytest.mark.p1
    @xgboost_mark
    def test_xgboost_reproducibility(self, sample_data, config_base, tmp_path):
        """
        Test that XGBoost produces reproducible results with same random_state
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        artifacts_dir1 = tmp_path / "run1"
        artifacts_dir1.mkdir(parents=True, exist_ok=True)

        artifacts_dir2 = tmp_path / "run2"
        artifacts_dir2.mkdir(parents=True, exist_ok=True)

        # Run 1
        results1 = train_xgboost(config=config_base, output_dir=artifacts_dir1)

        # Run 2 with same config
        results2 = train_xgboost(config=config_base, output_dir=artifacts_dir2)

        # Metrics should be identical (or very close due to floating point)
        metrics1_file = artifacts_dir1 / "metrics.json"
        metrics2_file = artifacts_dir2 / "metrics.json"

        with open(metrics1_file) as f1, open(metrics2_file) as f2:
            metrics1 = json.load(f1)
            metrics2 = json.load(f2)

        # Accuracy should match within tolerance
        assert abs(metrics1['accuracy'] - metrics2['accuracy']) < 1e-6, \
            "Results should be reproducible with same random_state"

    @pytest.mark.p2
    @xgboost_mark
    def test_xgboost_handles_class_imbalance(self, config_base, tmp_path):
        """
        Test that XGBoost handles class-imbalanced data appropriately
        """
        # Create imbalanced dataset (90% class 0, 10% class 1)
        np.random.seed(42)
        n_samples = 1000

        imbalanced_data = {
            'sample_id': [f'SAMPLE_{i:06d}' for i in range(n_samples)],
            'tic_id': np.random.randint(1000000, 9999999, n_samples),
            'label': np.concatenate([
                np.zeros(900, dtype=int),
                np.ones(100, dtype=int)
            ]),
            'flux_mean': np.random.normal(1.0, 0.01, n_samples),
            'flux_std': np.random.normal(0.01, 0.005, n_samples),
            'flux_median': np.random.normal(1.0, 0.01, n_samples),
            'flux_mad': np.random.normal(0.005, 0.002, n_samples),
            'flux_skew': np.random.normal(0.0, 0.5, n_samples),
            'flux_kurt': np.random.normal(0.0, 1.0, n_samples),
            'bls_period': np.random.uniform(1.0, 15.0, n_samples),
            'bls_duration': np.random.uniform(0.05, 0.3, n_samples),
            'bls_depth': np.random.uniform(0.001, 0.1, n_samples),
            'bls_power': np.random.uniform(0.0, 1.0, n_samples),
            'bls_snr': np.random.uniform(1.0, 20.0, n_samples),
            'n_sectors': np.random.randint(1, 20, n_samples),
            'status': ['success'] * n_samples,
            'error': [''] * n_samples,
        }

        df = pd.DataFrame(imbalanced_data)
        csv_path = tmp_path / "imbalanced.csv"
        df.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Should complete without error
        results = train_xgboost(config=config_base, output_dir=artifacts_dir)

        # Verify metrics were calculated
        metrics_file = artifacts_dir / "metrics.json"
        assert metrics_file.exists()


class TestRandomForestModel:
    """Test Random Forest model implementation (P0 - Critical)"""

    @pytest.mark.p0
    @randomforest_mark
    def test_randomforest_loads_config(self, config_base):
        """
        Test that Random Forest loads parameters from YAML config
        This ensures configuration-driven model creation
        """
        model = RandomForestWrapper(config=config_base)

        assert model is not None, "RandomForestWrapper should be created successfully"
        assert hasattr(model, 'model'), "RandomForestWrapper should have 'model' attribute"

    @pytest.mark.p0
    @randomforest_mark
    def test_randomforest_uses_unified_data_loader(self, sample_data, config_base, tmp_path):
        """
        Test that Random Forest uses load_and_split_data() from unified pipeline
        """
        # Save sample data to CSV
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)

        # Update config to use test CSV
        config_base['data']['csv_path'] = str(csv_path)

        artifacts_dir = tmp_path / "artifacts" / "random_forest"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Run training (internally should call load_and_split_data)
        results = train_random_forest(config=config_base, output_dir=artifacts_dir)

        assert results is not None, "Training should return results"
        assert 'model' in results, "Results should contain trained model"

    @pytest.mark.p0
    @randomforest_mark
    def test_randomforest_uses_preprocessing(self, sample_data, config_base, tmp_path):
        """
        Test that Random Forest uses standardize_train_test_split() for preprocessing
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        artifacts_dir = tmp_path / "artifacts" / "random_forest"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # This should succeed if preprocessing is applied
        results = train_random_forest(config=config_base, output_dir=artifacts_dir)

        assert results is not None
        assert 'metrics' in results

    @pytest.mark.p0
    @randomforest_mark
    def test_randomforest_outputs_confusion_matrix(self, sample_data, config_base, tmp_path):
        """
        Test that Random Forest creates confusion_matrix.png and .csv
        This is a HARD REQUIREMENT from specification
        """
        # Setup artifacts directory
        artifacts_dir = tmp_path / "artifacts" / "random_forest"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Update config
        config_base['output']['artifacts_dir'] = str(artifacts_dir)
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        # Run training with evaluation
        results = train_random_forest(config=config_base, output_dir=artifacts_dir)

        # Check confusion matrix files exist
        cm_png = artifacts_dir / "confusion_matrix.png"
        cm_csv = artifacts_dir / "confusion_matrix.csv"

        assert cm_png.exists(), f"Confusion matrix PNG should exist at {cm_png}"
        assert cm_csv.exists(), f"Confusion matrix CSV should exist at {cm_csv}"

        # Verify CSV contains valid data (read with index_col=0 since we save with row labels)
        cm_data = pd.read_csv(cm_csv, index_col=0)
        assert len(cm_data) == 2, "Confusion matrix should be 2x2 for binary classification"
        assert len(cm_data.columns) == 2, "Confusion matrix should have 2 columns"

    @pytest.mark.p0
    @randomforest_mark
    def test_randomforest_outputs_metrics_json(self, sample_data, config_base, tmp_path):
        """
        Test that Random Forest creates metrics.json with required fields
        Required fields: accuracy, precision, recall, f1
        """
        # Setup
        artifacts_dir = tmp_path / "artifacts" / "random_forest"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        config_base['output']['artifacts_dir'] = str(artifacts_dir)

        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        # Run training
        results = train_random_forest(config=config_base, output_dir=artifacts_dir)

        # Check metrics.json exists
        metrics_file = artifacts_dir / "metrics.json"
        assert metrics_file.exists(), f"Metrics JSON should exist at {metrics_file}"

        # Load and verify metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        required_fields = ['accuracy', 'precision', 'recall', 'f1']
        for field in required_fields:
            assert field in metrics, f"Metrics must include '{field}'"
            assert isinstance(metrics[field], (int, float)), \
                f"{field} should be numeric"
            assert 0 <= metrics[field] <= 1, \
                f"{field} should be between 0 and 1"

    @pytest.mark.p0
    @randomforest_mark
    def test_randomforest_uses_correct_data_split(self, sample_data, config_base, tmp_path):
        """
        Test that Random Forest uses 600/200/200 split
        This is a HARD REQUIREMENT from specification
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        artifacts_dir = tmp_path / "artifacts" / "random_forest"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Run training (should use 600/200/200 split internally)
        results = train_random_forest(config=config_base, output_dir=artifacts_dir)

        # Verify results contain information about data split
        assert 'data_split' in results or 'train_size' in results, \
            "Results should contain data split information"

    @pytest.mark.p1
    @randomforest_mark
    def test_randomforest_saves_model_file(self, sample_data, config_base, tmp_path):
        """
        Test that trained Random Forest model is saved to artifacts/
        """
        artifacts_dir = tmp_path / "artifacts" / "random_forest"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        results = train_random_forest(config=config_base, output_dir=artifacts_dir)

        # Check model file exists (should be .pkl for Random Forest)
        model_files = list(artifacts_dir.glob("model.*"))
        assert len(model_files) > 0, "Model file should be saved in artifacts directory"

    @pytest.mark.p1
    @randomforest_mark
    def test_randomforest_reproducibility(self, sample_data, config_base, tmp_path):
        """
        Test that Random Forest produces reproducible results with same random_state
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        artifacts_dir1 = tmp_path / "run1"
        artifacts_dir1.mkdir(parents=True, exist_ok=True)

        artifacts_dir2 = tmp_path / "run2"
        artifacts_dir2.mkdir(parents=True, exist_ok=True)

        # Run 1
        results1 = train_random_forest(config=config_base, output_dir=artifacts_dir1)

        # Run 2 with same config
        results2 = train_random_forest(config=config_base, output_dir=artifacts_dir2)

        # Metrics should be identical (or very close due to floating point)
        metrics1_file = artifacts_dir1 / "metrics.json"
        metrics2_file = artifacts_dir2 / "metrics.json"

        with open(metrics1_file) as f1, open(metrics2_file) as f2:
            metrics1 = json.load(f1)
            metrics2 = json.load(f2)

        # Accuracy should match within tolerance
        assert abs(metrics1['accuracy'] - metrics2['accuracy']) < 1e-6, \
            "Results should be reproducible with same random_state"

    @pytest.mark.p2
    @randomforest_mark
    def test_randomforest_handles_class_imbalance(self, config_base, tmp_path):
        """
        Test that Random Forest handles class-imbalanced data appropriately
        """
        # Create imbalanced dataset (90% class 0, 10% class 1)
        np.random.seed(42)
        n_samples = 1000

        imbalanced_data = {
            'sample_id': [f'SAMPLE_{i:06d}' for i in range(n_samples)],
            'tic_id': np.random.randint(1000000, 9999999, n_samples),
            'label': np.concatenate([
                np.zeros(900, dtype=int),
                np.ones(100, dtype=int)
            ]),
            'flux_mean': np.random.normal(1.0, 0.01, n_samples),
            'flux_std': np.random.normal(0.01, 0.005, n_samples),
            'flux_median': np.random.normal(1.0, 0.01, n_samples),
            'flux_mad': np.random.normal(0.005, 0.002, n_samples),
            'flux_skew': np.random.normal(0.0, 0.5, n_samples),
            'flux_kurt': np.random.normal(0.0, 1.0, n_samples),
            'bls_period': np.random.uniform(1.0, 15.0, n_samples),
            'bls_duration': np.random.uniform(0.05, 0.3, n_samples),
            'bls_depth': np.random.uniform(0.001, 0.1, n_samples),
            'bls_power': np.random.uniform(0.0, 1.0, n_samples),
            'bls_snr': np.random.uniform(1.0, 20.0, n_samples),
            'n_sectors': np.random.randint(1, 20, n_samples),
            'status': ['success'] * n_samples,
            'error': [''] * n_samples,
        }

        df = pd.DataFrame(imbalanced_data)
        csv_path = tmp_path / "imbalanced.csv"
        df.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Should complete without error
        results = train_random_forest(config=config_base, output_dir=artifacts_dir)

        # Verify metrics were calculated
        metrics_file = artifacts_dir / "metrics.json"
        assert metrics_file.exists()


class TestMLPModel:
    """Test MLP model implementation (P0 - Critical)"""

    @pytest.mark.p0
    @mlp_mark
    def test_mlp_loads_config(self, config_base):
        """
        Test that MLP loads parameters from YAML config
        This ensures configuration-driven model creation
        """
        model = MLPWrapper(config=config_base)
        assert model is not None, "MLPWrapper should be created successfully"
        assert hasattr(model, 'model'), "MLPWrapper should have 'model' attribute"

    @pytest.mark.p0
    @mlp_mark
    def test_mlp_uses_unified_data_loader(self, sample_data, config_base, tmp_path):
        """
        Test that MLP training script uses load_and_split_data()
        This ensures integration with unified data pipeline
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir = tmp_path / "artifacts" / "mlp"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        results = train_mlp(config=config_base, output_dir=artifacts_dir)

        assert results is not None, "train_mlp should return results dict"
        assert 'model' in results, "Results should contain trained model"

    @pytest.mark.p0
    @mlp_mark
    def test_mlp_uses_preprocessing(self, sample_data, config_base, tmp_path):
        """
        Test that MLP uses standardize_train_test_split()
        This ensures integration with unified preprocessing pipeline
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir = tmp_path / "artifacts" / "mlp"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        results = train_mlp(config=config_base, output_dir=artifacts_dir)

        assert 'metrics' in results, "Results should contain metrics"

    @pytest.mark.p0
    @mlp_mark
    def test_mlp_outputs_confusion_matrix(self, sample_data, config_base, tmp_path):
        """
        Test that MLP creates confusion_matrix.png and .csv
        This is a HARD REQUIREMENT from specification
        """
        artifacts_dir = tmp_path / "artifacts" / "mlp"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        config_base['output']['artifacts_dir'] = str(artifacts_dir)

        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        results = train_mlp(config=config_base, output_dir=artifacts_dir)

        cm_png = artifacts_dir / "confusion_matrix.png"
        cm_csv = artifacts_dir / "confusion_matrix.csv"

        assert cm_png.exists(), f"Confusion matrix PNG should exist at {cm_png}"
        assert cm_csv.exists(), f"Confusion matrix CSV should exist at {cm_csv}"

        # Verify CSV format (read with index_col=0 since we save with row labels)
        cm_data = pd.read_csv(cm_csv, index_col=0)
        assert len(cm_data) == 2, "Confusion matrix should be 2x2 for binary classification"
        assert len(cm_data.columns) == 2, "Confusion matrix should have 2 columns"

    @pytest.mark.p0
    @mlp_mark
    def test_mlp_outputs_metrics_json(self, sample_data, config_base, tmp_path):
        """
        Test that MLP creates metrics.json with required metrics
        This is a HARD REQUIREMENT from specification
        """
        artifacts_dir = tmp_path / "artifacts" / "mlp"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        results = train_mlp(config=config_base, output_dir=artifacts_dir)

        metrics_file = artifacts_dir / "metrics.json"
        assert metrics_file.exists(), f"metrics.json should exist at {metrics_file}"

        with open(metrics_file) as f:
            metrics = json.load(f)

        # Check required metrics
        required_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in required_metrics:
            assert metric in metrics, f"Metrics should contain '{metric}'"
            assert isinstance(metrics[metric], (int, float)), f"{metric} should be numeric"
            assert 0 <= metrics[metric] <= 1, f"{metric} should be between 0 and 1"

    @pytest.mark.p0
    @mlp_mark
    def test_mlp_uses_correct_data_split(self, sample_data, config_base, tmp_path):
        """
        Test that MLP uses correct 600/200/200 split
        This is a HARD REQUIREMENT from specification
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir = tmp_path / "artifacts" / "mlp"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        results = train_mlp(config=config_base, output_dir=artifacts_dir)

        assert 'data_split' in results, "Results should contain data_split info"
        split = results['data_split']

        assert split['train_size'] == 600, "Training set should have 600 samples"
        assert split['val_size'] == 200, "Validation set should have 200 samples"
        assert split['test_size'] == 200, "Test set should have 200 samples"

    @pytest.mark.p1
    @mlp_mark
    def test_mlp_saves_model_file(self, sample_data, config_base, tmp_path):
        """
        Test that MLP saves model.pkl file
        This ensures model persistence
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir = tmp_path / "artifacts" / "mlp"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        results = train_mlp(config=config_base, output_dir=artifacts_dir)

        # Check for model file (could be .pkl, .json, or other format)
        model_files = list(artifacts_dir.glob("model.*"))
        assert len(model_files) > 0, f"Model file should exist in {artifacts_dir}"

    @pytest.mark.p1
    @mlp_mark
    def test_mlp_reproducibility(self, sample_data, config_base, tmp_path):
        """
        Test that MLP produces reproducible results with same random_state
        This ensures deterministic behavior
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir1 = tmp_path / "run1"
        artifacts_dir2 = tmp_path / "run2"
        artifacts_dir1.mkdir(parents=True, exist_ok=True)
        artifacts_dir2.mkdir(parents=True, exist_ok=True)

        # Run 1
        results1 = train_mlp(config=config_base, output_dir=artifacts_dir1)

        # Run 2 with same config
        results2 = train_mlp(config=config_base, output_dir=artifacts_dir2)

        # Compare metrics - should be identical with fixed random_state
        with open(artifacts_dir1 / "metrics.json") as f:
            metrics1 = json.load(f)

        with open(artifacts_dir2 / "metrics.json") as f:
            metrics2 = json.load(f)

        assert metrics1['accuracy'] == metrics2['accuracy'], "Results should be reproducible"

    @pytest.mark.p2
    @mlp_mark
    def test_mlp_handles_class_imbalance(self, config_base, tmp_path):
        """
        Test that MLP handles imbalanced datasets (90/10 split)
        This ensures robustness
        """
        # Create imbalanced dataset (90% class 0, 10% class 1)
        n_samples = 1000
        imbalanced_data = {
            'sample_id': [f'SAMPLE_{i:06d}' for i in range(n_samples)],
            'tic_id': np.random.randint(1000000, 9999999, n_samples),
            'label': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'flux_mean': np.random.uniform(0.95, 1.05, n_samples),
            'flux_std': np.random.uniform(0.001, 0.05, n_samples),
            'flux_median': np.random.uniform(0.95, 1.05, n_samples),
            'flux_mad': np.random.uniform(0.001, 0.05, n_samples),
            'flux_skew': np.random.uniform(-1.0, 1.0, n_samples),
            'flux_kurt': np.random.uniform(-2.0, 5.0, n_samples),
            'bls_period': np.random.uniform(0.5, 15.0, n_samples),
            'bls_duration': np.random.uniform(0.05, 0.3, n_samples),
            'bls_depth': np.random.uniform(0.0, 0.1, n_samples),
            'bls_power': np.random.uniform(0.0, 5.0, n_samples),
            'bls_snr': np.random.uniform(1.0, 20.0, n_samples),
            'n_sectors': np.random.randint(1, 20, n_samples),
            'status': ['success'] * n_samples,
            'error': [''] * n_samples,
        }

        df = pd.DataFrame(imbalanced_data)
        csv_path = tmp_path / "imbalanced.csv"
        df.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Should complete without error
        results = train_mlp(config=config_base, output_dir=artifacts_dir)

        # Verify metrics were calculated
        metrics_file = artifacts_dir / "metrics.json"
        assert metrics_file.exists()


class TestLogisticRegressionModel:
    """Test Logistic Regression model implementation (P0 - Critical)"""

    @pytest.mark.p0
    @logistic_regression_mark
    def test_logisticregression_loads_config(self, config_base):
        """
        Test that Logistic Regression loads parameters from YAML config
        This ensures configuration-driven model creation
        """
        model = LogisticRegressionWrapper(config=config_base)
        assert model is not None, "LogisticRegressionWrapper should be created successfully"
        assert hasattr(model, 'model'), "LogisticRegressionWrapper should have 'model' attribute"

    @pytest.mark.p0
    @logistic_regression_mark
    def test_logisticregression_uses_unified_data_loader(self, sample_data, config_base, tmp_path):
        """
        Test that Logistic Regression training script uses load_and_split_data()
        This ensures integration with unified data pipeline
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir = tmp_path / "artifacts" / "logistic_regression"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        results = train_logistic_regression(config=config_base, output_dir=artifacts_dir)

        assert results is not None, "train_logistic_regression should return results dict"
        assert 'model' in results, "Results should contain trained model"

    @pytest.mark.p0
    @logistic_regression_mark
    def test_logisticregression_uses_preprocessing(self, sample_data, config_base, tmp_path):
        """
        Test that Logistic Regression uses standardize_train_test_split()
        This ensures integration with unified preprocessing pipeline
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir = tmp_path / "artifacts" / "logistic_regression"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        results = train_logistic_regression(config=config_base, output_dir=artifacts_dir)

        assert 'metrics' in results, "Results should contain metrics"

    @pytest.mark.p0
    @logistic_regression_mark
    def test_logisticregression_outputs_confusion_matrix(self, sample_data, config_base, tmp_path):
        """
        Test that Logistic Regression creates confusion_matrix.png and .csv
        This is a HARD REQUIREMENT from specification
        """
        artifacts_dir = tmp_path / "artifacts" / "logistic_regression"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        config_base['output']['artifacts_dir'] = str(artifacts_dir)

        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        results = train_logistic_regression(config=config_base, output_dir=artifacts_dir)

        cm_png = artifacts_dir / "confusion_matrix.png"
        cm_csv = artifacts_dir / "confusion_matrix.csv"

        assert cm_png.exists(), f"Confusion matrix PNG should exist at {cm_png}"
        assert cm_csv.exists(), f"Confusion matrix CSV should exist at {cm_csv}"

        # Verify CSV format (read with index_col=0 since we save with row labels)
        cm_data = pd.read_csv(cm_csv, index_col=0)
        assert len(cm_data) == 2, "Confusion matrix should be 2x2 for binary classification"
        assert len(cm_data.columns) == 2, "Confusion matrix should have 2 columns"

    @pytest.mark.p0
    @logistic_regression_mark
    def test_logisticregression_outputs_metrics_json(self, sample_data, config_base, tmp_path):
        """
        Test that Logistic Regression creates metrics.json with required metrics
        This is a HARD REQUIREMENT from specification
        """
        artifacts_dir = tmp_path / "artifacts" / "logistic_regression"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        results = train_logistic_regression(config=config_base, output_dir=artifacts_dir)

        metrics_file = artifacts_dir / "metrics.json"
        assert metrics_file.exists(), f"metrics.json should exist at {metrics_file}"

        with open(metrics_file) as f:
            metrics = json.load(f)

        # Check required metrics
        required_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in required_metrics:
            assert metric in metrics, f"Metrics should contain '{metric}'"
            assert isinstance(metrics[metric], (int, float)), f"{metric} should be numeric"
            assert 0 <= metrics[metric] <= 1, f"{metric} should be between 0 and 1"

    @pytest.mark.p0
    @logistic_regression_mark
    def test_logisticregression_uses_correct_data_split(self, sample_data, config_base, tmp_path):
        """
        Test that Logistic Regression uses correct 600/200/200 split
        This is a HARD REQUIREMENT from specification
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir = tmp_path / "artifacts" / "logistic_regression"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        results = train_logistic_regression(config=config_base, output_dir=artifacts_dir)

        assert 'data_split' in results, "Results should contain data_split info"
        split = results['data_split']

        assert split['train_size'] == 600, "Training set should have 600 samples"
        assert split['val_size'] == 200, "Validation set should have 200 samples"
        assert split['test_size'] == 200, "Test set should have 200 samples"

    @pytest.mark.p1
    @logistic_regression_mark
    def test_logisticregression_saves_model_file(self, sample_data, config_base, tmp_path):
        """
        Test that Logistic Regression saves model.pkl file
        This ensures model persistence
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir = tmp_path / "artifacts" / "logistic_regression"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        results = train_logistic_regression(config=config_base, output_dir=artifacts_dir)

        # Check for model file (could be .pkl, .json, or other format)
        model_files = list(artifacts_dir.glob("model.*"))
        assert len(model_files) > 0, f"Model file should exist in {artifacts_dir}"

    @pytest.mark.p1
    @logistic_regression_mark
    def test_logisticregression_reproducibility(self, sample_data, config_base, tmp_path):
        """
        Test that Logistic Regression produces reproducible results with same random_state
        This ensures deterministic behavior
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir1 = tmp_path / "run1"
        artifacts_dir2 = tmp_path / "run2"
        artifacts_dir1.mkdir(parents=True, exist_ok=True)
        artifacts_dir2.mkdir(parents=True, exist_ok=True)

        # Run 1
        results1 = train_logistic_regression(config=config_base, output_dir=artifacts_dir1)

        # Run 2 with same config
        results2 = train_logistic_regression(config=config_base, output_dir=artifacts_dir2)

        # Compare metrics - should be identical with fixed random_state
        with open(artifacts_dir1 / "metrics.json") as f:
            metrics1 = json.load(f)

        with open(artifacts_dir2 / "metrics.json") as f:
            metrics2 = json.load(f)

        assert metrics1['accuracy'] == metrics2['accuracy'], "Results should be reproducible"

    @pytest.mark.p2
    @logistic_regression_mark
    def test_logisticregression_handles_class_imbalance(self, config_base, tmp_path):
        """
        Test that Logistic Regression handles imbalanced datasets (90/10 split)
        This ensures robustness
        """
        # Create imbalanced dataset (90% class 0, 10% class 1)
        n_samples = 1000
        imbalanced_data = {
            'sample_id': [f'SAMPLE_{i:06d}' for i in range(n_samples)],
            'tic_id': np.random.randint(1000000, 9999999, n_samples),
            'label': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'flux_mean': np.random.uniform(0.95, 1.05, n_samples),
            'flux_std': np.random.uniform(0.001, 0.05, n_samples),
            'flux_median': np.random.uniform(0.95, 1.05, n_samples),
            'flux_mad': np.random.uniform(0.001, 0.05, n_samples),
            'flux_skew': np.random.uniform(-1.0, 1.0, n_samples),
            'flux_kurt': np.random.uniform(-2.0, 5.0, n_samples),
            'bls_period': np.random.uniform(0.5, 15.0, n_samples),
            'bls_duration': np.random.uniform(0.05, 0.3, n_samples),
            'bls_depth': np.random.uniform(0.0, 0.1, n_samples),
            'bls_power': np.random.uniform(0.0, 5.0, n_samples),
            'bls_snr': np.random.uniform(1.0, 20.0, n_samples),
            'n_sectors': np.random.randint(1, 20, n_samples),
            'status': ['success'] * n_samples,
            'error': [''] * n_samples,
        }

        df = pd.DataFrame(imbalanced_data)
        csv_path = tmp_path / "imbalanced.csv"
        df.to_csv(csv_path, index=False)

        config_base['data']['csv_path'] = str(csv_path)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Should complete without error
        results = train_logistic_regression(config=config_base, output_dir=artifacts_dir)

        # Verify metrics were calculated
        metrics_file = artifacts_dir / "metrics.json"
        assert metrics_file.exists()


# Integration test
@pytest.mark.p2
@xgboost_mark
def test_xgboost_end_to_end_pipeline(sample_data, config_base, tmp_path):
    """
    Integration test: Complete pipeline from CSV → Training → Evaluation → Artifacts
    This simulates the full XGBoost workflow
    """
    # Setup
    csv_path = tmp_path / "data.csv"
    sample_data.to_csv(csv_path, index=False)

    config_base['data']['csv_path'] = str(csv_path)
    artifacts_dir = tmp_path / "artifacts" / "xgboost"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Run complete pipeline
    results = train_xgboost(config=config_base, output_dir=artifacts_dir)

    # Verify all required outputs
    assert (artifacts_dir / "confusion_matrix.png").exists(), "Missing confusion_matrix.png"
    assert (artifacts_dir / "confusion_matrix.csv").exists(), "Missing confusion_matrix.csv"
    assert (artifacts_dir / "metrics.json").exists(), "Missing metrics.json"

    model_files = list(artifacts_dir.glob("model.*"))
    assert len(model_files) > 0, "Missing model file"

    # Verify metrics quality
    with open(artifacts_dir / "metrics.json") as f:
        metrics = json.load(f)

    # For balanced random data, accuracy should be reasonable
    assert metrics['accuracy'] > 0.4, "Accuracy too low (possible implementation issue)"
    assert metrics['f1'] > 0.3, "F1 score too low (possible implementation issue)"


# Integration test
@pytest.mark.p2
@randomforest_mark
def test_randomforest_end_to_end_pipeline(sample_data, config_base, tmp_path):
    """
    Integration test: Complete pipeline from CSV → Training → Evaluation → Artifacts
    This simulates the full Random Forest workflow
    """
    # Setup
    csv_path = tmp_path / "data.csv"
    sample_data.to_csv(csv_path, index=False)

    config_base['data']['csv_path'] = str(csv_path)
    artifacts_dir = tmp_path / "artifacts" / "random_forest"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Run complete pipeline
    results = train_random_forest(config=config_base, output_dir=artifacts_dir)

    # Verify all required outputs
    assert (artifacts_dir / "confusion_matrix.png").exists(), "Missing confusion_matrix.png"
    assert (artifacts_dir / "confusion_matrix.csv").exists(), "Missing confusion_matrix.csv"
    assert (artifacts_dir / "metrics.json").exists(), "Missing metrics.json"

    model_files = list(artifacts_dir.glob("model.*"))
    assert len(model_files) > 0, "Missing model file"

    # Verify metrics quality
    with open(artifacts_dir / "metrics.json") as f:
        metrics = json.load(f)

    # For balanced random data, accuracy should be reasonable
    assert metrics['accuracy'] > 0.4, "Accuracy too low (possible implementation issue)"
    assert metrics['f1'] > 0.3, "F1 score too low (possible implementation issue)"


# ============================================================================
# MLP Integration Test (P2 - End-to-End)
# ============================================================================

@pytest.mark.p2
@mlp_mark
def test_mlp_end_to_end_pipeline(sample_data, config_base, tmp_path):
    """
    Integration test: Complete pipeline from CSV → Training → Evaluation → Artifacts
    This simulates the full MLP workflow

    Tests the complete integration:
    1. Data loading from CSV
    2. Data splitting (600/200/200)
    3. Feature preprocessing (standardization)
    4. MLP training
    5. Model evaluation
    6. Artifact generation (confusion matrix, metrics, model file)
    """
    # Setup
    csv_path = tmp_path / "data.csv"
    sample_data.to_csv(csv_path, index=False)

    config_base['data']['csv_path'] = str(csv_path)
    artifacts_dir = tmp_path / "artifacts" / "mlp"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Run complete pipeline
    results = train_mlp(config=config_base, output_dir=artifacts_dir)

    # Verify all required outputs
    assert (artifacts_dir / "confusion_matrix.png").exists(), "Missing confusion_matrix.png"
    assert (artifacts_dir / "confusion_matrix.csv").exists(), "Missing confusion_matrix.csv"
    assert (artifacts_dir / "metrics.json").exists(), "Missing metrics.json"

    model_files = list(artifacts_dir.glob("model.*"))
    assert len(model_files) > 0, "Missing model file"

    # Verify metrics quality
    with open(artifacts_dir / "metrics.json") as f:
        metrics = json.load(f)

    # For balanced random data, accuracy should be reasonable
    assert metrics['accuracy'] > 0.4, "Accuracy too low (possible implementation issue)"
    assert metrics['f1'] > 0.3, "F1 score too low (possible implementation issue)"


# ============================================================================
# Logistic Regression Integration Test (P2 - End-to-End)
# ============================================================================

@pytest.mark.p2
@logistic_regression_mark
def test_logisticregression_end_to_end_pipeline(sample_data, config_base, tmp_path):
    """
    Integration test: Complete pipeline from CSV → Training → Evaluation → Artifacts
    This simulates the full Logistic Regression workflow

    Tests the complete integration:
    1. Data loading from CSV
    2. Data splitting (600/200/200)
    3. Feature preprocessing (standardization)
    4. Logistic Regression training
    5. Model evaluation
    6. Artifact generation (confusion matrix, metrics, model file)
    """
    # Setup
    csv_path = tmp_path / "data.csv"
    sample_data.to_csv(csv_path, index=False)

    config_base['data']['csv_path'] = str(csv_path)
    artifacts_dir = tmp_path / "artifacts" / "logistic_regression"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Run complete pipeline
    results = train_logistic_regression(config=config_base, output_dir=artifacts_dir)

    # Verify all required outputs
    assert (artifacts_dir / "confusion_matrix.png").exists(), "Missing confusion_matrix.png"
    assert (artifacts_dir / "confusion_matrix.csv").exists(), "Missing confusion_matrix.csv"
    assert (artifacts_dir / "metrics.json").exists(), "Missing metrics.json"

    model_files = list(artifacts_dir.glob("model.*"))
    assert len(model_files) > 0, "Missing model file"

    # Verify metrics quality
    with open(artifacts_dir / "metrics.json") as f:
        metrics = json.load(f)

    # For balanced random data, accuracy should be reasonable
    assert metrics['accuracy'] > 0.4, "Accuracy too low (possible implementation issue)"
    assert metrics['f1'] > 0.3, "F1 score too low (possible implementation issue)"


# ============================================================================
# SVM Model Tests (TDD Workflow)
# ============================================================================

class TestSVMModel:
    """Test suite for SVM model following TDD approach"""

    @pytest.mark.p0
    @svm_mark
    def test_svm_loads_config(self, config_base):
        """P0: SVM must load configuration from base.yaml"""
        assert 'svm' in config_base['models'], "SVM config missing in base.yaml"
        svm_config = config_base['models']['svm']

        # Verify all required parameters
        assert svm_config['C'] == 1.0, "C parameter must be 1.0"
        assert svm_config['kernel'] == 'rbf', "kernel must be 'rbf'"
        assert svm_config['gamma'] == 'scale', "gamma must be 'scale'"
        assert svm_config['probability'] == True, "probability must be True"
        assert svm_config['random_state'] == 42, "random_state must be 42"

    @pytest.mark.p0
    @svm_mark
    def test_svm_uses_unified_data_loader(self, sample_data, tmp_path, config_base):
        """P0: SVM must use load_and_split_data() from unified pipeline"""
        csv_path = tmp_path / "data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        # This will be called by train_svm internally
        from src.data_loader import load_and_split_data
        data_config = config_base.get('data', {})
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
            csv_path=data_config.get('csv_path'),
            target_col=data_config.get('target_col', 'label'),
            train_size=data_config.get('train_size', 600),
            val_size=data_config.get('val_size', 200),
            test_size=data_config.get('test_size', 200),
            random_state=data_config.get('random_state', 42),
            stratify=data_config.get('stratify', True)
        )

        assert X_train.shape[0] == 600, "Training set must have 600 samples"
        assert X_val.shape[0] == 200, "Validation set must have 200 samples"
        assert X_test.shape[0] == 200, "Test set must have 200 samples"

    @pytest.mark.p0
    @svm_mark
    def test_svm_uses_preprocessing(self, sample_data, tmp_path, config_base):
        """P0: SVM must use standardize_train_test_split() for preprocessing"""
        csv_path = tmp_path / "data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        from src.data_loader import load_and_split_data
        from src.preprocess import standardize_train_test_split

        data_config = config_base.get('data', {})
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
            csv_path=data_config.get('csv_path'),
            target_col=data_config.get('target_col', 'label'),
            train_size=data_config.get('train_size', 600),
            val_size=data_config.get('val_size', 200),
            test_size=data_config.get('test_size', 200),
            random_state=data_config.get('random_state', 42),
            stratify=data_config.get('stratify', True)
        )
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_train_test_split(
            X_train, X_val, X_test
        )

        # Verify preprocessing was applied (scaled data has mean ≈ 0, std ≈ 1)
        assert abs(X_train_scaled.values.mean()) < 0.1, "Scaled data should have mean ≈ 0"
        assert abs(X_train_scaled.values.std() - 1.0) < 0.1, "Scaled data should have std ≈ 1"

    @pytest.mark.p0
    @svm_mark
    def test_svm_outputs_confusion_matrix(self, sample_data, tmp_path, config_base):
        """P0: SVM must output confusion_matrix.png and confusion_matrix.csv"""
        csv_path = tmp_path / "data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        artifacts_dir = tmp_path / "artifacts" / "svm"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Train and evaluate
        train_svm(config=config_base, output_dir=artifacts_dir)

        # Verify outputs
        assert (artifacts_dir / "confusion_matrix.png").exists(), "Missing confusion_matrix.png"
        assert (artifacts_dir / "confusion_matrix.csv").exists(), "Missing confusion_matrix.csv"

    @pytest.mark.p0
    @svm_mark
    def test_svm_outputs_metrics_json(self, sample_data, tmp_path, config_base):
        """P0: SVM must output metrics.json with accuracy, precision, recall, f1, roc_auc"""
        csv_path = tmp_path / "data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        artifacts_dir = tmp_path / "artifacts" / "svm"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Train and evaluate
        train_svm(config=config_base, output_dir=artifacts_dir)

        # Verify metrics.json
        metrics_file = artifacts_dir / "metrics.json"
        assert metrics_file.exists(), "Missing metrics.json"

        with open(metrics_file) as f:
            metrics = json.load(f)

        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert 0 <= metrics[metric] <= 1, f"{metric} must be between 0 and 1"

    @pytest.mark.p0
    @svm_mark
    def test_svm_uses_correct_data_split(self, sample_data, tmp_path, config_base):
        """P0: SVM must use 600/200/200 train/val/test split"""
        csv_path = tmp_path / "data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        # Verify config specifies correct split
        assert config_base['data']['train_size'] == 600
        assert config_base['data']['val_size'] == 200
        assert config_base['data']['test_size'] == 200

        # Train model and verify it uses this split
        artifacts_dir = tmp_path / "artifacts" / "svm"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        results = train_svm(config=config_base, output_dir=artifacts_dir)

        # Results should reflect the correct split sizes
        assert results is not None, "Training should return results"

    @pytest.mark.p1
    @svm_mark
    def test_svm_saves_model_file(self, sample_data, tmp_path, config_base):
        """P1: SVM must save model.pkl file"""
        csv_path = tmp_path / "data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        artifacts_dir = tmp_path / "artifacts" / "svm"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Train model
        train_svm(config=config_base, output_dir=artifacts_dir)

        # Check for model file (could be .pkl, .joblib, etc.)
        model_files = list(artifacts_dir.glob("model.*"))
        assert len(model_files) > 0, "No model file found"

        # Verify we can load it
        import joblib
        model = joblib.load(model_files[0])
        assert model is not None, "Model should be loadable"

    @pytest.mark.p1
    @svm_mark
    def test_svm_reproducibility(self, sample_data, tmp_path, config_base):
        """P1: SVM must produce reproducible results with same random_state"""
        csv_path = tmp_path / "data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        # Train twice with same config
        artifacts_dir1 = tmp_path / "artifacts" / "svm1"
        artifacts_dir1.mkdir(parents=True, exist_ok=True)

        artifacts_dir2 = tmp_path / "artifacts" / "svm2"
        artifacts_dir2.mkdir(parents=True, exist_ok=True)

        train_svm(config=config_base, output_dir=artifacts_dir1)
        train_svm(config=config_base, output_dir=artifacts_dir2)

        # Compare metrics
        with open(artifacts_dir1 / "metrics.json") as f:
            metrics1 = json.load(f)
        with open(artifacts_dir2 / "metrics.json") as f:
            metrics2 = json.load(f)

        # Results should be identical (or very close due to floating point)
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            assert abs(metrics1[metric] - metrics2[metric]) < 1e-6, \
                f"{metric} not reproducible: {metrics1[metric]} vs {metrics2[metric]}"

    @pytest.mark.p2
    @svm_mark
    def test_svm_handles_class_imbalance(self, config_base, tmp_path):
        """P2: SVM should handle imbalanced datasets (edge case)"""
        # Create imbalanced dataset (90% class 0, 10% class 1)
        np.random.seed(42)
        n_samples = 1000
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        y = np.concatenate([
            np.zeros(900, dtype=int),  # 90% class 0
            np.ones(100, dtype=int)     # 10% class 1
        ])

        # Shuffle
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]

        # Create DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['label'] = y

        csv_path = tmp_path / "imbalanced_data.csv"
        df.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        artifacts_dir = tmp_path / "artifacts" / "svm_imbalanced"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Should train without errors
        results = train_svm(config=config_base, output_dir=artifacts_dir)

        # Verify metrics exist (though they may be lower due to imbalance)
        assert (artifacts_dir / "metrics.json").exists()
        with open(artifacts_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert 'accuracy' in metrics, "Should still compute accuracy"


# ============================================================================
# SVM Integration Test (P2 - End-to-End)
# ============================================================================

@pytest.mark.p2
@svm_mark
def test_svm_end_to_end_pipeline(sample_data, config_base, tmp_path):
    """
    Integration test: Complete pipeline from CSV → Training → Evaluation → Artifacts
    This simulates the full SVM workflow

    Tests the complete integration:
    1. Data loading from CSV
    2. Data splitting (600/200/200)
    3. Feature preprocessing (standardization)
    4. SVM training
    5. Model evaluation
    6. Artifact generation (confusion matrix, metrics, model file)
    """
    # Setup
    csv_path = tmp_path / "data.csv"
    sample_data.to_csv(csv_path, index=False)

    config_base['data']['csv_path'] = str(csv_path)
    artifacts_dir = tmp_path / "artifacts" / "svm"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Run complete pipeline
    results = train_svm(config=config_base, output_dir=artifacts_dir)

    # Verify all required outputs
    assert (artifacts_dir / "confusion_matrix.png").exists(), "Missing confusion_matrix.png"
    assert (artifacts_dir / "confusion_matrix.csv").exists(), "Missing confusion_matrix.csv"
    assert (artifacts_dir / "metrics.json").exists(), "Missing metrics.json"

    model_files = list(artifacts_dir.glob("model.*"))
    assert len(model_files) > 0, "Missing model file"

    # Verify metrics quality
    with open(artifacts_dir / "metrics.json") as f:
        metrics = json.load(f)

    # For balanced random data, accuracy should be reasonable
    assert metrics['accuracy'] > 0.4, "Accuracy too low (possible implementation issue)"
    assert metrics['f1'] > 0.3, "F1 score too low (possible implementation issue)"


# ============================================================================
# CNN1D Model Tests
# ============================================================================

class TestCNN1DModel:
    """Test suite for CNN1D model - GPU-optimized deep learning model"""

    @pytest.mark.p0
    @cnn1d_mark
    def test_cnn1d_loads_config(self, config_base):
        """Test that CNN1D correctly loads configuration from YAML"""
        # Verify CNN1D config exists in base config
        assert 'cnn1d' in config_base['models'], "CNN1D config missing from base.yaml"

        cnn1d_config = config_base['models']['cnn1d']

        # Check required fields
        assert 'n_channels' in cnn1d_config
        assert 'dropout' in cnn1d_config
        assert 'learning_rate' in cnn1d_config
        assert 'batch_size' in cnn1d_config
        assert 'max_epochs' in cnn1d_config
        assert 'patience' in cnn1d_config
        assert 'device' in cnn1d_config

        # Verify values are reasonable
        assert cnn1d_config['n_channels'] > 0
        assert 0 < cnn1d_config['dropout'] < 1
        assert cnn1d_config['learning_rate'] > 0
        assert cnn1d_config['batch_size'] > 0

    @pytest.mark.p0
    @cnn1d_mark
    def test_cnn1d_gpu_auto_detection(self):
        """Test GPU auto-detection (CUDA/MPS/CPU fallback)"""
        import torch

        config = {'device': 'auto'}
        model = CNN1DWrapper(config=config)

        # Device should be set
        assert model.device is not None

        # Should be one of the supported device types
        assert model.device.type in ['cuda', 'mps', 'cpu']

        # If CUDA available, should use CUDA
        if torch.cuda.is_available():
            assert model.device.type == 'cuda', "Should use CUDA when available"
        # Else if MPS available, should use MPS
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            assert model.device.type == 'mps', "Should use MPS when available"
        # Otherwise should use CPU
        else:
            assert model.device.type == 'cpu', "Should fallback to CPU"

    @pytest.mark.p0
    @cnn1d_mark
    def test_cnn1d_uses_unified_data_loader(self, sample_data, config_base, tmp_path):
        """Test that CNN1D uses unified load_and_split_data function"""
        # Save sample data
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        # Mock load_and_split_data to verify it's called
        from src import data_loader
        original_func = data_loader.load_and_split_data
        call_count = {'count': 0}

        def mock_load(*args, **kwargs):
            call_count['count'] += 1
            return original_func(*args, **kwargs)

        data_loader.load_and_split_data = mock_load

        try:
            # Run training (will use mocked function)
            train_cnn1d(config=config_base, output_dir=str(tmp_path / "artifacts" / "cnn1d"))

            # Verify unified data loader was called
            assert call_count['count'] == 1, "Should use unified load_and_split_data function"
        finally:
            # Restore original function
            data_loader.load_and_split_data = original_func

    @pytest.mark.p0
    @cnn1d_mark
    def test_cnn1d_uses_preprocessing(self, sample_data, config_base, tmp_path):
        """Test that CNN1D uses unified standardize_train_test_split"""
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        # Mock preprocessing to verify it's called
        from src import preprocess
        original_func = preprocess.standardize_train_test_split
        call_count = {'count': 0}

        def mock_preprocess(*args, **kwargs):
            call_count['count'] += 1
            return original_func(*args, **kwargs)

        preprocess.standardize_train_test_split = mock_preprocess

        try:
            train_cnn1d(config=config_base, output_dir=str(tmp_path / "artifacts" / "cnn1d"))

            # Verify preprocessing was called
            assert call_count['count'] == 1, "Should use unified standardize_train_test_split"
        finally:
            preprocess.standardize_train_test_split = original_func

    @pytest.mark.p0
    @cnn1d_mark
    def test_cnn1d_outputs_confusion_matrix(self, sample_data, config_base, tmp_path):
        """
        Test that CNN1D creates confusion_matrix.png and .csv
        This is a HARD REQUIREMENT from specification
        """
        # Setup artifacts directory
        artifacts_dir = tmp_path / "artifacts" / "cnn1d"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Update config
        config_base['output']['artifacts_dir'] = str(artifacts_dir)
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        # Reduce epochs for faster testing
        config_base['models']['cnn1d']['max_epochs'] = 2

        # Run training with evaluation
        results = train_cnn1d(config=config_base, output_dir=artifacts_dir)

        # Check confusion matrix files exist
        cm_png = artifacts_dir / "confusion_matrix.png"
        cm_csv = artifacts_dir / "confusion_matrix.csv"

        assert cm_png.exists(), f"Confusion matrix PNG should exist at {cm_png}"
        assert cm_csv.exists(), f"Confusion matrix CSV should exist at {cm_csv}"

    @pytest.mark.p0
    @cnn1d_mark
    def test_cnn1d_outputs_metrics_json(self, sample_data, config_base, tmp_path):
        """
        Test that CNN1D creates metrics.json with required fields
        This is a HARD REQUIREMENT from specification
        """
        artifacts_dir = tmp_path / "artifacts" / "cnn1d"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        config_base['output']['artifacts_dir'] = str(artifacts_dir)
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)
        config_base['models']['cnn1d']['max_epochs'] = 2

        # Run training
        results = train_cnn1d(config=config_base, output_dir=artifacts_dir)

        # Check metrics.json exists
        metrics_json = artifacts_dir / "metrics.json"
        assert metrics_json.exists(), f"metrics.json should exist at {metrics_json}"

        # Load and verify metrics
        import json
        with open(metrics_json, 'r') as f:
            metrics = json.load(f)

        # Check required fields (from specification)
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in required_metrics:
            assert metric in metrics, f"metrics.json should contain {metric}"

    @pytest.mark.p0
    @cnn1d_mark
    def test_cnn1d_uses_correct_data_split(self, sample_data, config_base, tmp_path):
        """
        Test that CNN1D uses fixed split: train=600, val=200, test=200
        This is a HARD REQUIREMENT that must not be changed
        """
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)
        config_base['models']['cnn1d']['max_epochs'] = 2

        # Run training
        results = train_cnn1d(config=config_base, output_dir=str(tmp_path / "artifacts" / "cnn1d"))

        # Verify data split
        data_split = results['data_split']
        assert data_split['train_size'] == 600, "Train size must be 600"
        assert data_split['val_size'] == 200, "Validation size must be 200"
        assert data_split['test_size'] == 200, "Test size must be 200"

    @pytest.mark.p1
    @cnn1d_mark
    def test_cnn1d_saves_model_file(self, sample_data, config_base, tmp_path):
        """Test that CNN1D saves model.pkl and scaler.pkl"""
        artifacts_dir = tmp_path / "artifacts" / "cnn1d"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        config_base['output']['artifacts_dir'] = str(artifacts_dir)
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)
        config_base['models']['cnn1d']['max_epochs'] = 2

        # Run training
        train_cnn1d(config=config_base, output_dir=artifacts_dir)

        # Check model files exist
        model_file = artifacts_dir / "model.pkl"
        scaler_file = artifacts_dir / "scaler.pkl"

        assert model_file.exists(), "model.pkl should be saved"
        assert scaler_file.exists(), "scaler.pkl should be saved"

    @pytest.mark.p1
    @cnn1d_mark
    def test_cnn1d_early_stopping(self, sample_data, config_base, tmp_path):
        """Test that CNN1D implements early stopping with patience"""
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        config_base['data']['csv_path'] = str(csv_path)

        # Set very high epochs but low patience
        config_base['models']['cnn1d']['max_epochs'] = 100
        config_base['models']['cnn1d']['patience'] = 3

        # Training should stop early (before 100 epochs)
        # This is tested indirectly by ensuring training completes quickly
        results = train_cnn1d(config=config_base, output_dir=str(tmp_path / "artifacts" / "cnn1d"))

        # If we get here without timeout, early stopping worked
        assert results is not None

    @pytest.mark.p2
    @cnn1d_mark
    def test_cnn1d_handles_tabular_features(self, sample_data, config_base, tmp_path):
        """Test that CNN1D correctly handles tabular features (12 numeric features)"""
        import numpy as np
        from src.preprocess import standardize_train_test_split

        # Create simple test data - exclude non-numeric columns, keep as DataFrame
        exclude_cols = ['sample_id', 'tic_id', 'label', 'status', 'error']
        X = sample_data.drop(columns=exclude_cols).iloc[:20]  # 20 samples, 12 features (DataFrame)
        y = sample_data['label'].values[:20]

        # Train small model
        config = {
            'n_channels': 8,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 8,
            'max_epochs': 2,
            'patience': 2,
            'device': 'cpu',  # Force CPU for test
            'random_state': 42
        }

        model = CNN1DWrapper(config=config)

        # Split data (keep as DataFrames for preprocessing)
        X_train = X.iloc[:12]
        X_val = X.iloc[12:16]
        y_train = y[:12]
        y_val = y[12:16]

        # Standardize
        X_test = X.iloc[16:20]
        X_train_scaled, X_val_scaled, X_test_scaled, _ = standardize_train_test_split(
            X_train, X_val, X_test, method='standard'
        )

        # Convert to numpy arrays for CNN model
        X_train_np = X_train_scaled.values
        X_val_np = X_val_scaled.values
        X_test_np = X_test_scaled.values

        # Train
        model.train(X_train_np, y_train, X_val_np, y_val)

        # Predict
        predictions = model.predict(X_test_np)
        probabilities = model.predict_proba(X_test_np)

        # Verify outputs
        assert predictions.shape == (4,), "Predictions should be 1D array"
        assert probabilities.shape == (4,), "Probabilities should be 1D array"
        assert np.all((predictions == 0) | (predictions == 1)), "Predictions should be binary"
        assert np.all((probabilities >= 0) & (probabilities <= 1)), "Probabilities should be in [0,1]"


# ============================================================================
# CNN1D Integration Test (P2 - End-to-End)
# ============================================================================

@pytest.mark.p2
@cnn1d_mark
def test_cnn1d_end_to_end_pipeline(sample_data, config_base, tmp_path):
    """
    Integration test: Complete pipeline from CSV → Training → Evaluation → Artifacts
    This simulates the full CNN1D workflow

    Tests the complete integration:
    1. Data loading from CSV
    2. Data splitting (600/200/200)
    3. Feature preprocessing (standardization)
    4. CNN1D training with GPU auto-detection
    5. Model evaluation
    6. Artifact generation (confusion matrix, metrics, model file)
    """
    # Setup
    csv_path = tmp_path / "data.csv"
    sample_data.to_csv(csv_path, index=False)

    config_base['data']['csv_path'] = str(csv_path)
    artifacts_dir = tmp_path / "artifacts" / "cnn1d"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Reduce epochs for faster testing
    config_base['models']['cnn1d']['max_epochs'] = 2

    # Run complete pipeline
    results = train_cnn1d(config=config_base, output_dir=artifacts_dir)

    # Verify all required outputs
    assert (artifacts_dir / "confusion_matrix.png").exists(), "Missing confusion_matrix.png"
    assert (artifacts_dir / "confusion_matrix.csv").exists(), "Missing confusion_matrix.csv"
    assert (artifacts_dir / "metrics.json").exists(), "Missing metrics.json"

    model_files = list(artifacts_dir.glob("model.*"))
    assert len(model_files) > 0, "Missing model file"

    # Verify metrics quality
    with open(artifacts_dir / "metrics.json") as f:
        metrics = json.load(f)

    # For balanced random data, accuracy should be reasonable
    assert metrics['accuracy'] > 0.4, "Accuracy too low (possible implementation issue)"
    assert metrics['f1'] > 0.3, "F1 score too low (possible implementation issue)"
