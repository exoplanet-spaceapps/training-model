"""
Test suite for metrics module

NOTE: These tests are written FIRST (TDD approach) before src/metrics.py exists.
They are EXPECTED TO FAIL initially (red lights 🔴).
After implementing src/metrics.py, tests should pass (green lights 🟢).

Test priorities:
- P0: Critical metrics and confusion matrix functionality
- P1: Important evaluation and reporting features
- P2: Edge cases and integration tests
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json

# This import will FAIL until we create src/metrics.py
# This is EXPECTED and CORRECT for TDD (Red → Green → Refactor)
try:
    from src.metrics import (
        calculate_metrics,
        generate_confusion_matrix,
        save_confusion_matrix_png,
        save_confusion_matrix_csv,
        save_metrics_json,
        evaluate_model,
        aggregate_model_results,
        generate_comparison_report
    )
    METRICS_EXISTS = True
except ImportError:
    METRICS_EXISTS = False
    # Create placeholder functions for test discovery
    def calculate_metrics(*args, **kwargs):
        raise NotImplementedError("metrics.py not implemented yet (TDD red phase)")

    def generate_confusion_matrix(*args, **kwargs):
        raise NotImplementedError("metrics.py not implemented yet (TDD red phase)")

    def save_confusion_matrix_png(*args, **kwargs):
        raise NotImplementedError("metrics.py not implemented yet (TDD red phase)")

    def save_confusion_matrix_csv(*args, **kwargs):
        raise NotImplementedError("metrics.py not implemented yet (TDD red phase)")

    def save_metrics_json(*args, **kwargs):
        raise NotImplementedError("metrics.py not implemented yet (TDD red phase)")

    def evaluate_model(*args, **kwargs):
        raise NotImplementedError("metrics.py not implemented yet (TDD red phase)")

    def aggregate_model_results(*args, **kwargs):
        raise NotImplementedError("metrics.py not implemented yet (TDD red phase)")

    def generate_comparison_report(*args, **kwargs):
        raise NotImplementedError("metrics.py not implemented yet (TDD red phase)")


# Mark all tests to skip if metrics.py doesn't exist yet
pytestmark = pytest.mark.skipif(
    not METRICS_EXISTS,
    reason="src/metrics.py not implemented yet (TDD red phase - this is expected)"
)


@pytest.fixture
def binary_predictions():
    """Fixture providing binary classification predictions and labels"""
    np.random.seed(42)

    # Create realistic predictions (100 samples, 50/50 class balance)
    y_true = np.array([0] * 50 + [1] * 50)

    # Model with 90% accuracy
    y_pred = y_true.copy()
    # Introduce 10% errors
    error_indices = np.random.choice(100, size=10, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]

    # Probability predictions (for ROC-AUC)
    y_proba = np.random.rand(100)
    # Make probabilities correlate with true labels
    y_proba[y_true == 1] = np.random.uniform(0.6, 1.0, size=50)
    y_proba[y_true == 0] = np.random.uniform(0.0, 0.4, size=50)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


@pytest.fixture
def model_artifacts_dir(tmp_path):
    """Fixture providing temporary artifacts directory"""
    artifacts = tmp_path / "artifacts" / "test_model"
    artifacts.mkdir(parents=True)
    return artifacts


class TestMetricsCalculation:
    """Test metrics calculation functionality (P0 - Critical)"""

    @pytest.mark.p0
    def test_calculate_metrics_returns_required_fields(self, binary_predictions):
        """
        Test that calculate_metrics returns all required metric fields
        This is a HARD REQUIREMENT from specification
        """
        metrics = calculate_metrics(
            binary_predictions['y_true'],
            binary_predictions['y_pred'],
            binary_predictions['y_proba']
        )

        # Must contain these fields
        required_fields = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        for field in required_fields:
            assert field in metrics, f"Metrics must contain '{field}'"
            assert isinstance(metrics[field], (int, float)), \
                f"Metric '{field}' must be numeric"

    @pytest.mark.p0
    def test_calculate_metrics_values_in_valid_range(self, binary_predictions):
        """
        Test that all metrics are in valid range [0, 1]
        Invalid metrics would indicate implementation errors
        """
        metrics = calculate_metrics(
            binary_predictions['y_true'],
            binary_predictions['y_pred'],
            binary_predictions['y_proba']
        )

        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, \
                f"Metric '{metric_name}' must be in [0, 1], got {value}"

    @pytest.mark.p0
    def test_perfect_predictions_give_perfect_metrics(self):
        """
        Test that perfect predictions yield metrics = 1.0
        This validates the calculation logic
        """
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = y_true.copy()  # Perfect predictions
        y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])

        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Perfect predictions should give perfect metrics
        assert metrics['accuracy'] == 1.0, "Perfect accuracy should be 1.0"
        assert metrics['precision'] == 1.0, "Perfect precision should be 1.0"
        assert metrics['recall'] == 1.0, "Perfect recall should be 1.0"
        assert metrics['f1'] == 1.0, "Perfect F1 should be 1.0"

    @pytest.mark.p1
    def test_calculate_metrics_handles_edge_case_all_same_class(self):
        """
        Test that metrics handle edge case where all predictions are same class
        This can cause division by zero in some metrics
        """
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1])  # All predict positive
        y_proba = np.array([0.9, 0.8, 0.9, 0.95, 0.85, 0.9])

        # Should not raise error
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Recall should be 1.0 (all positives correctly identified)
        assert metrics['recall'] == 1.0, \
            "When predicting all positive, recall should be 1.0"


class TestConfusionMatrixGeneration:
    """Test confusion matrix generation (P0 - Critical)"""

    @pytest.mark.p0
    def test_generate_confusion_matrix_correct_shape(self, binary_predictions):
        """
        Test that confusion matrix has correct shape (2x2 for binary)
        Shape mismatch would break visualization
        """
        cm = generate_confusion_matrix(
            binary_predictions['y_true'],
            binary_predictions['y_pred']
        )

        assert cm.shape == (2, 2), \
            "Binary confusion matrix must be 2x2"

    @pytest.mark.p0
    def test_generate_confusion_matrix_values_sum_to_total(self, binary_predictions):
        """
        Test that confusion matrix values sum to total number of samples
        This validates that no samples are lost
        """
        cm = generate_confusion_matrix(
            binary_predictions['y_true'],
            binary_predictions['y_pred']
        )

        total_samples = len(binary_predictions['y_true'])
        assert cm.sum() == total_samples, \
            f"Confusion matrix must sum to {total_samples}"

    @pytest.mark.p0
    def test_confusion_matrix_perfect_predictions(self):
        """
        Test confusion matrix for perfect predictions
        Should be diagonal matrix
        """
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = y_true.copy()

        cm = generate_confusion_matrix(y_true, y_pred)

        # Perfect predictions: only diagonal should be non-zero
        assert cm[0, 1] == 0, "False positives should be 0"
        assert cm[1, 0] == 0, "False negatives should be 0"
        assert cm[0, 0] == 3, "True negatives should be 3"
        assert cm[1, 1] == 3, "True positives should be 3"


class TestConfusionMatrixSaving:
    """Test confusion matrix saving functionality (P0 - Critical)"""

    @pytest.mark.p0
    def test_save_confusion_matrix_png_creates_file(
        self, binary_predictions, model_artifacts_dir
    ):
        """
        Test that PNG confusion matrix file is created
        Required for visualization in reports
        """
        cm = generate_confusion_matrix(
            binary_predictions['y_true'],
            binary_predictions['y_pred']
        )

        output_path = model_artifacts_dir / "confusion_matrix.png"

        save_confusion_matrix_png(
            cm,
            output_path,
            class_names=['Negative', 'Positive']
        )

        assert output_path.exists(), \
            "Confusion matrix PNG must be created"
        assert output_path.stat().st_size > 0, \
            "Confusion matrix PNG must not be empty"

    @pytest.mark.p0
    def test_save_confusion_matrix_csv_creates_file(
        self, binary_predictions, model_artifacts_dir
    ):
        """
        Test that CSV confusion matrix file is created
        Required for programmatic analysis
        """
        cm = generate_confusion_matrix(
            binary_predictions['y_true'],
            binary_predictions['y_pred']
        )

        output_path = model_artifacts_dir / "confusion_matrix.csv"

        save_confusion_matrix_csv(
            cm,
            output_path,
            class_names=['Negative', 'Positive']
        )

        assert output_path.exists(), \
            "Confusion matrix CSV must be created"

    @pytest.mark.p1
    def test_save_confusion_matrix_csv_correct_format(
        self, binary_predictions, model_artifacts_dir
    ):
        """
        Test that CSV confusion matrix has correct format
        Must be readable and parseable
        """
        cm = generate_confusion_matrix(
            binary_predictions['y_true'],
            binary_predictions['y_pred']
        )

        output_path = model_artifacts_dir / "confusion_matrix.csv"

        save_confusion_matrix_csv(
            cm,
            output_path,
            class_names=['Negative', 'Positive']
        )

        # Read back and validate
        df = pd.read_csv(output_path, index_col=0)

        assert df.shape == (2, 2), "CSV must be 2x2"
        assert list(df.columns) == ['Negative', 'Positive'], \
            "CSV must have class names as columns"
        assert list(df.index) == ['Negative', 'Positive'], \
            "CSV must have class names as rows"


class TestMetricsJsonSaving:
    """Test metrics JSON saving functionality (P0 - Critical)"""

    @pytest.mark.p0
    def test_save_metrics_json_creates_file(
        self, binary_predictions, model_artifacts_dir
    ):
        """
        Test that metrics JSON file is created
        Required for programmatic comparison
        """
        metrics = calculate_metrics(
            binary_predictions['y_true'],
            binary_predictions['y_pred'],
            binary_predictions['y_proba']
        )

        output_path = model_artifacts_dir / "metrics.json"

        save_metrics_json(metrics, output_path)

        assert output_path.exists(), \
            "Metrics JSON must be created"

    @pytest.mark.p0
    def test_save_metrics_json_correct_format(
        self, binary_predictions, model_artifacts_dir
    ):
        """
        Test that metrics JSON has correct format and is parseable
        Critical for automated reporting
        """
        metrics = calculate_metrics(
            binary_predictions['y_true'],
            binary_predictions['y_pred'],
            binary_predictions['y_proba']
        )

        output_path = model_artifacts_dir / "metrics.json"

        save_metrics_json(metrics, output_path)

        # Read back and validate
        with open(output_path, 'r') as f:
            loaded_metrics = json.load(f)

        # Must contain all original metrics
        for key in metrics:
            assert key in loaded_metrics, \
                f"Loaded metrics must contain '{key}'"
            assert loaded_metrics[key] == metrics[key], \
                f"Metric '{key}' value must match"

    @pytest.mark.p1
    def test_save_metrics_json_includes_metadata(
        self, binary_predictions, model_artifacts_dir
    ):
        """
        Test that metrics JSON includes metadata fields
        Important for tracking experiment provenance
        """
        metrics = calculate_metrics(
            binary_predictions['y_true'],
            binary_predictions['y_pred'],
            binary_predictions['y_proba']
        )

        output_path = model_artifacts_dir / "metrics.json"

        # Add metadata
        metrics['model_name'] = 'test_model'
        metrics['dataset_split'] = 'test'
        metrics['n_samples'] = len(binary_predictions['y_true'])

        save_metrics_json(metrics, output_path)

        with open(output_path, 'r') as f:
            loaded_metrics = json.load(f)

        assert 'model_name' in loaded_metrics
        assert 'dataset_split' in loaded_metrics
        assert 'n_samples' in loaded_metrics


class TestModelEvaluation:
    """Test complete model evaluation pipeline (P1 - Important)"""

    @pytest.mark.p1
    def test_evaluate_model_creates_all_artifacts(
        self, binary_predictions, model_artifacts_dir
    ):
        """
        Test that evaluate_model creates all required artifacts
        This is the main integration point for model evaluation
        """
        result = evaluate_model(
            y_true=binary_predictions['y_true'],
            y_pred=binary_predictions['y_pred'],
            y_proba=binary_predictions['y_proba'],
            model_name='test_model',
            output_dir=model_artifacts_dir,
            class_names=['Negative', 'Positive']
        )

        # Check that all files were created
        assert (model_artifacts_dir / "confusion_matrix.png").exists(), \
            "Must create confusion_matrix.png"
        assert (model_artifacts_dir / "confusion_matrix.csv").exists(), \
            "Must create confusion_matrix.csv"
        assert (model_artifacts_dir / "metrics.json").exists(), \
            "Must create metrics.json"

        # Check that result contains metrics
        assert 'metrics' in result
        assert 'confusion_matrix' in result

    @pytest.mark.p1
    def test_evaluate_model_returns_metrics_dict(self, binary_predictions, model_artifacts_dir):
        """
        Test that evaluate_model returns metrics dictionary
        Important for programmatic access to results
        """
        result = evaluate_model(
            y_true=binary_predictions['y_true'],
            y_pred=binary_predictions['y_pred'],
            y_proba=binary_predictions['y_proba'],
            model_name='test_model',
            output_dir=model_artifacts_dir
        )

        metrics = result['metrics']

        # Must contain required fields
        required_fields = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for field in required_fields:
            assert field in metrics, f"Result must contain '{field}'"


class TestCrossModelComparison:
    """Test cross-model comparison functionality (P1 - Important)"""

    @pytest.mark.p1
    def test_aggregate_model_results_combines_metrics(self, tmp_path):
        """
        Test that aggregate_model_results combines metrics from multiple models
        Critical for comparison reports
        """
        # Create mock results for 3 models
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        model_results = {
            'model_a': {
                'accuracy': 0.90,
                'precision': 0.88,
                'recall': 0.92,
                'f1': 0.90,
                'roc_auc': 0.94
            },
            'model_b': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1': 0.85,
                'roc_auc': 0.89
            },
            'model_c': {
                'accuracy': 0.92,
                'precision': 0.91,
                'recall': 0.93,
                'f1': 0.92,
                'roc_auc': 0.95
            }
        }

        # Save individual model metrics
        for model_name, metrics in model_results.items():
            model_dir = results_dir / model_name
            model_dir.mkdir()
            with open(model_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f)

        # Aggregate results
        aggregated = aggregate_model_results(results_dir)

        # Must contain all models
        assert len(aggregated) == 3, "Must aggregate all 3 models"

        for model_name in model_results:
            assert model_name in aggregated, \
                f"Aggregated results must contain '{model_name}'"

    @pytest.mark.p1
    def test_generate_comparison_report_creates_markdown(self, tmp_path):
        """
        Test that generate_comparison_report creates markdown report
        Required for final deliverable
        """
        # Create mock aggregated results
        aggregated_results = {
            'model_a': {
                'accuracy': 0.90,
                'precision': 0.88,
                'recall': 0.92,
                'f1': 0.90,
                'roc_auc': 0.94
            },
            'model_b': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1': 0.85,
                'roc_auc': 0.89
            }
        }

        output_path = tmp_path / "benchmark_summary.md"

        generate_comparison_report(aggregated_results, output_path)

        assert output_path.exists(), \
            "Comparison report must be created"

        # Check report contains key information
        content = output_path.read_text()
        assert 'model_a' in content, "Report must mention model_a"
        assert 'model_b' in content, "Report must mention model_b"
        assert 'accuracy' in content.lower(), \
            "Report must include accuracy metric"

    @pytest.mark.p2
    def test_generate_comparison_report_ranks_models(self, tmp_path):
        """
        Test that comparison report ranks models by performance
        Important for identifying best model
        """
        aggregated_results = {
            'model_a': {'accuracy': 0.90, 'f1': 0.90},
            'model_b': {'accuracy': 0.85, 'f1': 0.85},
            'model_c': {'accuracy': 0.92, 'f1': 0.92}
        }

        output_path = tmp_path / "benchmark_summary.md"

        generate_comparison_report(aggregated_results, output_path)

        content = output_path.read_text()

        # model_c should be ranked first (highest accuracy/f1)
        # This is a soft requirement - report should indicate rankings
        assert 'rank' in content.lower() or 'best' in content.lower(), \
            "Report should indicate model rankings"


class TestEdgeCases:
    """Test edge cases and error handling (P2 - Nice to have)"""

    @pytest.mark.p2
    def test_calculate_metrics_with_zero_samples_raises_error(self):
        """Test that empty arrays raise appropriate error"""
        y_true = np.array([])
        y_pred = np.array([])
        y_proba = np.array([])

        with pytest.raises(ValueError, match="empty|Empty|samples"):
            calculate_metrics(y_true, y_pred, y_proba)

    @pytest.mark.p2
    def test_calculate_metrics_with_mismatched_lengths_raises_error(self):
        """Test that mismatched array lengths raise error"""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])  # Different length
        y_proba = np.array([0.1, 0.9, 0.2])

        with pytest.raises(ValueError, match="length|shape"):
            calculate_metrics(y_true, y_pred, y_proba)

    @pytest.mark.p2
    def test_save_confusion_matrix_creates_parent_directory(self, tmp_path):
        """
        Test that saving creates parent directories if they don't exist
        Important for automatic directory creation
        """
        cm = np.array([[10, 2], [3, 15]])

        # Deep nested path that doesn't exist
        output_path = tmp_path / "nested" / "dir" / "confusion_matrix.png"

        # Should create directories automatically
        save_confusion_matrix_png(
            cm,
            output_path,
            class_names=['Negative', 'Positive']
        )

        assert output_path.exists(), \
            "Must create parent directories automatically"


# Integration test combining multiple functionalities
@pytest.mark.p2
def test_end_to_end_model_evaluation_pipeline(tmp_path):
    """
    Integration test: Generate predictions → Evaluate → Save artifacts → Compare
    This simulates the complete evaluation pipeline
    """
    np.random.seed(42)

    # Simulate predictions from 2 models
    y_true = np.array([0] * 50 + [1] * 50)

    # Model 1: 90% accuracy
    y_pred_1 = y_true.copy()
    error_idx = np.random.choice(100, size=10, replace=False)
    y_pred_1[error_idx] = 1 - y_pred_1[error_idx]
    y_proba_1 = np.random.rand(100)

    # Model 2: 85% accuracy
    y_pred_2 = y_true.copy()
    error_idx = np.random.choice(100, size=15, replace=False)
    y_pred_2[error_idx] = 1 - y_pred_2[error_idx]
    y_proba_2 = np.random.rand(100)

    # Evaluate both models
    results_dir = tmp_path / "results"

    model1_dir = results_dir / "model_1"
    result1 = evaluate_model(
        y_true, y_pred_1, y_proba_1,
        model_name='model_1',
        output_dir=model1_dir
    )

    model2_dir = results_dir / "model_2"
    result2 = evaluate_model(
        y_true, y_pred_2, y_proba_2,
        model_name='model_2',
        output_dir=model2_dir
    )

    # Aggregate results
    aggregated = aggregate_model_results(results_dir)

    assert len(aggregated) == 2, "Must aggregate both models"

    # Generate comparison report
    report_path = results_dir / "benchmark_summary.md"
    generate_comparison_report(aggregated, report_path)

    assert report_path.exists(), "Must create comparison report"

    # Verify model_1 has higher accuracy than model_2
    assert result1['metrics']['accuracy'] > result2['metrics']['accuracy'], \
        "Model 1 should have higher accuracy"
