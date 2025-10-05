"""
Metrics Module for NASA Exoplanet Detection

This module provides comprehensive model evaluation utilities including:
- Metrics calculation (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix generation and visualization
- Artifact saving (PNG, CSV, JSON)
- Cross-model comparison and reporting

Author: NASA Exoplanet ML Team
Date: 2025-01-05
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix as sklearn_confusion_matrix
)

# Set matplotlib to use non-interactive backend (prevents Tcl/Tk errors)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels (binary: 0 or 1)
        y_pred: Predicted labels (binary: 0 or 1)
        y_proba: Predicted probabilities for positive class (optional, for ROC-AUC)

    Returns:
        Dictionary containing:
        - accuracy: Overall correctness
        - precision: Positive predictive value
        - recall: Sensitivity / true positive rate
        - f1: Harmonic mean of precision and recall
        - roc_auc: Area under ROC curve (if y_proba provided)

    Raises:
        ValueError: If arrays are empty or have mismatched lengths

    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 0])
        >>> y_proba = np.array([0.1, 0.9, 0.8, 0.2])
        >>> metrics = calculate_metrics(y_true, y_pred, y_proba)
        >>> print(metrics['accuracy'])
        1.0
    """
    # Validation
    if len(y_true) == 0:
        raise ValueError("y_true cannot be empty")

    if len(y_pred) == 0:
        raise ValueError("y_pred cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples"
        )

    if y_proba is not None and len(y_true) != len(y_proba):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_proba has {len(y_proba)} samples"
        )

    # Calculate core metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }

    # Calculate ROC-AUC if probabilities provided
    if y_proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            # If only one class present, ROC-AUC is undefined
            metrics['roc_auc'] = 0.0

    return metrics


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Generate confusion matrix for binary classification.

    Args:
        y_true: True labels (binary: 0 or 1)
        y_pred: Predicted labels (binary: 0 or 1)

    Returns:
        2x2 numpy array:
        [[TN, FP],
         [FN, TP]]

    Raises:
        ValueError: If arrays are empty or have mismatched lengths

    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> cm = generate_confusion_matrix(y_true, y_pred)
        >>> print(cm)
        [[2 0]
         [1 1]]
    """
    # Validation
    if len(y_true) == 0:
        raise ValueError("y_true cannot be empty")

    if len(y_pred) == 0:
        raise ValueError("y_pred cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples"
        )

    # Generate confusion matrix
    cm = sklearn_confusion_matrix(y_true, y_pred)

    return cm


def save_confusion_matrix_png(
    cm: np.ndarray,
    output_path: Path,
    class_names: Optional[List[str]] = None,
    model_name: Optional[str] = None
) -> None:
    """
    Save confusion matrix as PNG visualization.

    Args:
        cm: Confusion matrix array (2x2)
        output_path: Path to save PNG file
        class_names: List of class names (default: ['Class 0', 'Class 1'])
        model_name: Model name for title (optional)

    Example:
        >>> cm = np.array([[45, 5], [3, 47]])
        >>> save_confusion_matrix_png(cm, Path('cm.png'), ['Negative', 'Positive'])
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Default class names
    if class_names is None:
        class_names = ['Class 0', 'Class 1']

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )

    # Labels and title
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    if model_name:
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, pad=20)
    else:
        plt.title('Confusion Matrix', fontsize=14, pad=20)

    # Add metrics to plot
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    metrics_text = f'Accuracy: {accuracy:.3f}\nTP: {tp}, TN: {tn}\nFP: {fp}, FN: {fn}'
    plt.text(
        0.02, 0.98, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_confusion_matrix_csv(
    cm: np.ndarray,
    output_path: Path,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Save confusion matrix as CSV file.

    Args:
        cm: Confusion matrix array (2x2)
        output_path: Path to save CSV file
        class_names: List of class names (default: ['Class 0', 'Class 1'])

    Example:
        >>> cm = np.array([[45, 5], [3, 47]])
        >>> save_confusion_matrix_csv(cm, Path('cm.csv'), ['Negative', 'Positive'])
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Default class names
    if class_names is None:
        class_names = ['Class 0', 'Class 1']

    # Create DataFrame
    df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    # Save to CSV with index (row labels)
    df.to_csv(output_path, index=True)


def save_metrics_json(
    metrics: Dict[str, float],
    output_path: Path,
    model_name: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save metrics dictionary as JSON file.

    Args:
        metrics: Dictionary of metric name -> value
        output_path: Path to save JSON file
        model_name: Model name to include in JSON (optional)
        additional_info: Additional metadata to include (optional)

    Example:
        >>> metrics = {'accuracy': 0.92, 'precision': 0.90, 'recall': 0.94, 'f1': 0.92}
        >>> save_metrics_json(metrics, Path('metrics.json'), model_name='CNN')
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build output dictionary - metrics at top level
    output = dict(metrics)

    if model_name:
        output['model_name'] = model_name

    if additional_info:
        output.update(additional_info)

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = 'model',
    output_dir: Path = Path('artifacts'),
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Complete model evaluation pipeline.

    Generates all evaluation artifacts:
    - Confusion matrix (PNG and CSV)
    - Metrics JSON

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
        model_name: Name of model for file naming
        output_dir: Directory to save artifacts (Path or str)
        class_names: List of class names for confusion matrix

    Returns:
        Dictionary containing:
        - metrics: Dictionary of calculated metrics
        - confusion_matrix: Confusion matrix array
        - artifacts: Dictionary of created file paths

    Example:
        >>> result = evaluate_model(
        ...     y_true=y_test,
        ...     y_pred=predictions,
        ...     y_proba=probabilities,
        ...     model_name='RandomForest',
        ...     output_dir=Path('artifacts/random_forest'),
        ...     class_names=['Non-Exoplanet', 'Exoplanet']
        ... )
        >>> print(result['metrics']['accuracy'])
        0.92
    """
    # Convert string to Path if needed
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)

    # Generate confusion matrix
    cm = generate_confusion_matrix(y_true, y_pred)

    # Save confusion matrix PNG
    cm_png_path = output_dir / 'confusion_matrix.png'
    save_confusion_matrix_png(cm, cm_png_path, class_names, model_name)

    # Save confusion matrix CSV
    cm_csv_path = output_dir / 'confusion_matrix.csv'
    save_confusion_matrix_csv(cm, cm_csv_path, class_names)

    # Save metrics JSON
    metrics_json_path = output_dir / 'metrics.json'
    save_metrics_json(metrics, metrics_json_path, model_name)

    # Build result
    result = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'artifacts': {
            'confusion_matrix_png': str(cm_png_path),
            'confusion_matrix_csv': str(cm_csv_path),
            'metrics_json': str(metrics_json_path)
        }
    }

    return result


def aggregate_model_results(
    results_dir: Path,
    model_dirs: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics from multiple model directories.

    Args:
        results_dir: Directory containing model subdirectories
        model_dirs: List of specific model directory names (optional)
                   If None, searches for all subdirectories

    Returns:
        Dictionary mapping model_name -> metrics

    Example:
        >>> aggregated = aggregate_model_results(Path('results'))
        >>> print(aggregated['random_forest']['accuracy'])
        0.92
    """
    aggregated = {}

    # If model_dirs not specified, find all subdirectories
    if model_dirs is None:
        if not results_dir.exists():
            return aggregated

        model_dirs = [
            d.name for d in results_dir.iterdir()
            if d.is_dir()
        ]

    # Load metrics from each model
    for model_name in model_dirs:
        model_dir = results_dir / model_name
        metrics_path = model_dir / 'metrics.json'

        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                data = json.load(f)

                # Extract metrics (handle both formats)
                if 'metrics' in data:
                    # Old nested format
                    aggregated[model_name] = data['metrics']
                else:
                    # New flattened format - filter out metadata keys
                    metrics_data = {
                        k: v for k, v in data.items()
                        if k not in ['model_name'] and isinstance(v, (int, float))
                    }
                    aggregated[model_name] = metrics_data

    return aggregated


def generate_comparison_report(
    aggregated_results: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = 'Model Comparison Report'
) -> None:
    """
    Generate markdown comparison report from aggregated results.

    Args:
        aggregated_results: Dictionary mapping model_name -> metrics
        output_path: Path to save markdown report
        title: Report title

    Example:
        >>> results = {
        ...     'random_forest': {'accuracy': 0.92, 'f1': 0.91},
        ...     'svm': {'accuracy': 0.89, 'f1': 0.88}
        ... }
        >>> generate_comparison_report(results, Path('report.md'))
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Start building markdown
    lines = [
        f'# {title}',
        '',
        f'**Generated**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        '## Summary',
        '',
        f'Total models evaluated: **{len(aggregated_results)}**',
        ''
    ]

    # Create comparison table
    if aggregated_results:
        # Get all metric names (use first model as reference)
        first_model = next(iter(aggregated_results.values()))
        metric_names = list(first_model.keys())

        # Table header
        lines.append('## Metrics Comparison')
        lines.append('')
        header = '| Model | ' + ' | '.join(metric_names) + ' |'
        separator = '|' + '|'.join(['---'] * (len(metric_names) + 1)) + '|'
        lines.append(header)
        lines.append(separator)

        # Table rows (sorted by accuracy descending)
        sorted_models = sorted(
            aggregated_results.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )

        for model_name, metrics in sorted_models:
            row = f'| **{model_name}** |'
            for metric_name in metric_names:
                value = metrics.get(metric_name, 0.0)
                row += f' {value:.4f} |'
            lines.append(row)

        lines.append('')

        # Find best model for each metric
        lines.append('## Best Models by Metric')
        lines.append('')

        for metric_name in metric_names:
            best_model = max(
                aggregated_results.items(),
                key=lambda x: x[1].get(metric_name, 0)
            )
            lines.append(
                f'- **{metric_name}**: {best_model[0]} '
                f'({best_model[1].get(metric_name, 0):.4f})'
            )

        lines.append('')

        # Overall best model (by accuracy)
        best_overall = max(
            aggregated_results.items(),
            key=lambda x: x[1].get('accuracy', 0)
        )
        lines.append('## Overall Best Model')
        lines.append('')
        lines.append(f'**{best_overall[0]}** with accuracy: {best_overall[1].get("accuracy", 0):.4f}')
        lines.append('')

        # Model details
        lines.append('## Model Details')
        lines.append('')

        for model_name, metrics in sorted_models:
            lines.append(f'### {model_name}')
            lines.append('')
            for metric_name, value in metrics.items():
                lines.append(f'- {metric_name}: {value:.4f}')
            lines.append('')

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    """
    Example usage and verification
    """
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Example: Generate sample predictions and evaluate
    print("=" * 60)
    print("Metrics Module Verification")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    y_true = np.array([0] * 50 + [1] * 50)

    # Model with 90% accuracy
    y_pred = y_true.copy()
    error_indices = np.random.choice(100, size=10, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]

    # Probabilities
    y_proba = np.random.rand(100)
    y_proba[y_true == 1] = np.random.uniform(0.6, 1.0, size=50)
    y_proba[y_true == 0] = np.random.uniform(0.0, 0.4, size=50)

    # Calculate metrics
    print("\n--- Metrics Calculation ---")
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Generate confusion matrix
    print("\n--- Confusion Matrix ---")
    cm = generate_confusion_matrix(y_true, y_pred)
    print(cm)

    # Evaluate model (creates all artifacts)
    print("\n--- Model Evaluation ---")
    output_dir = Path('artifacts/example_model')
    result = evaluate_model(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        model_name='ExampleModel',
        output_dir=output_dir,
        class_names=['Non-Exoplanet', 'Exoplanet']
    )

    print(f"✓ Created artifacts in: {output_dir}")
    print(f"  - Confusion matrix PNG: {result['artifacts']['confusion_matrix_png']}")
    print(f"  - Confusion matrix CSV: {result['artifacts']['confusion_matrix_csv']}")
    print(f"  - Metrics JSON: {result['artifacts']['metrics_json']}")

    print("\n✓ Metrics module verification complete!")
