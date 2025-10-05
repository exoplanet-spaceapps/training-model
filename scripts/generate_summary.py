#!/usr/bin/env python3
"""
Generate benchmark summary from model artifacts.
"""

import json
from datetime import datetime
from pathlib import Path


def generate_summary():
    """Generate benchmark summary markdown report."""

    # Models to check
    models = ["xgboost", "random_forest", "mlp", "logistic_regression", "svm"]
    results = []

    for model in models:
        metrics_path = Path(f"artifacts/{model}/metrics.json")
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    data = json.load(f)
                    results.append({
                        'model': model,
                        'accuracy': data.get('test', {}).get('accuracy', 'N/A'),
                        'precision': data.get('test', {}).get('precision', 'N/A'),
                        'recall': data.get('test', {}).get('recall', 'N/A'),
                        'f1': data.get('test', {}).get('f1', 'N/A'),
                        'roc_auc': data.get('test', {}).get('roc_auc', 'N/A'),
                        'time': data.get('training_time_seconds', 'N/A')
                    })
            except Exception as e:
                print(f"Warning: Failed to load metrics for {model}: {e}")
                results.append({
                    'model': model,
                    'accuracy': 'ERROR',
                    'precision': 'ERROR',
                    'recall': 'ERROR',
                    'f1': 'ERROR',
                    'roc_auc': 'ERROR',
                    'time': 'ERROR'
                })
        else:
            results.append({
                'model': model,
                'accuracy': 'MISSING',
                'precision': 'MISSING',
                'recall': 'MISSING',
                'f1': 'MISSING',
                'roc_auc': 'MISSING',
                'time': 'MISSING'
            })

    # Generate markdown report
    report = f"""# Model Benchmark Summary

**Run Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data Split:** 600 train / 200 val / 200 test
**Source:** balanced_features.csv (1000 samples)

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Time (s) |
|-------|----------|-----------|--------|-----|---------|----------|
"""

    for r in results:
        model_name = r['model'].replace('_', ' ').title()
        acc = f"{r['accuracy']:.4f}" if isinstance(r['accuracy'], float) else r['accuracy']
        prec = f"{r['precision']:.4f}" if isinstance(r['precision'], float) else r['precision']
        rec = f"{r['recall']:.4f}" if isinstance(r['recall'], float) else r['recall']
        f1 = f"{r['f1']:.4f}" if isinstance(r['f1'], float) else r['f1']
        roc = f"{r['roc_auc']:.4f}" if isinstance(r['roc_auc'], float) else r['roc_auc']
        time_val = f"{r['time']:.2f}" if isinstance(r['time'], float) else r['time']

        report += f"| {model_name} | {acc} | {prec} | {rec} | {f1} | {roc} | {time_val} |\n"

    report += """
## Best Performing Models

"""

    # Find best models
    valid_results = [r for r in results if isinstance(r['accuracy'], float)]
    if valid_results:
        best_acc = max(valid_results, key=lambda x: x['accuracy'])
        report += f"- **Best Accuracy:** {best_acc['model'].replace('_', ' ').title()} ({best_acc['accuracy']:.4f})\n"

    valid_f1 = [r for r in results if isinstance(r['f1'], float)]
    if valid_f1:
        best_f1 = max(valid_f1, key=lambda x: x['f1'])
        report += f"- **Best F1 Score:** {best_f1['model'].replace('_', ' ').title()} ({best_f1['f1']:.4f})\n"

    valid_roc = [r for r in results if isinstance(r['roc_auc'], float)]
    if valid_roc:
        best_roc = max(valid_roc, key=lambda x: x['roc_auc'])
        report += f"- **Best ROC-AUC:** {best_roc['model'].replace('_', ' ').title()} ({best_roc['roc_auc']:.4f})\n"

    report += """
## Artifacts

- **Confusion Matrices:** `artifacts/{model}/confusion_matrix.png`
- **Model Files:** `artifacts/{model}/model.*`
- **Metrics:** `artifacts/{model}/metrics.json`

## Data Pipeline

All models use the unified pipeline:
1. Data loading from `balanced_features.csv`
2. Stratified split: 600/200/200 (train/val/test)
3. Standardization with StandardScaler
4. Cross-validation on training set
5. Evaluation on held-out test set

## Notes

- Random state: 42 (reproducible results)
- Stratification ensures balanced class distribution
- All metrics calculated on test set
- Models with missing/error status failed to complete training
"""

    # Save report
    Path("results").mkdir(exist_ok=True)
    output_path = Path("results/benchmark_summary.md")
    with open(output_path, "w") as f:
        f.write(report)

    print(f"[SUCCESS] Benchmark summary generated: {output_path}")

    # Print summary to console
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for r in results:
        status = "[OK]" if isinstance(r['accuracy'], float) else "[FAIL]"
        acc_str = f"{r['accuracy']:.4f}" if isinstance(r['accuracy'], float) else r['accuracy']
        print(f"{status} {r['model']:20s} - Accuracy: {acc_str}")
    print("="*60)


if __name__ == "__main__":
    generate_summary()
