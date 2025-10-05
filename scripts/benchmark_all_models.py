"""
Unified Model Benchmarking and Comparison
==========================================
Runs all 6 models and generates comprehensive comparison report

Models:
- XGBoost
- Random Forest
- MLP (Multi-Layer Perceptron)
- Logistic Regression
- SVM (Support Vector Machine)
- CNN1D (1D Convolutional Neural Network)

Author: NASA Exoplanet ML
Date: 2025-10-05
"""

import json
import time
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import unified training functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.xgboost.train import train_xgboost
from src.models.random_forest.train import train_random_forest
from src.models.mlp.train import train_mlp
from src.models.logistic_regression.train import train_logistic_regression
from src.models.svm.train import train_svm
from src.models.cnn1d.train import train_cnn1d

# Configure plotting
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("UNIFIED MODEL BENCHMARKING - ALL 6 MODELS")
print("="*80)

# Configuration
CONFIG_PATH = Path('configs/base.yaml')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================
# TRAIN ALL MODELS
# ============================================

benchmark_results = {}

models = [
    ('XGBoost', train_xgboost),
    ('Random Forest', train_random_forest),
    ('MLP', train_mlp),
    ('Logistic Regression', train_logistic_regression),
    ('SVM', train_svm),
    ('CNN1D', train_cnn1d)
]

print("\n[TRAINING ALL MODELS]")
for model_name, train_func in models:
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        results = train_func(config=CONFIG_PATH)
        training_time = time.time() - start_time

        # Extract metrics from evaluate_model result
        # results['metrics'] is the evaluate_model return value
        # which itself has a 'metrics' key containing the actual metrics
        eval_result = results['metrics']
        metrics = eval_result['metrics']

        benchmark_results[model_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'training_time': training_time,
            'data_split': results['data_split']
        }

        print(f"\n[OK] {model_name} completed in {training_time:.2f}s")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")

    except Exception as e:
        print(f"\n[FAILED] {model_name} failed: {str(e)}")
        benchmark_results[model_name] = {
            'error': str(e),
            'training_time': time.time() - start_time
        }

print(f"\n{'='*80}")
print("ALL MODELS TRAINED")
print(f"{'='*80}")

# ============================================
# CREATE COMPARISON VISUALIZATIONS
# ============================================

print("\n[GENERATING VISUALIZATIONS]")

# Filter successful models
successful_models = {k: v for k, v in benchmark_results.items() if 'error' not in v}

if len(successful_models) == 0:
    print("[ERROR] No models trained successfully. Exiting.")
    sys.exit(1)

# Create DataFrame
df_comparison = pd.DataFrame(successful_models).T
df_comparison.index.name = 'Model'

# Visualization 1: Performance Metrics Comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Performance Comparison - All Metrics', fontsize=16, weight='bold')

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'training_time']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time (s)']
colors = plt.cm.Set3(np.linspace(0, 1, len(successful_models)))

for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    values = df_comparison[metric].values
    model_names = df_comparison.index

    bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel(name, fontsize=11, weight='bold')
    ax.set_title(f'{name} Comparison', fontsize=12, weight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')

    if metric != 'training_time':
        ax.set_ylim([0, 1.05])

    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        if metric == 'training_time':
            label = f'{val:.2f}s'
        else:
            label = f'{val:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                label, ha='center', va='bottom', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'benchmark_all_metrics.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: benchmark_all_metrics.png")
plt.close()

# Visualization 2: Ranking Table
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Create ranking by ROC-AUC
df_ranked = df_comparison.sort_values('roc_auc', ascending=False)

table_data = []
for rank, (model_name, row) in enumerate(df_ranked.iterrows(), 1):
    table_data.append([
        rank,
        model_name,
        f"{row['accuracy']:.4f}",
        f"{row['precision']:.4f}",
        f"{row['recall']:.4f}",
        f"{row['f1']:.4f}",
        f"{row['roc_auc']:.4f}",
        f"{row['training_time']:.2f}s"
    ])

table = ax.table(
    cellText=table_data,
    colLabels=['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Time'],
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(8):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best performer
table[(1, 0)].set_facecolor('#2ecc71')
for i in range(1, 8):
    table[(1, i)].set_facecolor('#2ecc71')
    table[(1, i)].set_text_props(weight='bold')

# Color code other rows
row_colors = ['#ecf0f1', 'white']
for i in range(2, len(table_data) + 1):
    for j in range(8):
        table[(i, j)].set_facecolor(row_colors[i % 2])

ax.set_title('Model Performance Rankings (by ROC-AUC)', fontsize=14, weight='bold', pad=20)

plt.savefig(RESULTS_DIR / 'benchmark_ranking_table.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: benchmark_ranking_table.png")
plt.close()

# Visualization 3: Radar Chart
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='polar')

# Metrics for radar chart (exclude training_time)
radar_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
num_vars = len(radar_metrics)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Plot each model
for idx, (model_name, row) in enumerate(df_comparison.iterrows()):
    values = [row[metric] for metric in radar_metrics]
    values += values[:1]  # Complete the circle

    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

# Fix axis to go in the right order and start at 12 o'clock
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label
ax.set_xticks(angles[:-1])
ax.set_xticklabels([m.replace('_', ' ').title() for m in radar_metrics], size=11)

# Set y-limits and labels
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.set_title('Performance Metrics Radar Chart', size=14, weight='bold', pad=20)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'benchmark_radar_chart.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: benchmark_radar_chart.png")
plt.close()

# ============================================
# GENERATE MARKDOWN REPORT
# ============================================

print("\n[GENERATING MARKDOWN REPORT]")

report_path = RESULTS_DIR / 'benchmark_summary.md'

with open(report_path, 'w') as f:
    f.write("# Model Benchmarking Report\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## Overview\n\n")
    f.write(f"This report compares {len(successful_models)} machine learning models ")
    f.write("trained on the NASA Exoplanet dataset using a unified pipeline.\n\n")

    f.write("### Dataset Information\n\n")
    first_model = list(successful_models.values())[0]
    f.write(f"- **Total Samples:** {sum(first_model['data_split'].values())}\n")
    f.write(f"- **Training Set:** {first_model['data_split']['train_size']} samples\n")
    f.write(f"- **Validation Set:** {first_model['data_split']['val_size']} samples\n")
    f.write(f"- **Test Set:** {first_model['data_split']['test_size']} samples\n")
    f.write(f"- **Source:** `configs/base.yaml`\n\n")

    f.write("## Performance Rankings\n\n")
    f.write("Models ranked by ROC-AUC score:\n\n")
    f.write("| Rank | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Time (s) |\n")
    f.write("|------|-------|----------|-----------|--------|----|---------|---------|\n")

    for rank, (model_name, row) in enumerate(df_ranked.iterrows(), 1):
        medal = "[1st]" if rank == 1 else "[2nd]" if rank == 2 else "[3rd]" if rank == 3 else "     "
        f.write(f"| {rank} {medal} | **{model_name}** | "
                f"{row['accuracy']:.4f} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['f1']:.4f} | "
                f"{row['roc_auc']:.4f} | {row['training_time']:.2f} |\n")

    f.write("\n## Detailed Analysis\n\n")

    # Best performers
    best_acc = df_comparison['accuracy'].idxmax()
    best_auc = df_comparison['roc_auc'].idxmax()
    fastest = df_comparison['training_time'].idxmin()

    f.write("### Best Performers\n\n")
    f.write(f"- **Highest Accuracy:** {best_acc} ({df_comparison.loc[best_acc, 'accuracy']:.4f})\n")
    f.write(f"- **Highest ROC-AUC:** {best_auc} ({df_comparison.loc[best_auc, 'roc_auc']:.4f})\n")
    f.write(f"- **Fastest Training:** {fastest} ({df_comparison.loc[fastest, 'training_time']:.2f}s)\n\n")

    f.write("### Model Comparison Summary\n\n")

    for model_name, row in df_comparison.iterrows():
        f.write(f"#### {model_name}\n\n")
        f.write(f"- **Accuracy:** {row['accuracy']:.4f}\n")
        f.write(f"- **Precision:** {row['precision']:.4f}\n")
        f.write(f"- **Recall:** {row['recall']:.4f}\n")
        f.write(f"- **F1-Score:** {row['f1']:.4f}\n")
        f.write(f"- **ROC-AUC:** {row['roc_auc']:.4f}\n")
        f.write(f"- **Training Time:** {row['training_time']:.2f} seconds\n\n")

    f.write("## Visualizations\n\n")
    f.write("![All Metrics Comparison](benchmark_all_metrics.png)\n\n")
    f.write("![Ranking Table](benchmark_ranking_table.png)\n\n")
    f.write("![Radar Chart](benchmark_radar_chart.png)\n\n")

    f.write("## Key Findings\n\n")

    # Calculate statistics
    avg_acc = df_comparison['accuracy'].mean()
    avg_auc = df_comparison['roc_auc'].mean()
    total_time = df_comparison['training_time'].sum()

    f.write(f"- Average Accuracy across all models: **{avg_acc:.4f}**\n")
    f.write(f"- Average ROC-AUC across all models: **{avg_auc:.4f}**\n")
    f.write(f"- Total training time for all models: **{total_time:.2f} seconds**\n")
    f.write(f"- Performance variance (ROC-AUC std): **{df_comparison['roc_auc'].std():.4f}**\n\n")

    f.write("## Recommendations\n\n")

    if best_auc == fastest:
        f.write(f"- **{best_auc}** achieves both best performance and fastest training - "
                f"recommended for production use.\n")
    else:
        f.write(f"- For **accuracy-critical applications**: Use **{best_auc}** "
                f"(ROC-AUC: {df_comparison.loc[best_auc, 'roc_auc']:.4f})\n")
        f.write(f"- For **speed-critical applications**: Use **{fastest}** "
                f"(Training time: {df_comparison.loc[fastest, 'training_time']:.2f}s)\n")

    f.write("\n## Artifacts\n\n")
    f.write("All model artifacts saved to:\n\n")
    for model_name in successful_models.keys():
        model_dir = model_name.lower().replace(' ', '_')
        f.write(f"- `artifacts/{model_dir}/`\n")
        f.write(f"  - `model.pkl` - Trained model\n")
        f.write(f"  - `confusion_matrix.png` - Confusion matrix visualization\n")
        f.write(f"  - `confusion_matrix.csv` - Confusion matrix data\n")
        f.write(f"  - `metrics.json` - Performance metrics\n\n")

print(f"[OK] Saved: {report_path}")

# ============================================
# SAVE JSON RESULTS
# ============================================

print("\n[SAVING JSON RESULTS]")

json_results = {
    'timestamp': datetime.now().isoformat(),
    'config': str(CONFIG_PATH),
    'models': benchmark_results,
    'summary': {
        'total_models': len(successful_models),
        'failed_models': len(benchmark_results) - len(successful_models),
        'best_accuracy': best_acc,
        'best_roc_auc': best_auc,
        'fastest_model': fastest,
        'average_accuracy': float(avg_acc),
        'average_roc_auc': float(avg_auc),
        'total_training_time': float(total_time)
    }
}

json_path = RESULTS_DIR / 'benchmark_results.json'
with open(json_path, 'w') as f:
    json.dump(json_results, f, indent=2)

print(f"[OK] Saved: {json_path}")

# ============================================
# PRINT SUMMARY
# ============================================

print("\n" + "="*80)
print("BENCHMARKING COMPLETE!")
print("="*80)

print(f"\n[SUMMARY]:")
print(f"  - Models Trained: {len(successful_models)}/{len(benchmark_results)}")
print(f"  - Best ROC-AUC: {best_auc} ({df_comparison.loc[best_auc, 'roc_auc']:.4f})")
print(f"  - Best Accuracy: {best_acc} ({df_comparison.loc[best_acc, 'accuracy']:.4f})")
print(f"  - Fastest: {fastest} ({df_comparison.loc[fastest, 'training_time']:.2f}s)")
print(f"  - Total Time: {total_time:.2f}s")

print(f"\n[OUTPUT FILES]:")
print(f"  - Report: {report_path}")
print(f"  - JSON: {json_path}")
print(f"  - Visualizations: {RESULTS_DIR}/*.png")

print("\n" + "="*80)
