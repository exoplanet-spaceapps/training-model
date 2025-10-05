"""
Generate comprehensive final report with comparison charts, confusion matrices, and PDF
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
RESULTS_DIR = Path('results')
ARTIFACTS_DIR = Path('artifacts')
OUTPUT_DIR = Path('final_report')
OUTPUT_DIR.mkdir(exist_ok=True)

# Load benchmark results
with open(RESULTS_DIR / 'benchmark_results.json', 'r', encoding='utf-8') as f:
    benchmark_data = json.load(f)

models_data = benchmark_data['models']
model_names = list(models_data.keys())

print("[INFO] Generating Final Comparison Report...")
print(f"Models: {', '.join(model_names)}")

# ============================================================================
# 1. CREATE PERFORMANCE COMPARISON BAR CHARTS
# ============================================================================

def create_comparison_charts():
    """Create comprehensive performance comparison bar charts"""

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'roc_auc': 'ROC-AUC'
    }

    # Extract data
    data = {metric: [] for metric in metrics}
    for model_name in model_names:
        for metric in metrics:
            data[metric].append(models_data[model_name][metric])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('6-Model Performance Comparison - NASA Exoplanet Detection',
                 fontsize=16, fontweight='bold', y=0.995)

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(model_names)))

    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(range(len(model_names)), data[metric], color=colors, alpha=0.8, edgecolor='black')

        # Styling
        ax.set_xlabel('Models', fontweight='bold', fontsize=11)
        ax.set_ylabel(metric_labels[metric], fontweight='bold', fontsize=11)
        ax.set_title(f'{metric_labels[metric]} Comparison', fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, data[metric])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Highlight best performer
        best_idx = np.argmax(data[metric])
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)

    # Training time comparison (6th subplot)
    ax = axes[1, 2]
    times = [models_data[model_name]['training_time'] for model_name in model_names]
    bars = ax.bar(range(len(model_names)), times, color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Models', fontweight='bold', fontsize=11)
    ax.set_ylabel('Training Time (seconds)', fontweight='bold', fontsize=11)
    ax.set_title('Training Time Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2f}s',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Highlight fastest
    fastest_idx = np.argmin(times)
    bars[fastest_idx].set_edgecolor('green')
    bars[fastest_idx].set_linewidth(3)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'model_comparison_charts.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved comparison charts: {output_path}")
    plt.close()

    return output_path

# ============================================================================
# 2. CREATE COMBINED CONFUSION MATRICES GRID
# ============================================================================

def create_confusion_matrices_grid():
    """Create a grid showing all 6 confusion matrices"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confusion Matrices - All 6 Models (Test Set: 200 samples)',
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, model_name in enumerate(model_names):
        ax = axes[idx // 3, idx % 3]

        # Load confusion matrix CSV
        cm_path = ARTIFACTS_DIR / model_name.lower().replace(' ', '_') / 'confusion_matrix.csv'

        if cm_path.exists():
            cm_df = pd.read_csv(cm_path, index_col=0)
            cm = cm_df.values

            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       ax=ax, cbar=True, square=True,
                       xticklabels=['No Exoplanet', 'Exoplanet'],
                       yticklabels=['No Exoplanet', 'Exoplanet'],
                       annot_kws={'size': 14, 'weight': 'bold'})

            # Styling
            ax.set_title(f'{model_name}\n(Accuracy: {models_data[model_name]["accuracy"]:.4f})',
                        fontweight='bold', fontsize=11)
            ax.set_xlabel('Predicted', fontweight='bold', fontsize=10)
            ax.set_ylabel('Actual', fontweight='bold', fontsize=10)

            # Add metrics annotation
            metrics_text = (f"Precision: {models_data[model_name]['precision']:.4f}\n"
                          f"Recall: {models_data[model_name]['recall']:.4f}\n"
                          f"F1-Score: {models_data[model_name]['f1']:.4f}")
            ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, f'No data for\n{model_name}',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'all_confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved confusion matrices grid: {output_path}")
    plt.close()

    return output_path

# ============================================================================
# 3. CREATE COMPREHENSIVE RESULTS TABLE
# ============================================================================

def create_results_table():
    """Create detailed results table as image and CSV"""

    # Prepare data
    table_data = []
    for rank, model_name in enumerate(sorted(model_names,
                                             key=lambda x: models_data[x]['roc_auc'],
                                             reverse=True), 1):
        model = models_data[model_name]

        # Add medal emojis
        medal = ''
        if rank == 1:
            medal = '🥇'
        elif rank == 2:
            medal = '🥈'
        elif rank == 3:
            medal = '🥉'

        table_data.append({
            'Rank': f"{rank} {medal}",
            'Model': model_name,
            'Accuracy': f"{model['accuracy']:.4f}",
            'Precision': f"{model['precision']:.4f}",
            'Recall': f"{model['recall']:.4f}",
            'F1-Score': f"{model['f1']:.4f}",
            'ROC-AUC': f"{model['roc_auc']:.4f}",
            'Time (s)': f"{model['training_time']:.2f}"
        })

    df = pd.DataFrame(table_data)

    # Save as CSV
    csv_path = OUTPUT_DIR / 'model_comparison_table.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved table CSV: {csv_path}")

    # Create table image
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.08, 0.18, 0.11, 0.11, 0.09, 0.11, 0.11, 0.10])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

    # Style rows
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')

            # Highlight top 3 in rank column
            if j == 0 and i <= 3:
                table[(i, j)].set_facecolor('#FFD700' if i == 1 else '#C0C0C0' if i == 2 else '#CD7F32')

    plt.title('Model Performance Ranking Table\nNASA Exoplanet Detection - Test Set Results',
             fontsize=14, fontweight='bold', pad=20)

    # Add summary footer
    summary_text = f"Dataset: 1000 samples | Train: 600 | Validation: 200 | Test: 200 | Total Time: {benchmark_data['summary']['total_training_time']:.2f}s"
    plt.figtext(0.5, 0.02, summary_text, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save
    img_path = OUTPUT_DIR / 'model_comparison_table.png'
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved table image: {img_path}")
    plt.close()

    return df, csv_path, img_path

# ============================================================================
# 4. CREATE COMPREHENSIVE PDF REPORT
# ============================================================================

def create_pdf_report(comparison_chart_path, confusion_matrices_path, table_img_path, table_df):
    """Create comprehensive PDF report"""

    pdf_path = OUTPUT_DIR / 'final_model_comparison_report.pdf'

    with PdfPages(pdf_path) as pdf:
        # PAGE 1: Title and Summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.7, 'NASA Exoplanet Detection',
                ha='center', fontsize=28, fontweight='bold')
        fig.text(0.5, 0.63, '6-Model Machine Learning Pipeline',
                ha='center', fontsize=20)
        fig.text(0.5, 0.56, 'Final Comparison Report',
                ha='center', fontsize=18, style='italic')

        # Summary statistics
        summary = benchmark_data['summary']
        summary_text = f"""
        Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Dataset Information:
        • Total Samples: 1,000
        • Training Set: 600 samples
        • Validation Set: 200 samples
        • Test Set: 200 samples

        Pipeline Summary:
        • Total Models Tested: {summary['total_models']}
        • All Models Successful: [OK]
        • Total Training Time: {summary['total_training_time']:.2f} seconds

        Best Performers:
        • Highest Accuracy: {summary['best_accuracy']} ({models_data[summary['best_accuracy']]['accuracy']:.4f})
        • Highest ROC-AUC: {summary['best_roc_auc']} ({models_data[summary['best_roc_auc']]['roc_auc']:.4f})
        • Fastest Training: {summary['fastest_model']} ({models_data[summary['fastest_model']]['training_time']:.2f}s)

        Average Performance:
        • Accuracy: {summary['average_accuracy']:.4f}
        • ROC-AUC: {summary['average_roc_auc']:.4f}
        """

        fig.text(0.1, 0.45, summary_text, fontsize=11, verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # PAGE 2: Performance Comparison Charts
        img = plt.imread(comparison_chart_path)
        fig = plt.figure(figsize=(11, 8.5))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # PAGE 3: Results Table
        img = plt.imread(table_img_path)
        fig = plt.figure(figsize=(11, 8.5))
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # PAGE 4: All Confusion Matrices
        img = plt.imread(confusion_matrices_path)
        fig = plt.figure(figsize=(11, 8.5))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Confusion Matrices - All Models', fontsize=14, fontweight='bold', pad=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # PAGE 5-10: Individual Model Details
        for model_name in sorted(model_names, key=lambda x: models_data[x]['roc_auc'], reverse=True):
            fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
            fig.suptitle(f'Detailed Analysis: {model_name}', fontsize=16, fontweight='bold')

            # Left: Confusion Matrix
            cm_path = ARTIFACTS_DIR / model_name.lower().replace(' ', '_') / 'confusion_matrix.png'
            if cm_path.exists():
                img = plt.imread(cm_path)
                axes[0].imshow(img)
                axes[0].axis('off')
                axes[0].set_title('Confusion Matrix', fontweight='bold')

            # Right: Metrics Summary
            axes[1].axis('off')
            model = models_data[model_name]

            metrics_text = f"""
            Performance Metrics:

            Accuracy:     {model['accuracy']:.4f}
            Precision:    {model['precision']:.4f}
            Recall:       {model['recall']:.4f}
            F1-Score:     {model['f1']:.4f}
            ROC-AUC:      {model['roc_auc']:.4f}

            Training Time: {model['training_time']:.2f} seconds

            Data Split:
            • Train: {model['data_split']['train_size']} samples
            • Validation: {model['data_split']['val_size']} samples
            • Test: {model['data_split']['test_size']} samples
            """

            axes[1].text(0.1, 0.9, metrics_text, fontsize=12, verticalalignment='top',
                        family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'NASA Exoplanet Detection - 6-Model Comparison Report'
        d['Author'] = 'ML Training Pipeline'
        d['Subject'] = 'Machine Learning Model Comparison'
        d['Keywords'] = 'NASA, Exoplanet, Machine Learning, Comparison'
        d['CreationDate'] = datetime.now()

    print(f"[OK] Saved comprehensive PDF report: {pdf_path}")
    return pdf_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  FINAL REPORT GENERATOR - NASA Exoplanet Detection")
    print("="*70 + "\n")

    # Generate all components
    print("Step 1:  Generating performance comparison charts...")
    comparison_chart_path = create_comparison_charts()

    print("\nStep 2:  Generating confusion matrices grid...")
    confusion_matrices_path = create_confusion_matrices_grid()

    print("\nStep 3:  Creating results table...")
    table_df, csv_path, table_img_path = create_results_table()

    print("\nStep 4:  Generating comprehensive PDF report...")
    pdf_path = create_pdf_report(comparison_chart_path, confusion_matrices_path,
                                 table_img_path, table_df)

    print("\n" + "="*70)
    print("  [OK] REPORT GENERATION COMPLETE!")
    print("="*70)
    print(f"\n[DIR] Output directory: {OUTPUT_DIR.absolute()}")
    print("\n[FILES] Generated files:")
    print(f"   - Comparison Charts: {comparison_chart_path.name}")
    print(f"   - Confusion Matrices: {confusion_matrices_path.name}")
    print(f"   - Results Table (CSV): {csv_path.name}")
    print(f"   - Results Table (PNG): {table_img_path.name}")
    print(f"   - Comprehensive PDF: {pdf_path.name}")
    print("\n" + "="*70 + "\n")
