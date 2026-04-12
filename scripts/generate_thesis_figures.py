#!/usr/bin/env python3
"""Generate thesis-ready scientific figures using a Nature/Science-style palette."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib

# Scientific plotting style configuration
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10

# Nature-style color palette (color-blind friendly)
COLORS = {
    'early': '#2878B5',      # Deep blue - Early Fusion
    'late': '#9AC9DB',       # Light blue - Late Fusion
    'deam': '#C82423',       # Red - DEAM metrics
    'mtg': '#FFBE7A',        # Orange - MTG metrics
    'train': '#2878B5',      # Training
    'val': '#C82423',        # Validation
    'ccc': '#2878B5',        # CCC
    'auc': '#FFBE7A',        # AUC
}

def load_metrics(file_path):
    """Load metrics.csv."""
    df = pd.read_csv(file_path)
    return df

def plot_training_curves(early_df, late_df, save_path):
    """
    Figure 1: training curve comparison.
    Shows training loss and validation metric trends for Early vs Late Fusion.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Process Comparison: Early vs Late Fusion', fontsize=14, fontweight='bold')

    # Subplot 1: training loss
    ax1 = axes[0, 0]
    ax1.plot(early_df['epoch'], early_df['train_loss'], color=COLORS['early'], linewidth=2, label='Early Fusion')
    ax1.plot(late_df['epoch'], late_df['train_loss'], color=COLORS['late'], linewidth=2, linestyle='--', label='Late Fusion')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('(a) Training Loss Convergence')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: DEAM CCC
    ax2 = axes[0, 1]
    ax2.plot(early_df['epoch'], early_df['val_deam_ccc'], color=COLORS['early'], linewidth=2, label='Early Fusion')
    ax2.plot(late_df['epoch'], late_df['val_deam_ccc'], color=COLORS['late'], linewidth=2, linestyle='--', label='Late Fusion')
    ax2.axhline(y=0.65, color='gray', linestyle=':', alpha=0.7, label='SOTA Arousal CCC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('DEAM CCC (Validation)')
    ax2.set_title('(b) DEAM Concordance Correlation')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: MTG ROC-AUC
    ax3 = axes[1, 0]
    ax3.plot(early_df['epoch'], early_df['mtg_roc_auc_micro'], color=COLORS['early'], linewidth=2, label='Early Fusion')
    ax3.plot(late_df['epoch'], late_df['mtg_roc_auc_micro'], color=COLORS['late'], linewidth=2, linestyle='--', label='Late Fusion')
    ax3.axhline(y=0.78, color='gray', linestyle=':', alpha=0.7, label='MERT SOTA')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MTG ROC-AUC (Micro)')
    ax3.set_title('(c) MTG Classification Performance')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: RMSE
    ax4 = axes[1, 1]
    ax4.plot(early_df['epoch'], early_df['val_deam_rmse'], color=COLORS['early'], linewidth=2, label='Early Fusion')
    ax4.plot(late_df['epoch'], late_df['val_deam_rmse'], color=COLORS['late'], linewidth=2, linestyle='--', label='Late Fusion')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('DEAM RMSE')
    ax4.set_title('(d) DEAM Root Mean Square Error')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_final_comparison(early_df, late_df, save_path):
    """
    Figure 2: final performance bar chart.
    Compares key metrics across both tasks.
    """
    # Use the final epoch metrics
    early_final = early_df.iloc[-1]
    late_final = late_df.iloc[-1]

    metrics = ['DEAM CCC', 'DEAM RMSE', 'MTG ROC-AUC\n(Micro)', 'MTG ROC-AUC\n(Macro)']
    early_values = [
        early_final['val_deam_ccc'],
        early_final['val_deam_rmse'],
        early_final['mtg_roc_auc_micro'],
        early_final['mtg_roc_auc_macro']
    ]
    late_values = [
        late_final['val_deam_ccc'],
        late_final['val_deam_rmse'],
        late_final['mtg_roc_auc_micro'],
        late_final['mtg_roc_auc_macro']
    ]

    # SOTA reference values for comparison
    sota_values = [0.65, 0.85, 0.781, 0.65]  # Approximate SOTA values

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, early_values, width, label='Early Fusion', color=COLORS['early'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, late_values, width, label='Late Fusion', color=COLORS['late'], edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, sota_values, width, label='SOTA Reference', color='lightgray', edgecolor='black', linewidth=0.5, alpha=0.7)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score')
    ax.set_title('Final Performance Comparison Across Tasks', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_deam_breakdown(early_df, late_df, save_path):
    """
    Figure 3: DEAM metric breakdown.
    Compares Valence and Arousal CCC scores.
    """
    early_final = early_df.iloc[-1]
    late_final = late_df.iloc[-1]

    categories = ['Valence\nCCC', 'Arousal\nCCC', 'Overall\nCCC', 'Pearson\nr']
    early_values = [
        early_final['val_deam_ccc_valence'],
        early_final['val_deam_ccc_arousal'],
        early_final['val_deam_ccc_epoch'],
        early_final['val_deam_pearson']
    ]
    late_values = [
        late_final['val_deam_ccc_valence'],
        late_final['val_deam_ccc_arousal'],
        late_final['val_deam_ccc_epoch'],
        late_final['val_deam_pearson']
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x - width/2, early_values, width, label='Early Fusion', color=COLORS['early'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, late_values, width, label='Late Fusion', color=COLORS['late'], edgecolor='black', linewidth=0.5)

    # Add reference lines
    ax.axhline(y=0.60, color='red', linestyle='--', alpha=0.5, label='Valence SOTA (~0.60)')
    ax.axhline(y=0.70, color='orange', linestyle='--', alpha=0.5, label='Arousal SOTA (~0.70)')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('DEAM Regression Task: Detailed Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_mtg_breakdown(early_df, late_df, save_path):
    """
    Figure 4: MTG metric breakdown.
    Shows detailed classification metrics.
    """
    early_final = early_df.iloc[-1]
    late_final = late_df.iloc[-1]

    categories = ['ROC-AUC\n(Micro)', 'ROC-AUC\n(Macro)', 'PR-AUC\n(Micro)', 'PR-AUC\n(Macro)', 'F1\n(Micro)', 'F1\n(Macro)']
    early_values = [
        early_final['mtg_roc_auc_micro'],
        early_final['mtg_roc_auc_macro'],
        early_final['mtg_pr_auc_micro'],
        early_final['mtg_pr_auc_macro'],
        early_final['mtg_f1_micro'],
        early_final['mtg_f1_macro']
    ]
    late_values = [
        late_final['mtg_roc_auc_micro'],
        late_final['mtg_roc_auc_macro'],
        late_final['mtg_pr_auc_micro'],
        late_final['mtg_pr_auc_macro'],
        late_final['mtg_f1_micro'],
        late_final['mtg_f1_macro']
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, early_values, width, label='Early Fusion', color=COLORS['early'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, late_values, width, label='Late Fusion', color=COLORS['late'], edgecolor='black', linewidth=0.5)

    # Add SOTA reference line
    ax.axhline(y=0.781, color='gray', linestyle='--', alpha=0.5, label='MERT SOTA ROC-AUC (0.781)')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_ylabel('Score')
    ax.set_title('MTG Multi-label Classification: Detailed Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_convergence_analysis(early_df, late_df, save_path):
    """
    Figure 5: convergence speed analysis.
    Shows epochs required to reach stable performance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Compute moving averages
    window = 3
    early_ccc_smooth = early_df['val_deam_ccc'].rolling(window=window, min_periods=1).mean()
    late_ccc_smooth = late_df['val_deam_ccc'].rolling(window=window, min_periods=1).mean()
    early_auc_smooth = early_df['mtg_roc_auc_micro'].rolling(window=window, min_periods=1).mean()
    late_auc_smooth = late_df['mtg_roc_auc_micro'].rolling(window=window, min_periods=1).mean()

    # Left panel: DEAM CCC convergence
    ax1.plot(early_df['epoch'], early_ccc_smooth, color=COLORS['early'], linewidth=2.5, label='Early Fusion')
    ax1.plot(late_df['epoch'], late_ccc_smooth, color=COLORS['late'], linewidth=2.5, label='Late Fusion')
    ax1.axvline(x=10, color='gray', linestyle=':', alpha=0.7, label='Early Stopping Check')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('DEAM CCC (Smoothed)')
    ax1.set_title('(a) DEAM Task Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right panel: MTG AUC convergence
    ax2.plot(early_df['epoch'], early_auc_smooth, color=COLORS['early'], linewidth=2.5, label='Early Fusion')
    ax2.plot(late_df['epoch'], late_auc_smooth, color=COLORS['late'], linewidth=2.5, label='Late Fusion')
    ax2.axvline(x=10, color='gray', linestyle=':', alpha=0.7, label='Early Stopping Check')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MTG ROC-AUC (Smoothed)')
    ax2.set_title('(b) MTG Task Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Model Convergence Speed Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def generate_summary_table(early_df, late_df, save_path):
    """Generate a summary table and save it as CSV."""
    early_final = early_df.iloc[-1]
    late_final = late_df.iloc[-1]

    summary_data = {
        'Metric': [
            'DEAM CCC (Overall)', 'DEAM CCC (Valence)', 'DEAM CCC (Arousal)',
            'DEAM RMSE', 'DEAM Pearson r',
            'MTG ROC-AUC (Micro)', 'MTG ROC-AUC (Macro)',
            'MTG PR-AUC (Micro)', 'MTG PR-AUC (Macro)',
            'MTG F1 (Micro)', 'MTG F1 (Macro)'
        ],
        'Early Fusion': [
            early_final['val_deam_ccc_epoch'], early_final['val_deam_ccc_valence'], early_final['val_deam_ccc_arousal'],
            early_final['val_deam_rmse'], early_final['val_deam_pearson'],
            early_final['mtg_roc_auc_micro'], early_final['mtg_roc_auc_macro'],
            early_final['mtg_pr_auc_micro'], early_final['mtg_pr_auc_macro'],
            early_final['mtg_f1_micro'], early_final['mtg_f1_macro']
        ],
        'Late Fusion': [
            late_final['val_deam_ccc_epoch'], late_final['val_deam_ccc_valence'], late_final['val_deam_ccc_arousal'],
            late_final['val_deam_rmse'], late_final['val_deam_pearson'],
            late_final['mtg_roc_auc_micro'], late_final['mtg_roc_auc_macro'],
            late_final['mtg_pr_auc_micro'], late_final['mtg_pr_auc_macro'],
            late_final['mtg_f1_micro'], late_final['mtg_f1_macro']
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df['Difference'] = summary_df['Early Fusion'] - summary_df['Late Fusion']
    summary_df['Better'] = summary_df['Difference'].apply(lambda x: 'Early' if x > 0 else 'Late')

    summary_df.to_csv(save_path, index=False, float_format='%.4f')
    print(f"Saved: {save_path}")
    return summary_df

def main():
    """Main entry point."""
    # Data paths
    early_metrics_path = "runs/experiment_suite_20260308_130926/early_baseline/unified_mood_model_early/metrics.csv"
    late_metrics_path = "runs/experiment_suite_20260308_230655/late_baseline/unified_mood_model_late/metrics.csv"

    # Output directory
    output_dir = Path("thesis_figures")
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading metrics data...")
    early_df = load_metrics(early_metrics_path)
    late_df = load_metrics(late_metrics_path)

    print(f"Early Fusion: {len(early_df)} epochs")
    print(f"Late Fusion: {len(late_df)} epochs")

    # Generate figures
    print("\nGenerating figures...")

    plot_training_curves(early_df, late_df, output_dir / "fig1_training_curves.png")
    plot_final_comparison(early_df, late_df, output_dir / "fig2_final_comparison.png")
    plot_deam_breakdown(early_df, late_df, output_dir / "fig3_deam_breakdown.png")
    plot_mtg_breakdown(early_df, late_df, output_dir / "fig4_mtg_breakdown.png")
    plot_convergence_analysis(early_df, late_df, output_dir / "fig5_convergence.png")

    # Generate summary table
    summary_df = generate_summary_table(early_df, late_df, output_dir / "results_summary.csv")
    print("\n=== Results Summary ===")
    print(summary_df.to_string(index=False))

    print(f"\nAll figures saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
