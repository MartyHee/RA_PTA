"""
Generate figures for model tuning and compression report (real_raw_1000).

Only generates charts from existing experiment outputs — no training, no quantization.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
C_BASELINE = '#4A7FB5'
C_TUNED = '#2E8B57'
C_FP32 = '#4A7FB5'
C_QUANTIZED = '#DAA520'
C_DNN = '#4A7FB5'
C_WD = '#CD853F'
C_MM = '#2E8B57'
C_DNN_Q = '#8FB5D9'
C_WD_Q = '#E0B87A'
C_MM_Q = '#7FCC9E'


def setup_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis='both', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def fig_tuning_val_auc():
    """Top-10 trials by val_auc from random search."""
    trials = [
        ("T16", 0.9980), ("T17", 0.9978), ("T12", 0.9970),
        ("T18", 0.9965), ("T8", 0.9957), ("T5", 0.9937),
        ("T10", 0.9933), ("T15", 0.9843), ("T3", 0.9837),
        ("T4", 0.9707),
    ]
    labels, vals = zip(*trials)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [C_TUNED if l == "T16" else C_BASELINE for l in labels]
    bars = ax.bar(labels, vals, color=colors, edgecolor='grey', linewidth=0.5)
    ax.axhline(y=0.9894, color='red', linestyle='--', linewidth=1, label='Baseline val_auc (0.9894)')
    ax.legend(fontsize=9, loc='lower right')

    setup_axis(ax, 'Multimodal Random Search — Top 10 Trials by Val AUC',
               'Trial ID (T16 = best, green)', 'Val AUC')
    ax.set_ylim(0.94, 1.005)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'tuning_val_auc.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {path}")


def fig_tuned_vs_baseline_metrics():
    """Baseline vs Tuned Multimodal test metrics."""
    metrics = ['AUC', 'F1', 'Precision', 'Recall']
    baseline = [0.9746, 0.8929, 0.9615, 0.8333]
    tuned = [0.9928, 0.9107, 0.9808, 0.8500]

    x = np.arange(len(metrics))
    w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, baseline, w, label='Baseline (7,857 params)', color=C_BASELINE, edgecolor='grey', linewidth=0.5)
    ax.bar(x + w / 2, tuned, w, label='Tuned (2,649 params)', color=C_TUNED, edgecolor='grey', linewidth=0.5)

    for i in range(len(metrics)):
        ax.text(i - w / 2, baseline[i] + 0.006, f'{baseline[i]:.4f}',
                ha='center', va='bottom', fontsize=8, color=C_BASELINE)
        ax.text(i + w / 2, tuned[i] + 0.006, f'{tuned[i]:.4f}',
                ha='center', va='bottom', fontsize=8, color=C_TUNED)

    setup_axis(ax, 'Multimodal Baseline vs Tuned — Test Metrics (real_raw_1000)',
               'Metric', 'Score')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.75, 1.02)
    ax.legend(fontsize=9, loc='lower right')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'tuned_vs_baseline_metrics.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {path}")


def fig_compression_model_size():
    """FP32 vs Quantized model size for all three model types."""
    models = ['Multimodal\n(2,649)', 'DNN\n(41,177)', 'Wide & Deep\n(42,760)']
    fp32 = [0.0136, 0.1607, 0.1678]
    quant = [0.0099, 0.1379, 0.1453]

    x = np.arange(len(models))
    w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w / 2, fp32, w, label='FP32', color=C_FP32, edgecolor='grey', linewidth=0.5)
    b2 = ax.bar(x + w / 2, quant, w, label='Quantized (INT8 Lin.)', color=C_QUANTIZED, edgecolor='grey', linewidth=0.5)

    for i in range(len(models)):
        ax.text(i - w / 2, fp32[i] + 0.002, f'{fp32[i]:.4f} MB',
                ha='center', va='bottom', fontsize=8, color=C_FP32)
        ratio = quant[i] / fp32[i]
        ax.text(i + w / 2, quant[i] + 0.002, f'{quant[i]:.4f} MB\n({ratio:.2f}x)',
                ha='center', va='bottom', fontsize=8, color=C_QUANTIZED)

    setup_axis(ax, 'Model Size: FP32 vs Dynamic Quantization',
               'Model (params)', 'Model Size (MB)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'compression_model_size.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {path}")


def fig_compression_latency():
    """Latency comparison across all model versions (CPU benchmark)."""
    labels = [
        'Multimodal\nFP32 CPU', 'Multimodal\nQuantized',
        'DNN\nFP32 CPU', 'DNN\nQuantized',
        'W&D\nFP32 CPU', 'W&D\nQuantized',
    ]
    latencies = [0.1134, 0.9258, 0.1753, 0.7707, 0.2788, 0.9428]
    colors = [C_MM, C_MM_Q, C_DNN, C_DNN_Q, C_WD, C_WD_Q]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, latencies, color=colors, edgecolor='grey', linewidth=0.5)

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f'{val:.4f} ms', ha='center', va='bottom', fontsize=8)

    # Highlight the degradation
    ax.axhline(y=0.1134, color=C_MM, linestyle=':', linewidth=1, alpha=0.6)
    ax.axhline(y=0.1753, color=C_DNN, linestyle=':', linewidth=1, alpha=0.6)
    ax.axhline(y=0.2788, color=C_WD, linestyle=':', linewidth=1, alpha=0.6)

    setup_axis(ax, 'Inference Latency — FP32 CPU vs Quantized (CPU, batch=1)',
               'Model Version', 'Avg Latency (ms)')
    ax.set_yscale('log')
    ax.set_ylim(0.05, 5)

    # Add note
    ax.text(0.5, -0.18, 'All metrics on CPU for fair comparison. Quantized = dynamic quantization on nn.Linear.',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='grey')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'compression_latency.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {path}")


def fig_compression_auc_f1():
    """AUC vs F1 trade-off across FP32 and Quantized versions."""
    labels = [
        'Multimodal FP32', 'Multimodal Quant',
        'DNN FP32', 'DNN Quant',
        'W&D FP32', 'W&D Quant',
    ]
    auc_vals = [0.99278, 0.99185, 0.97167, 0.96741, 0.95111, 0.94130]
    f1_vals = [0.91071, 0.92035, 0.86239, 0.84404, 0.84404, 0.82143]
    colors = [C_MM, C_MM_Q, C_DNN, C_DNN_Q, C_WD, C_WD_Q]
    markers = ['o', 's', 'o', 's', 'o', 's']
    sizes = [120, 100, 120, 100, 120, 100]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i in range(len(labels)):
        ax.scatter(auc_vals[i], f1_vals[i], c=colors[i], s=sizes[i],
                   marker=markers[i], edgecolors='grey', linewidth=0.5, zorder=5)
        offset_y = 0.006 if i % 2 == 0 else -0.016
        ax.text(auc_vals[i], f1_vals[i] + offset_y, labels[i],
                ha='center', va='bottom' if offset_y > 0 else 'top',
                fontsize=8, color=colors[i], fontweight='bold')

    # Arrows from FP32 -> Quantized for each model
    for idx in [0, 2, 4]:  # FP32 indices
        ax.annotate('', xy=(auc_vals[idx + 1], f1_vals[idx + 1]),
                    xytext=(auc_vals[idx], f1_vals[idx]),
                    arrowprops=dict(arrowstyle='->', color='grey', lw=1, alpha=0.5))

    setup_axis(ax, 'AUC vs F1: FP32 → Quantized Trajectory',
               'AUC', 'F1')
    ax.set_xlim(0.92, 1.005)
    ax.set_ylim(0.78, 0.95)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'compression_auc_f1.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {path}")


if __name__ == '__main__':
    print("Generating tuning & compression figures...")
    fig_tuning_val_auc()
    fig_tuned_vs_baseline_metrics()
    fig_compression_model_size()
    fig_compression_latency()
    fig_compression_auc_f1()
    print("All figures generated.")