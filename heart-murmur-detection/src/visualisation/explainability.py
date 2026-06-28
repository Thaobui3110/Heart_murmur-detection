"""
src/visualisation/explainability.py
Hàm vẽ cho Phase 4 — Analysis & Explainability.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.visualisation.style import MURMUR_COLORS, setup_style


CLASSES = ['Present', 'Unknown', 'Absent']


def plot_confusion_matrix(cm, labels=None, title='Confusion Matrix',
                          weighted_accuracy=None, save_path=None):
    """
    Vẽ confusion matrix: raw counts (trái) và normalized (phải).

    Parameters
    ----------
    cm : np.ndarray (3, 3)
    labels : list of str
    weighted_accuracy : float, optional
    save_path : str, optional
    """
    if labels is None:
        labels = CLASSES

    setup_style()

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ['d', '.1%'],
        ['Raw Counts', 'Normalized (Recall)']
    ):
        im = ax.imshow(data, cmap='Blues', vmin=0,
                       vmax=data.max() * 1.1)

        # Annotate từng ô
        for i in range(len(labels)):
            for j in range(len(labels)):
                val_raw  = cm[i, j]
                val_norm = cm_norm[i, j]
                text = f'{val_raw}\n({val_norm:.1%})' if fmt == 'd' else f'{val_norm:.1%}\n(n={val_raw})'
                color = 'white' if data[i, j] > data.max() * 0.6 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(subtitle, fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Title tổng + weighted accuracy
    wa_str = f'  |  Weighted Accuracy = {weighted_accuracy:.3f}' if weighted_accuracy else ''
    fig.suptitle(f'{title}{wa_str}', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return fig, axes

def plot_roc_curve(y_true_binary, y_score, operating_point_score=0.0,
                   auc_value=None, save_path=None):
    """
    Vẽ ROC curve cho binary murmur detection (Present vs không-Present).

    Parameters
    ----------
    y_true_binary : array-like, 1 = Present, 0 = Absent (Unknown đã loại)
    y_score : array-like, score liên tục (c_mn_max)
    operating_point_score : float, ngưỡng hiện tại để đánh dấu operating point
    """
    from sklearn.metrics import roc_curve, auc as sk_auc

    setup_style()

    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    roc_auc = sk_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))

    # ROC curve + AUC shading
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.08, color='steelblue')

    # Baseline
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')

    # Operating point (threshold = 0)
    idx = np.argmin(np.abs(thresholds - operating_point_score))
    ax.scatter(fpr[idx], tpr[idx], s=100, color='crimson', zorder=5,
               label=f'Operating point (threshold={operating_point_score})\n'
                     f'FPR={fpr[idx]:.3f}, TPR={tpr[idx]:.3f}')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curve — Murmur Present vs Non-Present', fontsize=13,
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return fig, ax, roc_auc


def plot_per_class_metrics(df_metrics, df_plos_ref=None, save_path=None):
    """
    Bar chart grouped: Sensitivity, Specificity, PPV theo class.
    Overlay PLOS 2024 reference nếu có.
    """
    setup_style()

    metrics_to_plot = ['Sensitivity', 'Specificity', 'PPV', 'F1']
    classes = df_metrics.index.tolist()
    n_metrics = len(metrics_to_plot)
    n_classes  = len(classes)

    x = np.arange(n_classes)
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for i, metric in enumerate(metrics_to_plot):
        vals = df_metrics[metric].values
        bars = ax.bar(x + i * width, vals, width,
                      label=metric, color=colors[i], alpha=0.85)

        # PLOS reference — chỉ Sensitivity và PPV
        if df_plos_ref is not None and metric in df_plos_ref.columns:
            ref_vals = df_plos_ref[metric].values
            for j, (bar, ref) in enumerate(zip(bars, ref_vals)):
                ax.hlines(ref, bar.get_x(), bar.get_x() + bar.get_width(),
                          colors='black', linewidths=1.5, linestyles='--')

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_title('Per-class Metrics vs PLOS 2024 Reference (dashed)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Annotate giá trị trên mỗi bar
    for bar_group in ax.containers:
        ax.bar_label(bar_group, fmt='%.2f', fontsize=8, padding=2)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return fig, ax


def plot_confidence_scatter(df_pat_agg, save_path=None):
    """
    Tái tạo PLOS Fig.4: scatter C(M-N) vs C(ω̂), tô màu theo true_murmur.

    Parameters
    ----------
    df_pat_agg : pd.DataFrame
        Cần cột: 'c_mn_max', 'c_hat_min', 'true_murmur'
    """
    setup_style()

    class_order = ['Present', 'Unknown', 'Absent']
    colors = {
        'Present': '#e74c3c',
        'Unknown': '#f39c12',
        'Absent':  '#2980b9',
    }
    markers = {'Present': 'o', 'Unknown': 's', 'Absent': '^'}

    fig, ax = plt.subplots(figsize=(8, 7))

    for cls in class_order:
        mask = df_pat_agg['true_murmur'] == cls
        ax.scatter(
            df_pat_agg.loc[mask, 'c_mn_max'],
            df_pat_agg.loc[mask, 'c_hat_min'],
            c=colors[cls],
            marker=markers[cls],
            s=30, alpha=0.6, linewidths=0.3, edgecolors='white',
            label=f'{cls} (n={mask.sum()})',
            zorder=3
        )

    # Đường phân chia
    ax.axvline(x=0,    color='black', lw=1.5, ls='--', alpha=0.7,
               label='C(M−N) = 0')
    ax.axhline(y=0.65, color='gray',  lw=1.5, ls=':',  alpha=0.7,
               label='C(ω̂) = 0.65 (Unknown threshold)')

    # Label 4 vùng
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
# Thay 3 dòng text hiện tại bằng:
    ax.text( 0.75,  0.97, 'PRESENT\nzone', transform=ax.transAxes,
             ha='center', va='top', fontsize=9, color='#e74c3c',
             alpha=0.7, style='italic', fontweight='bold')
    ax.text( 0.08,  0.97, 'ABSENT\nzone',  transform=ax.transAxes,
             ha='center', va='top', fontsize=9, color='#2980b9',
             alpha=0.7, style='italic', fontweight='bold')
    ax.text( 0.08,  0.25, 'UNKNOWN\nzone', transform=ax.transAxes,
             ha='center', va='bottom', fontsize=9, color='#f39c12',
             alpha=0.7, style='italic', fontweight='bold')

    ax.set_xlabel('C(M−N)  [Murmur confidence − Normal confidence]', fontsize=12)
    ax.set_ylabel('C(ω̂)  [Max segmentation confidence]', fontsize=12)
    ax.set_title('HSMM Confidence Scatter\n(Reproduction of PLOS Fig. 4)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9,
              bbox_to_anchor=(1.0, 0.0))
    ax.grid(True, alpha=0.25)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return fig, ax


def plot_reliability_diagram(df_rec, confidence_col='c_hat',
                              n_bins=10, save_path=None):
    """
    Reliability diagram: C(ω̂) vs fraction correct per bin.
    'Correct' ở recording level: Present rec với c_mn > 0, hoặc Absent rec với c_mn < 0.

    Parameters
    ----------
    df_rec : pd.DataFrame
        Cần cột: c_hat, c_mn, true_murmur (recording-level)
    """
    setup_style()

    # Định nghĩa "correct" ở recording level
    # Present recording đoán đúng nếu c_mn > 0
    # Absent recording đoán đúng nếu c_mn < 0
    # Unknown recording → exclude (không có ground truth rõ ràng)
    df_eval = df_rec[df_rec['true_murmur'] != 'Unknown'].copy()
    df_eval['correct'] = (
        ((df_eval['true_murmur'] == 'Present') & (df_eval['c_mn'] > 0)) |
        ((df_eval['true_murmur'] == 'Absent')  & (df_eval['c_mn'] < 0))
    ).astype(int)

    # Bin theo c_hat
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_acc   = []
    bin_conf  = []
    bin_count = []

    for i in range(n_bins):
        mask = (df_eval['c_hat'] >= bins[i]) & (df_eval['c_hat'] < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_acc.append(df_eval.loc[mask, 'correct'].mean())
        bin_conf.append(df_eval.loc[mask, 'c_hat'].mean())
        bin_count.append(mask.sum())

    bin_acc   = np.array(bin_acc)
    bin_conf  = np.array(bin_conf)
    bin_count = np.array(bin_count)

    # ECE
    ece = np.sum(bin_count * np.abs(bin_acc - bin_conf)) / bin_count.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel trái: Reliability diagram ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
    ax.bar(bin_conf, bin_acc, width=0.07, alpha=0.6, color='steelblue',
           label='Model', edgecolor='steelblue')
    ax.plot(bin_conf, bin_acc, 'o-', color='steelblue', lw=2, ms=6)

    # Shade gap (over/underconfidence)
    ax.fill_between(bin_conf, bin_conf, bin_acc,
                    alpha=0.15, color='red', label='Calibration gap')

    ax.set_xlabel('Mean Confidence C(ω̂)', fontsize=12)
    ax.set_ylabel('Fraction Correct', fontsize=12)
    ax.set_title(f'Reliability Diagram\n(ECE = {ece:.4f})', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # --- Panel phải: Histogram số lượng mỗi bin ---
    ax2 = axes[1]
    ax2.bar(bin_conf, bin_count, width=0.07, color='#95a5a6',
            edgecolor='gray', alpha=0.8)
    ax2.set_xlabel('Confidence C(ω̂)', fontsize=12)
    ax2.set_ylabel('Number of Recordings', fontsize=12)
    ax2.set_title('Confidence Distribution\n(recording-level)', fontsize=13,
                  fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Calibration Analysis — HSMM Segmentation Confidence',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')

    print(f'ECE = {ece:.4f}')
    print(f'Bins used: {len(bin_count)} | Total recordings: {bin_count.sum()}')
    return fig, axes, ece

def plot_frequency_importance(importance, freqs,
                               phase2_correlation=None,
                               save_path=None):
    """
    Vẽ frequency-band ablation importance.

    Parameters
    ----------
    importance : np.ndarray (41,)
        Mean ΔC(M−N) per frequency bin
    freqs : np.ndarray (41,)
        Frequency values in Hz
    phase2_correlation : np.ndarray (101,) hoặc None
        Rank-biserial r từ Phase 2 Task 2.5c (optional, để so sánh)
    """
    setup_style()

    has_phase2 = phase2_correlation is not None
    n_panels   = 2 if has_phase2 else 1
    fig, axes  = plt.subplots(1, n_panels,
                               figsize=(13 if has_phase2 else 8, 5))
    if n_panels == 1:
        axes = [axes]

    # --- Panel 1: Ablation importance ---
    ax = axes[0]

    # Color bars by importance level
    colors = ['#e74c3c' if v == importance.max()
              else '#3498db' if v > 0
              else '#95a5a6'
              for v in importance]

    bars = ax.bar(freqs, importance, width=16,
                  color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

    # Peak annotation
    peak_idx  = importance.argmax()
    peak_freq = freqs[peak_idx]
    ax.annotate(
        f'Peak: {peak_freq} Hz\n(importance={importance[peak_idx]:.4f})',
        xy=(peak_freq, importance[peak_idx]),
        xytext=(peak_freq + 80, importance[peak_idx] * 0.9),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.8)
    )

    ax.axhline(y=0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Mean ΔC(M−N)\n[importance = confidence drop when ablated]',
                  fontsize=11)
    ax.set_title('Frequency-band Ablation Importance\n'
                 '(higher = model relies more on this frequency band)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-10, 820)
    ax.grid(axis='y', alpha=0.3)

    # Shade vùng 0–400 Hz (murmur energy zone)
    ax.axvspan(0, 400, alpha=0.06, color='#e74c3c',
               label='Murmur energy zone (0–400 Hz)')
    ax.legend(fontsize=9)

    # --- Panel 2: So sánh với Phase 2 correlation ---
    if has_phase2:
        ax2 = axes[1]

        # Phase 2 correlation chỉ lấy 41 bins đầu (0–800 Hz)
        corr_41 = phase2_correlation[:41] if len(phase2_correlation) > 41 \
                  else phase2_correlation

        # Normalize cả 2 về [0, 1] để so sánh shape
        imp_norm  = (importance - importance.min()) / \
                    (importance.max() - importance.min() + 1e-8)
        corr_norm = (corr_41 - corr_41.min()) / \
                    (corr_41.max() - corr_41.min() + 1e-8)

        ax2.plot(freqs, imp_norm, 'o-', color='#e74c3c', lw=2, ms=5,
                 label='Ablation importance (normalized)', alpha=0.85)
        ax2.plot(freqs[:len(corr_norm)], corr_norm, 's--',
                 color='#2980b9', lw=2, ms=5,
                 label='Phase 2 correlation (normalized)', alpha=0.85)

        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Normalized score', fontsize=11)
        ax2.set_title('Post-model Ablation vs Pre-model Correlation\n'
                      '(Phase 2 Task 2.5c)',
                      fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.set_xlim(-10, 820)
        ax2.grid(alpha=0.3)

        # Tính correlation giữa 2 profiles
        from scipy.stats import spearmanr
        rho, pval = spearmanr(importance, corr_41[:len(importance)])
        ax2.text(0.98, 0.05,
                 f'Spearman ρ = {rho:.3f}\n(p={pval:.3f})',
                 transform=ax2.transAxes, ha='right', va='bottom',
                 fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white',
                           edgecolor='gray', alpha=0.8))

    plt.suptitle('Per-frequency-band Feature Importance Analysis',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return fig