"""
Generate the supervised-baseline-vs-few-shot comparison figures and the
one-glance summary requested in the final-defense panel revisions.

  Figure 4.6a - PR-AUC vs K: Cross-Device & Cross-Position FSL vs Supervised baseline
  Figure 4.6b - F1 vs K:     Cross-Device & Cross-Position FSL vs Supervised baseline
  Figure 4.7  - One-glance summary: grouped bars at K = 10 across all five metrics
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path('results')
FIG_DIR = RESULTS_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [1, 3, 5, 10]
DEVICES = ['M7', 'X7', 'X300']

# Palette consistent with regen_nb4_figures.py
COLOR_XDEV = '#2196F3'   # blue  - cross-device FSL
COLOR_XPOS = '#4CAF50'   # green - cross-position FSL
COLOR_BASE = '#616161'   # gray  - supervised baseline reference

# Supervised CNN baseline grand mean across all devices (results/exp1_baseline.txt)
BASELINE = {
    'accuracy': 0.735,
    'precision': 0.748,
    'recall': 0.973,
    'f1': 0.839,
    'auc_pr': 0.949,
}

with open(RESULTS_DIR / 'exp2_cross_device.json') as f:
    exp2 = json.load(f)
with open(RESULTS_DIR / 'exp3_cross_position.json') as f:
    exp3 = json.load(f)


def _get(d, key):
    """dict.get tolerant of str/int key variants."""
    if not isinstance(d, dict):
        return None
    return d.get(str(key), d.get(key))


def _is_num(v):
    return isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))


def exp2_grandmean(metric, k):
    """Mean (and sample std) across the 3 target devices at shot K.
    Matches the Cross-Device grand-mean rows in exp2-3_summary.txt."""
    vals = []
    for dev in DEVICES:
        entry = _get(exp2.get(dev, {}), k)
        v = entry.get(metric) if isinstance(entry, dict) else None
        if _is_num(v):
            vals.append(float(v))
    mean = float(np.mean(vals)) if vals else float('nan')
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def exp3_grandmean(metric, k):
    """Mean (and sample std) across all device x fold runs at shot K.
    Matches the Cross-Position grand-mean rows in exp2-3_summary.txt."""
    vals = []
    for dev in exp3:
        for fold in exp3[dev]:
            entry = _get(exp3[dev][fold], k)
            v = entry.get(metric) if isinstance(entry, dict) else None
            if _is_num(v):
                vals.append(float(v))
    mean = float(np.mean(vals)) if vals else float('nan')
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def save_figure(fig, base, bbox_inches=None):
    base = Path(base).with_suffix('')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches=bbox_inches)
    fig.savefig(base.with_suffix('.png'), dpi=600, bbox_inches=bbox_inches)
    print(f'Saved: {base.name}.{{pdf,png}}')


# ===== FIGURE 4.6a / 4.6b - K-shot comparison against the supervised baseline =====
def draw_comparison(ax, metric, baseline_val, ylabel, ylim, legend=True):
    """Draw the FSL-vs-baseline comparison for one metric onto a given axis."""
    xdev = [exp2_grandmean(metric, k) for k in K_LIST]
    xpos = [exp3_grandmean(metric, k) for k in K_LIST]
    xdev_m, xdev_s = [m for m, _ in xdev], [s for _, s in xdev]
    xpos_m, xpos_s = [m for m, _ in xpos], [s for _, s in xpos]

    ax.axhline(baseline_val, color=COLOR_BASE, linestyle='--', linewidth=1.3,
               label=f'Supervised CNN Baseline ({baseline_val:.3f})', zorder=1)
    ax.errorbar(K_LIST, xpos_m, yerr=xpos_s, color=COLOR_XPOS, marker='s',
                capsize=3, elinewidth=0.9, linewidth=1.8,
                label='Cross-Position FSL (grand mean)', zorder=3)
    ax.errorbar(K_LIST, xdev_m, yerr=xdev_s, color=COLOR_XDEV, marker='o',
                capsize=3, elinewidth=0.9, linewidth=1.8,
                label='Cross-Device FSL (grand mean)', zorder=2)
    ax.set_xlabel('K (Support Samples Per Class)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(K_LIST)
    ax.set_ylim(*ylim)
    if legend:
        ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    return xdev_m, xpos_m


def comparison_curve(metric, ylabel, baseline_val, ylim, out_name):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xdev_m, xpos_m = draw_comparison(ax, metric, baseline_val, ylabel, ylim)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / out_name)
    plt.close()
    print(f'  Cross-Device  {metric}: {[round(v, 3) for v in xdev_m]}')
    print(f'  Cross-Position {metric}: {[round(v, 3) for v in xpos_m]}')
    print(f'  Baseline {metric}: {baseline_val}')


print('Figure 4.6a - PR-AUC vs K (FSL vs supervised baseline)...')
comparison_curve('auc_pr', 'PR-AUC', BASELINE['auc_pr'], (0.55, 1.0),
                 'fig4_6a_comparison_pr_auc_kshot')
print('Figure 4.6b - F1 vs K (FSL vs supervised baseline)...')
comparison_curve('f1', 'F1', BASELINE['f1'], (0.45, 1.0),
                 'fig4_6b_comparison_f1_kshot')

# ===== FIGURE 4.7 - One-glance summary: grouped bars at K = 10 =====
print('Figure 4.7 - one-glance summary (grouped bars at K = 10)...')
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_pr']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'PR-AUC']
sup_vals = [BASELINE[m] for m in metrics]
xdev_vals = [exp2_grandmean(m, 10)[0] for m in metrics]
xpos_vals = [exp3_grandmean(m, 10)[0] for m in metrics]

x = np.arange(len(metrics))
width = 0.27
fig, ax = plt.subplots(figsize=(8.5, 5.0))
b1 = ax.bar(x - width, sup_vals, width, color=COLOR_BASE, alpha=0.9,
            label='Supervised CNN Baseline (all labels)')
b2 = ax.bar(x, xdev_vals, width, color=COLOR_XDEV, alpha=0.9,
            label='Cross-Device FSL (K = 10)')
b3 = ax.bar(x + width, xpos_vals, width, color=COLOR_XPOS, alpha=0.9,
            label='Cross-Position FSL (K = 10)')
for bars in (b1, b2, b3):
    ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0.4, 1.08)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=8,
          frameon=False)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
save_figure(fig, FIG_DIR / 'fig4_7_summary_k10_barchart', bbox_inches='tight')
plt.close()
print(f'  Supervised     : {[round(v, 3) for v in sup_vals]}')
print(f'  Cross-Device   : {[round(v, 3) for v in xdev_vals]}')
print(f'  Cross-Position : {[round(v, 3) for v in xpos_vals]}')
print('  (order: Accuracy, Precision, Recall, F1, PR-AUC)')

# ===== Combined side-by-side version of Figure 4.6 (PR-AUC | F1) =====
# Mirrors the existing results/figures/side by side/ composites.
print('Combined side-by-side Figure 4.6 (PR-AUC | F1)...')
SBS_DIR = FIG_DIR / 'side by side'
SBS_DIR.mkdir(parents=True, exist_ok=True)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
draw_comparison(axL, 'auc_pr', BASELINE['auc_pr'], 'PR-AUC', (0.55, 1.0))
draw_comparison(axR, 'f1', BASELINE['f1'], 'F1', (0.45, 1.0))
fig.tight_layout()
save_figure(fig, SBS_DIR / 'fig4_6_comparison_kshot')
plt.close()

print('\nAll revision figures saved to results/figures/.')
