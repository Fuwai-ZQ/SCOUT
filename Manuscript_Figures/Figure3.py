"""
Figure 3 – SCOUT multi-task validation composite figure (Nature Medicine style).
Panel a: Pareto frontier (review rate vs accuracy) with LOWESS fit.
Panel b: Baseline vs SCOUT accuracy across cohorts.
Panel c: Efficiency metrics (review rate + efficiency ratio).
Panel d: Mechanism breakdown (M1/M2/M3 triggered vs detected).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings('ignore')

# ── Nature Medicine style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.linewidth': 0.5,
    'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'legend.fontsize': 9, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Color scheme ──────────────────────────────────────────────────────────────
colors = {
    'scatter': '#4A7298', 'scatter_edge': '#2E4A62',
    'lowess': '#C44536', 'ci': '#C44536', 'ref_line': '#888888',
    'baseline': '#B4D4E8', 'final': '#2166AC',
    'review_bar': '#E6E6E6', 'efficiency_line': '#C45100',
    'M1': '#E64B35', 'M2': '#2166AC', 'M3': '#00A087',
    'grid': '#EBEBEB',
}

# ── Load CSV data ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'Table1_All_Combinations.csv')

df = pd.read_csv(CSV_PATH)
plot_data = df[df['Main_Model'] == 'DS3.1_withThinking'].copy()
plot_data['Review_Rate_Pct'] = plot_data['Review_Rate'] * 100
plot_data['Final_Acc_Pct'] = plot_data['Final_Acc'] * 100
x_scatter = plot_data['Review_Rate_Pct'].values
y_scatter = plot_data['Final_Acc_Pct'].values

# ── Panel B & C data (hardcoded from analysis results) ────────────────────────
accuracy_data = {
    'Diagnosis': {
        'cohorts': ['Discovery Cohort\n(n=405)', 'MIMIC-IV\n(n=1,866)', 'Multi-reader Study\n(n=110)'],
        'baseline': [92.4, 94.1, 90.9], 'final': [100.0, 100.0, 99.1],
    },
    'Screening': {
        'cohorts': ['MIMIC-IV\n(n=901)', 'PORTAL\n(n=2,472)'],
        'baseline': [95.7, 94.0], 'final': [100.0, 99.3],
    },
    'Counting': {
        'cohorts': ['CN RCTs\n(n=286)'],
        'baseline': [99.0], 'final': [100.0],
    }
}

efficiency_data = {
    'Diagnosis': {
        'cohorts': ['Discovery Cohort', 'MIMIC-IV', 'Multi-reader Study'],
        'review_rate': [54.1, 52.2, 54.5], 'efficiency_ratio': [1.85, 1.92, 1.65],
    },
    'Screening': {
        'cohorts': ['MIMIC-IV', 'PORTAL'],
        'review_rate': [28.9, 16.7], 'efficiency_ratio': [3.5, 6.0],
    },
    'Counting': {
        'cohorts': ['CN RCTs'],
        'review_rate': [17.8], 'efficiency_ratio': [5.6],
    }
}

mechanism_data = {
    'Diagnosis': {
        'Discovery Cohort': {
            'total_samples': 405, 'total_errors': 31,
            'M1': {'triggered': 189, 'detected': 28},
            'M2': {'triggered': 16, 'detected': 11},
            'M3': {'triggered': 48, 'detected': 8},
        },
        'MIMIC-IV': {
            'total_samples': 1866, 'total_errors': 110,
            'M1': {'triggered': 791, 'detected': 98},
            'M2': {'triggered': 145, 'detected': 59},
            'M3': {'triggered': 298, 'detected': 28},
        },
        'Multi-reader Study': {
            'total_samples': 110, 'total_errors': 10,
            'M1': {'triggered': 54, 'detected': 9},
            'M2': {'triggered': 3, 'detected': 1},
            'M3': {'triggered': 14, 'detected': 1},
        },
    },
    'Screening': {
        'MIMIC-IV': {
            'total_samples': 901, 'total_errors': 131,
            'M1': {'triggered': 115, 'detected': 104},
            'M2': {'triggered': 101, 'detected': 52},
            'M3': {'triggered': 107, 'detected': 53},
        },
        'PORTAL': {
            'total_samples': 2472, 'total_errors': 220,
            'M1': {'triggered': 175, 'detected': 162},
            'M2': {'triggered': 45, 'detected': 24},
            'M3': {'triggered': 199, 'detected': 117},
        },
    },
    'Counting': {
        'CN RCTs': {
            'total_samples': 286, 'total_errors': 3,
            'M1': {'triggered': 46, 'detected': 3},
            'M2': {'triggered': 5, 'detected': 2},
            'M3': {'triggered': 8, 'detected': 0},
        },
    }
}


# ── LOWESS functions ──────────────────────────────────────────────────────────
def local_regression(x, y, x_grid, bandwidth=0.2):
    y_fit = np.zeros_like(x_grid)
    h = bandwidth * (x.max() - x.min())
    for i, x0 in enumerate(x_grid):
        distances = np.abs(x - x0)
        weights = np.where(distances < h, (1 - (distances / h) ** 3) ** 3, 0)
        if weights.sum() > 0:
            w_sum = weights.sum()
            x_wm = np.sum(weights * x) / w_sum
            y_wm = np.sum(weights * y) / w_sum
            num = np.sum(weights * (x - x_wm) * (y - y_wm))
            den = np.sum(weights * (x - x_wm) ** 2)
            if den > 0:
                slope = num / den
                y_fit[i] = slope * x0 + (y_wm - slope * x_wm)
            else:
                y_fit[i] = y_wm
        else:
            y_fit[i] = np.nan
    return y_fit


def bootstrap_ci(x, y, x_grid, n_bootstrap=150, bandwidth=0.2):
    np.random.seed(42)
    n = len(x)
    y_boots = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        y_boots.append(local_regression(x[idx], y[idx], x_grid, bandwidth))
    y_boots = np.array(y_boots)
    return np.nanpercentile(y_boots, 2.5, axis=0), np.nanpercentile(y_boots, 97.5, axis=0)


x_grid = np.linspace(x_scatter.min(), x_scatter.max(), 150)
y_fit = local_regression(x_scatter, y_scatter, x_grid, bandwidth=0.2)
ci_lower, ci_upper = bootstrap_ci(x_scatter, y_scatter, x_grid)

# ── Create figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 8.2), dpi=300)
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 0.95, 1.15],
              width_ratios=[0.65, 1.35], hspace=0.42, wspace=0.25,
              left=0.085, right=0.92, top=0.96, bottom=0.065)

# ── Panel A: Pareto curve ────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.fill_between(x_grid, ci_lower, ci_upper, color=colors['ci'], alpha=0.15, linewidth=0)
ax_a.scatter(x_scatter, y_scatter, s=8, c=colors['scatter'],
             edgecolors=colors['scatter_edge'], linewidths=0.2, alpha=0.4, zorder=3)
ax_a.plot(x_grid, y_fit, color=colors['lowess'], linewidth=1.2, zorder=4)
ax_a.axhline(y=100, linestyle='--', color=colors['ref_line'], linewidth=0.5, alpha=0.7, zorder=1)

ax_a.set_xlabel('Review rate (%)')
ax_a.set_ylabel('Final accuracy (%)')
ax_a.set_xlim(-2, 72); ax_a.set_ylim(92, 100.8)
ax_a.set_xticks([0, 20, 40, 60]); ax_a.set_yticks([92, 94, 96, 98, 100])

# Mark optimal point (accuracy ≥99.99% with minimum review rate)
optimal_mask = y_scatter >= 99.99
if optimal_mask.any():
    opt_idx = np.where(optimal_mask)[0]
    min_rr_idx = opt_idx[np.argmin(x_scatter[opt_idx])]
    ax_a.scatter([x_scatter[min_rr_idx]], [y_scatter[min_rr_idx]], s=60,
                 facecolors='none', edgecolors='red', linewidths=0.8, zorder=5)

legend_a = [
    plt.Line2D([0], [0], color=colors['lowess'], linewidth=1, label='LOWESS'),
    mpatches.Patch(facecolor=colors['ci'], alpha=0.15, edgecolor='none', label='95% CI'),
]
ax_a.legend(handles=legend_a, loc='lower right', frameon=False, fontsize=6)
ax_a.text(-0.12, 1.06, 'a', transform=ax_a.transAxes, fontsize=12, fontweight='bold', va='top')
ax_a.yaxis.grid(True, linestyle='-', alpha=0.25, color=colors['grid'], zorder=0)
ax_a.set_axisbelow(True)

# ── Panel B: Accuracy comparison ─────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
all_cohorts, all_baseline, all_final = [], [], []
task_boundaries, task_labels = [], []
pos = 0
for task, data in accuracy_data.items():
    task_labels.append((pos + len(data['cohorts']) / 2 - 0.5, task))
    all_cohorts.extend(data['cohorts'])
    all_baseline.extend(data['baseline'])
    all_final.extend(data['final'])
    pos += len(data['cohorts'])
    task_boundaries.append(pos - 0.5)

x_pos = np.arange(len(all_cohorts))
width = 0.32
for b in task_boundaries[:-1]:
    ax_b.axvline(x=b, color='#CCCCCC', linestyle=':', linewidth=0.6, alpha=0.8)

bars_base = ax_b.bar(x_pos - width / 2, all_baseline, width, color=colors['baseline'],
                     edgecolor='white', linewidth=0.3, label='Baseline AI')
bars_final = ax_b.bar(x_pos + width / 2, all_final, width, color=colors['final'],
                      edgecolor='white', linewidth=0.3, label='With SCOUT')

for bar, val in zip(bars_base, all_baseline):
    ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
              f'{val:.1f}', ha='center', va='bottom', fontsize=5.5, color='#333333')
for bar, val in zip(bars_final, all_final):
    ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
              f'{val:.1f}', ha='center', va='bottom', fontsize=5.5, color='black')

for p, label in task_labels:
    x_off = 0.4 if label == 'Diagnosis' else 0
    ax_b.text(p + x_off, 103, label, ha='center', va='bottom', fontsize=8, color='black')

ax_b.set_ylabel('Final Clinical Performance (%)')
ax_b.set_ylim(90, 106); ax_b.set_yticks([90, 92, 94, 96, 98, 100])
ax_b.set_xticks(x_pos); ax_b.set_xticklabels(all_cohorts, fontsize=6, ha='center')
ax_b.legend(loc='upper left', fontsize=6, bbox_to_anchor=(0.01, 0.92))
ax_b.text(-0.08, 1.06, 'b', transform=ax_b.transAxes, fontsize=12, fontweight='bold', va='top')
ax_b.yaxis.grid(True, linestyle='-', alpha=0.25, color=colors['grid'], zorder=0)
ax_b.set_axisbelow(True)

# ── Panel C: Efficiency metrics ──────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, :])
ax_c_twin = ax_c.twinx()

all_eff_cohorts, all_rr, all_er = [], [], []
eff_boundaries, eff_labels = [], []
pos = 0
for task, data in efficiency_data.items():
    eff_labels.append((pos + len(data['cohorts']) / 2 - 0.5, task))
    all_eff_cohorts.extend(data['cohorts'])
    all_rr.extend(data['review_rate'])
    all_er.extend(data['efficiency_ratio'])
    pos += len(data['cohorts'])
    eff_boundaries.append(pos - 0.5)

x_pos_c = np.arange(len(all_eff_cohorts))
width_c = 0.55
for b in eff_boundaries[:-1]:
    ax_c.axvline(x=b, color='#CCCCCC', linestyle=':', linewidth=0.6, alpha=0.8)

bars_c = ax_c.bar(x_pos_c, all_rr, width_c, color=colors['review_bar'],
                  edgecolor='#CCCCCC', linewidth=0.4, alpha=0.9, zorder=3)
ax_c_twin.plot(x_pos_c, all_er, color=colors['efficiency_line'], marker='o', linewidth=1.8,
               markersize=7, markerfacecolor='white', markeredgewidth=1.4, zorder=4)

for bar, rr in zip(bars_c, all_rr):
    ax_c.text(bar.get_x() + bar.get_width() / 2, rr + 1.5, f'{rr:.1f}%',
              ha='center', va='bottom', fontsize=7, color='black')
for i, er in enumerate(all_er):
    ax_c_twin.text(x_pos_c[i], er + 0.35, f'{er:.1f}×', ha='center', va='bottom',
                   fontsize=8, color=colors['efficiency_line'])
for p, label in eff_labels:
    ax_c.text(p, 85, label, ha='center', va='bottom', fontsize=8, color='black')

ax_c.set_ylim(0, 100)
ax_c_twin.set_ylim(0, max(all_er) + 2.0)
ax_c.set_ylabel('Review rate (%)', color='black')
ax_c_twin.set_ylabel('Efficiency ratio', color='black', rotation=270, labelpad=12)
ax_c.set_xticks(x_pos_c); ax_c.set_xticklabels(all_eff_cohorts, fontsize=8)

ax_c.spines['left'].set_color('black'); ax_c.tick_params(axis='y', colors='black')
ax_c_twin.spines['right'].set_visible(True); ax_c_twin.spines['right'].set_color('black')
ax_c_twin.spines['right'].set_linewidth(0.5); ax_c_twin.tick_params(axis='y', colors='black')

ax_c.text(-0.05, 1.16, 'c', transform=ax_c.transAxes, fontsize=12, fontweight='bold', va='top')
legend_c = [
    mpatches.Patch(facecolor=colors['review_bar'], edgecolor='#CCCCCC', label='Review rate', alpha=0.9),
    Line2D([0], [0], color=colors['efficiency_line'], marker='o', markerfacecolor='white',
           markeredgewidth=1.4, label='Efficiency ratio', linewidth=1.8)
]
ax_c.legend(handles=legend_c, loc='upper left', fontsize=6, ncol=1, bbox_to_anchor=(0.01, 0.98))
ax_c.yaxis.grid(True, linestyle='-', alpha=0.25, color=colors['grid'], zorder=0)
ax_c.set_axisbelow(True)

# ── Panel D: Mechanism breakdown ─────────────────────────────────────────────
ax_d = fig.add_subplot(gs[2, :])

all_labels, all_triggered, all_detected, all_mech_colors = [], [], [], []
cohort_positions, cohort_labels = [], []
task_boundaries_d, task_labels_d = [], []

pos = 0
mechanism_order = ['M1', 'M2', 'M3']
for task in ['Diagnosis', 'Screening', 'Counting']:
    task_start = pos
    for cohort in mechanism_data[task]:
        cohort_start = pos
        cdata = mechanism_data[task][cohort]
        total = cdata['total_samples']
        for mech in mechanism_order:
            all_triggered.append(cdata[mech]['triggered'] / total * 100)
            all_detected.append(cdata[mech]['detected'] / total * 100)
            all_labels.append('')
            all_mech_colors.append(colors[mech])
            pos += 1
        cohort_positions.append(cohort_start + 1.0)
        cohort_labels.append(cohort)
    task_labels_d.append((task_start + (pos - task_start) / 2 - 0.5, task))
    task_boundaries_d.append(pos - 0.5)

x_pos_d = np.arange(len(all_labels))
width_d = 0.95

for b in task_boundaries_d[:-1]:
    ax_d.axvline(x=b, color='#CCCCCC', linestyle=':', linewidth=0.6, alpha=0.8)

# Background (triggered) and foreground (detected) bars
for x, trig in zip(x_pos_d, all_triggered):
    ax_d.bar(x, trig, width_d, color='#E8E8E8', alpha=0.8,
             edgecolor='#D0D0D0', linewidth=0.3, zorder=2)
for x, det, c in zip(x_pos_d, all_detected, all_mech_colors):
    ax_d.bar(x, det, width_d, color=c, alpha=0.9, edgecolor=c, linewidth=0.3, zorder=3)

# Triggered labels
for x, trig, det in zip(x_pos_d, all_triggered, all_detected):
    gap = trig - det
    offset = 5.0 if gap < 2 else (3.5 if gap < 4 else 1.8)
    ax_d.text(x, trig + offset, f'{trig:.1f}', ha='center', va='bottom', fontsize=5, color='black')

# Detected labels
for x, val, trig, c in zip(x_pos_d, all_detected, all_triggered, all_mech_colors):
    if val >= 6:
        ax_d.text(x, val / 2, f'{val:.1f}', ha='center', va='center', fontsize=5, color='white')
    elif val >= 0.1:
        ax_d.text(x, val + 0.8, f'{val:.1f}', ha='center', va='bottom', fontsize=4.5, color=c)
    else:
        ax_d.text(x, 1.0, f'{val:.1f}', ha='center', va='bottom', fontsize=4.5, color=c)

ax_d.set_xticks(x_pos_d)
ax_d.set_xticklabels([''] * len(all_labels))

for cp, cl in zip(cohort_positions, cohort_labels):
    ax_d.text(cp, -3.5, cl, ha='center', va='top', fontsize=7, color='black')
for p, label in task_labels_d:
    ax_d.text(p, max(all_triggered) + 8, label, ha='center', va='bottom', fontsize=8, color='black')

ax_d.set_ylim(0, max(all_triggered) + 16)
ax_d.set_ylabel('Percentage of total samples (%)')
ax_d.text(-0.04, 1.08, 'd', transform=ax_d.transAxes, fontsize=12, fontweight='bold', va='top')
ax_d.yaxis.grid(True, linestyle='-', alpha=0.25, color=colors['grid'], zorder=0)
ax_d.set_axisbelow(True)

legend_d = [
    mpatches.Patch(facecolor=colors['M1'], alpha=0.9, edgecolor=colors['M1'],
                   linewidth=0.5, label='S1 (Heterogeneity)'),
    mpatches.Patch(facecolor=colors['M2'], alpha=0.9, edgecolor=colors['M2'],
                   linewidth=0.5, label='S2 (Stochasticity)'),
    mpatches.Patch(facecolor=colors['M3'], alpha=0.9, edgecolor=colors['M3'],
                   linewidth=0.5, label='S3 (Reasoning)'),
    mpatches.Patch(facecolor='#E8E8E8', alpha=0.8, edgecolor='#D0D0D0',
                   linewidth=0.5, label='Triggered'),
    mpatches.Patch(facecolor='#666666', alpha=0.9, edgecolor='#666666',
                   linewidth=0.5, label='Detected'),
]
ax_d.legend(handles=legend_d, loc='upper center', ncol=5, fontsize=5.5,
            bbox_to_anchor=(0.5, 1.05), frameon=False, columnspacing=0.8, handletextpad=0.4)

# ── Save ──────────────────────────────────────────────────────────────────────
output_png = os.path.join(SCRIPT_DIR, 'Figure3.png')
output_pdf = os.path.join(SCRIPT_DIR, 'Figure3.pdf')
plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Figure saved: {output_png}")
print(f"Figure saved: {output_pdf}")
