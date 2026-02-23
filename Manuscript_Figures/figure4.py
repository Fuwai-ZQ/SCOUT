"""
Figure 4 – Clinical validation 3×2 composite figure (Nature Medicine style).
Panel a: Audit efficiency (time per case).
Panel b: Cost-benefit analysis by physician seniority.
Panel c: Overall accuracy comparison (control vs experiment).
Panel d: Stratified accuracy by physician seniority.
Panel e: Positive interaction outcomes.
Panel f: Negative interaction outcomes.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Nature style configuration ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.linewidth': 0.8,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    'control': '#B0B0B0', 'experiment': '#E64B35',
    'ai_compute': '#3C5488', 'ai_baseline': '#4DBBD5',
    'correct_accept': '#2166AC', 'successful_corr': '#1B7837',
    'incorrect_override': '#B2182B', 'automation_bias': '#E66101',
    'ineffective_corr': '#7B3294', 'silent_failure': '#878787',
}

# ── Create figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 9.5))
gs = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.35,
                      left=0.09, right=0.97, top=0.96, bottom=0.05)

# ── Panel A: Time per case ────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

time_manual = np.array([100, 40, 60, 36, 45, 45, 20]) / 54
time_ai = np.array([40, 21, 26, 17, 15, 17, 10]) / 56
diff = time_manual - time_ai
n = len(diff)

stat_wilcoxon, p_val_time = stats.wilcoxon(time_manual, time_ai, alternative='two-sided')

# Bootstrap CI for mean difference
np.random.seed(42)
boot_means = [np.mean(diff[np.random.choice(n, n, replace=True)]) for _ in range(10000)]
ci_lower = np.percentile(boot_means, 2.5)
ci_upper = np.percentile(boot_means, 97.5)
mean_diff = np.mean(diff)
reduction_pct = mean_diff / np.mean(time_manual) * 100

bp = ax_a.boxplot([time_manual, time_ai], positions=[0, 1], widths=0.5,
                  patch_artist=True, showfliers=False)
bp['boxes'][0].set_facecolor(COLORS['control']); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(COLORS['experiment']); bp['boxes'][1].set_alpha(0.7)
for box in bp['boxes']:
    box.set_edgecolor('black'); box.set_linewidth(0.8)
for median in bp['medians']:
    median.set_color('black'); median.set_linewidth(1.2)
for w in bp['whiskers']:
    w.set_linewidth(0.8)
for c in bp['caps']:
    c.set_linewidth(0.8)

for i in range(len(time_manual)):
    ax_a.plot([0, 1], [time_manual[i], time_ai[i]], color='gray', alpha=0.4, linewidth=0.6, zorder=1)

np.random.seed(42)
jitter = 0.08
ax_a.scatter(np.random.uniform(-jitter, jitter, len(time_manual)),
             time_manual, color=COLORS['control'], s=25, alpha=0.6,
             edgecolor='white', linewidth=0.5, zorder=3)
ax_a.scatter(1 + np.random.uniform(-jitter, jitter, len(time_ai)),
             time_ai, color=COLORS['experiment'], s=25, alpha=0.6,
             edgecolor='white', linewidth=0.5, zorder=3)

y_max = max(time_manual.max(), time_ai.max())
ax_a.text(0, time_manual.max() + 0.08, f'{np.mean(time_manual):.2f}',
          ha='center', va='bottom', fontsize=8, fontweight='bold')
ax_a.text(1, time_ai.max() + 0.08, f'{np.mean(time_ai):.2f}',
          ha='center', va='bottom', fontsize=8, fontweight='bold')

bracket_y = y_max * 1.35
ax_a.plot([0, 0, 1, 1], [bracket_y, bracket_y + 0.05, bracket_y + 0.05, bracket_y],
          color='black', linewidth=0.8)
p_text = f'P = {p_val_time:.3f}' if p_val_time >= 0.001 else 'P < 0.001'
ax_a.text(0.5, bracket_y + 0.08, p_text, ha='center', va='bottom', fontsize=8, fontweight='bold')
ax_a.text(0.5, bracket_y - 0.02,
          f'Δ {mean_diff:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]\n↓ {reduction_pct:.1f}%',
          ha='center', va='top', fontsize=6.5, linespacing=1.2)

ax_a.set_xticks([0, 1])
ax_a.set_xticklabels(['Control\n(Manual review)', 'Experiment\n(SCOUT-assisted)'])
ax_a.set_ylabel('Time per case (min)')
ax_a.set_ylim(0, y_max * 1.85)
ax_a.set_title('a', loc='left', fontweight='bold', fontsize=12)

# ── Panel B: Cost-benefit analysis ───────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

scenarios = ['Junior\nResidents', 'Senior\nResidents', 'Attending\nPhysicians']
wage_rates = [1.3, 1.7, 2.5]  # CNY per minute
avg_time_manual = np.mean(time_manual)
avg_time_ai = np.mean(time_ai)

manual_costs = [avg_time_manual * w for w in wage_rates]
ai_labor_costs = [avg_time_ai * w for w in wage_rates]
ai_compute_cost = 0.16
ai_total_costs = [l + ai_compute_cost for l in ai_labor_costs]
rois = [(m - l) / ai_compute_cost for m, l in zip(manual_costs, ai_labor_costs)]

bar_width = 0.35
index = np.arange(len(scenarios))

ax_b.bar(index, manual_costs, bar_width, label='Control (labor review)',
         color=COLORS['control'], edgecolor='black', linewidth=0.5, alpha=0.85)
ax_b.bar(index + bar_width, ai_labor_costs, bar_width, label='Experiment (labor review)',
         color=COLORS['experiment'], edgecolor='black', linewidth=0.5, alpha=0.85)
ax_b.bar(index + bar_width, [ai_compute_cost] * 3, bar_width, bottom=ai_labor_costs,
         label='Experiment (AI review)', color=COLORS['ai_compute'],
         edgecolor='black', linewidth=0.5, alpha=0.85)

for i, (m, a) in enumerate(zip(manual_costs, ai_total_costs)):
    ax_b.text(i, m + 0.08, f'¥{m:.2f}', ha='center', va='bottom', fontsize=7)
    ax_b.text(i + bar_width, a + 0.08, f'¥{a:.2f}', ha='center', va='bottom', fontsize=7)

max_height = max(manual_costs)
for i in range(len(scenarios)):
    ax_b.text(index[i] + bar_width / 2, max_height * 1.15,
              f'ROI: {rois[i]:.1f}×', ha='center', fontsize=7, color='#2E7D32', fontweight='bold')

ax_b.set_xticks(index + bar_width / 2); ax_b.set_xticklabels(scenarios)
ax_b.set_ylabel('Cost per case (CNY)')
ax_b.set_ylim(0, max_height * 1.45)
ax_b.legend(loc='upper left', fontsize=6)
ax_b.set_title('b', loc='left', fontweight='bold', fontsize=12)

# ── Panel C: Overall accuracy ────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

acc_control, acc_experiment = 0.894, 0.908
ai_baseline_experiment, ai_baseline_control = 0.893, 0.926
groups = ['Control\n(Manual review)', 'Experiment\n(SCOUT-assisted)']
accuracies = [acc_control, acc_experiment]

bars = ax_c.bar(groups, accuracies, width=0.5,
                color=[COLORS['control'], COLORS['experiment']],
                edgecolor='black', linewidth=0.8, alpha=0.85)

bar_ctrl, bar_exp = bars[0], bars[1]
ax_c.hlines(y=ai_baseline_control, xmin=bar_ctrl.get_x() - 0.1,
            xmax=bar_ctrl.get_x() + bar_ctrl.get_width() + 0.1,
            colors=COLORS['ai_baseline'], linestyles='--', linewidth=1.5)
ax_c.hlines(y=ai_baseline_experiment, xmin=bar_exp.get_x() - 0.1,
            xmax=bar_exp.get_x() + bar_exp.get_width() + 0.1,
            colors=COLORS['ai_baseline'], linestyles='--', linewidth=1.5,
            label='AI baseline accuracy')

for bar, acc in zip(bars, accuracies):
    ax_c.text(bar.get_x() + bar.get_width() / 2, acc + 0.001,
              f'{acc * 100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax_c.text(bar_ctrl.get_x() + bar_ctrl.get_width() / 2, ai_baseline_control - 0.055,
          f'{ai_baseline_control * 100:.1f}%', ha='center', va='top', fontsize=7, color=COLORS['ai_baseline'])
ax_c.text(bar_exp.get_x() + bar_exp.get_width() / 2, ai_baseline_experiment - 0.025,
          f'{ai_baseline_experiment * 100:.1f}%', ha='center', va='top', fontsize=7, color=COLORS['ai_baseline'])

y_bracket = max(accuracies) + 0.025
ax_c.plot([0, 0, 1, 1], [y_bracket - 0.005, y_bracket, y_bracket, y_bracket - 0.005],
          color='black', linewidth=0.8)
ax_c.text(0.5, y_bracket + 0.008, 'P = 0.596', ha='center', fontsize=8, fontweight='bold')

ax_c.set_ylim(0.60, 1.05)
ax_c.set_ylabel('Accuracy')
ax_c.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax_c.set_yticklabels(['60%', '70%', '80%', '90%', '100%'])
ax_c.legend(loc='upper left', fontsize=6)
ax_c.set_title('c', loc='left', fontweight='bold', fontsize=12)

# ── Panel D: Stratified accuracy ─────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

seniority = ['Junior\nResidents', 'Senior\nResidents', 'Attending\nPhysicians']
control_acc = [0.787, 0.914, 0.972]
experiment_acc = [0.857, 0.905, 0.964]
ai_baselines_control = [0.926, 0.926, 0.926]
ai_baselines_experiment = [0.893, 0.893, 0.893]

x = np.arange(len(seniority))
width = 0.35
ax_d.bar(x - width / 2, control_acc, width, label='Control',
         color=COLORS['control'], edgecolor='black', linewidth=0.5, alpha=0.85)
ax_d.bar(x + width / 2, experiment_acc, width, label='Experiment',
         color=COLORS['experiment'], edgecolor='black', linewidth=0.5, alpha=0.85)

for i in range(len(seniority)):
    ax_d.hlines(y=ai_baselines_control[i], xmin=x[i] - width, xmax=x[i],
                colors=COLORS['ai_baseline'], linestyles='--', linewidth=1.2)
    ax_d.hlines(y=ai_baselines_experiment[i], xmin=x[i], xmax=x[i] + width,
                colors=COLORS['ai_baseline'], linestyles='--', linewidth=1.2)

for i, (c, e) in enumerate(zip(control_acc, experiment_acc)):
    ax_d.text(x[i] - width / 2, c + 0.008, f'{c * 100:.1f}%', ha='center', va='bottom', fontsize=6.5)
    ax_d.text(x[i] + width / 2, e + 0.008, f'{e * 100:.1f}%', ha='center', va='bottom', fontsize=6.5)

ax_d.set_xticks(x); ax_d.set_xticklabels(seniority)
ax_d.set_ylabel('Accuracy')
ax_d.set_ylim(0.60, 1.12)
ax_d.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax_d.set_yticklabels(['60%', '70%', '80%', '90%', '100%'])

legend_elements = [
    mpatches.Patch(facecolor=COLORS['control'], edgecolor='black', label='Control'),
    mpatches.Patch(facecolor=COLORS['experiment'], edgecolor='black', label='Experiment'),
    plt.Line2D([0], [0], color=COLORS['ai_baseline'], linestyle='--', linewidth=1.2,
               label='AI baseline accuracy')
]
ax_d.legend(handles=legend_elements, loc='upper left', fontsize=5.5)
ax_d.set_title('d', loc='left', fontweight='bold', fontsize=12)

# ── Panel E: Positive interaction outcomes ────────────────────────────────────
ax_e = fig.add_subplot(gs[2, 0])

physician_levels = ['Junior\nResidents', 'Senior\nResidents', 'Attending\nPhysicians']
x_int = np.arange(len(physician_levels))
marker_size = 7
line_width = 2.0

control_correct_accept = [77.8, 88.9, 91.7]
exp_correct_accept = [83.9, 86.9, 88.4]
control_successful_corr = [0.9, 2.5, 5.6]
exp_successful_corr = [1.8, 3.6, 8.0]

ax_e.plot(x_int, control_correct_accept, color=COLORS['correct_accept'],
          linewidth=line_width, markersize=marker_size, label='Correct Acceptance (Control)',
          marker='o', markerfacecolor='white', markeredgewidth=1.5, linestyle='--', alpha=0.8)
ax_e.plot(x_int, exp_correct_accept, color=COLORS['correct_accept'],
          linewidth=line_width, markersize=marker_size, label='Correct Acceptance (Experiment)',
          marker='o', markerfacecolor=COLORS['correct_accept'], markeredgewidth=1.5, linestyle='-')

ax_e.plot(x_int, control_successful_corr, color=COLORS['successful_corr'],
          linewidth=line_width, markersize=marker_size, label='Successful Correction (Control)',
          marker='s', markerfacecolor='white', markeredgewidth=1.5, linestyle='--', alpha=0.8)
ax_e.plot(x_int, exp_successful_corr, color=COLORS['successful_corr'],
          linewidth=line_width, markersize=marker_size, label='Successful Correction (Experiment)',
          marker='s', markerfacecolor=COLORS['successful_corr'], markeredgewidth=1.5, linestyle='-')

ax_e.set_xticks(x_int); ax_e.set_xticklabels(physician_levels)
ax_e.set_ylabel('Proportion (%)'); ax_e.set_ylim(-5, 105)
ax_e.set_yticks([0, 20, 40, 60, 80, 100])

legend_e = [
    Line2D([0], [0], color=COLORS['correct_accept'], linewidth=2, linestyle='--',
           marker='o', markerfacecolor='white', markeredgewidth=1.5, markersize=5,
           label='Correct Acceptance (Control)', alpha=0.8),
    Line2D([0], [0], color=COLORS['correct_accept'], linewidth=2,
           marker='o', markerfacecolor=COLORS['correct_accept'], markeredgewidth=1.5, markersize=5,
           label='Correct Acceptance (Experiment)'),
    Line2D([0], [0], color=COLORS['successful_corr'], linewidth=2, linestyle='--',
           marker='s', markerfacecolor='white', markeredgewidth=1.5, markersize=5,
           label='Successful Correction (Control)', alpha=0.8),
    Line2D([0], [0], color=COLORS['successful_corr'], linewidth=2,
           marker='s', markerfacecolor=COLORS['successful_corr'], markeredgewidth=1.5, markersize=5,
           label='Successful Correction (Experiment)'),
]
ax_e.legend(handles=legend_e, loc='upper left', fontsize=5.5, frameon=False,
            bbox_to_anchor=(0.02, 0.58))
ax_e.set_title('e', loc='left', fontweight='bold', fontsize=12)
ax_e.text(0.5, 1.02, 'Positive Interaction Outcomes', transform=ax_e.transAxes,
          ha='center', fontsize=9, fontweight='normal')

# ── Panel F: Negative interaction outcomes ────────────────────────────────────
ax_f = fig.add_subplot(gs[2, 1])

control_incorrect_override = [14.8, 3.7, 0.9]
exp_incorrect_override = [5.4, 2.4, 0.9]
control_automation_bias = [6.5, 3.7, 0.9]
exp_automation_bias = [5.4, 4.8, 0.9]
control_ineffective_corr = [0.0, 1.2, 0.9]
exp_ineffective_corr = [1.8, 0.6, 0.0]
exp_silent_failure = [1.8, 1.8, 1.8]

ax_f.plot(x_int, control_incorrect_override, color=COLORS['incorrect_override'],
          linewidth=line_width, markersize=marker_size,
          marker='o', markerfacecolor='white', markeredgewidth=1.5, linestyle='--', alpha=0.8)
ax_f.plot(x_int, exp_incorrect_override, color=COLORS['incorrect_override'],
          linewidth=line_width, markersize=marker_size,
          marker='o', markerfacecolor=COLORS['incorrect_override'], markeredgewidth=1.5, linestyle='-')

ax_f.plot(x_int, control_automation_bias, color=COLORS['automation_bias'],
          linewidth=line_width, markersize=marker_size,
          marker='s', markerfacecolor='white', markeredgewidth=1.5, linestyle='--', alpha=0.8)
ax_f.plot(x_int, exp_automation_bias, color=COLORS['automation_bias'],
          linewidth=line_width, markersize=marker_size,
          marker='s', markerfacecolor=COLORS['automation_bias'], markeredgewidth=1.5, linestyle='-')

ax_f.plot(x_int, control_ineffective_corr, color=COLORS['ineffective_corr'],
          linewidth=line_width, markersize=marker_size,
          marker='^', markerfacecolor='white', markeredgewidth=1.5, linestyle='--', alpha=0.8)
ax_f.plot(x_int, exp_ineffective_corr, color=COLORS['ineffective_corr'],
          linewidth=line_width, markersize=marker_size,
          marker='^', markerfacecolor=COLORS['ineffective_corr'], markeredgewidth=1.5, linestyle='-')

ax_f.plot(x_int, exp_silent_failure, color=COLORS['silent_failure'],
          linewidth=line_width, markersize=marker_size - 1,
          marker='D', markerfacecolor=COLORS['silent_failure'], markeredgewidth=1.5, linestyle='-')

ax_f.set_xticks(x_int); ax_f.set_xticklabels(physician_levels)
ax_f.set_ylabel('Proportion (%)'); ax_f.set_ylim(-1, 18); ax_f.set_yticks([0, 5, 10, 15])

legend_f = [
    Line2D([0], [0], color=COLORS['incorrect_override'], linewidth=2, linestyle='--',
           marker='o', markerfacecolor='white', markeredgewidth=1.5, markersize=5,
           label='Incorrect Override (Control)', alpha=0.8),
    Line2D([0], [0], color=COLORS['incorrect_override'], linewidth=2,
           marker='o', markerfacecolor=COLORS['incorrect_override'], markeredgewidth=1.5, markersize=5,
           label='Incorrect Override (Experiment)'),
    Line2D([0], [0], color=COLORS['automation_bias'], linewidth=2, linestyle='--',
           marker='s', markerfacecolor='white', markeredgewidth=1.5, markersize=5,
           label='Automation Bias (Control)', alpha=0.8),
    Line2D([0], [0], color=COLORS['automation_bias'], linewidth=2,
           marker='s', markerfacecolor=COLORS['automation_bias'], markeredgewidth=1.5, markersize=5,
           label='Automation Bias (Experiment)'),
    Line2D([0], [0], color=COLORS['ineffective_corr'], linewidth=2, linestyle='--',
           marker='^', markerfacecolor='white', markeredgewidth=1.5, markersize=5,
           label='Ineffective Correction (Control)', alpha=0.8),
    Line2D([0], [0], color=COLORS['ineffective_corr'], linewidth=2,
           marker='^', markerfacecolor=COLORS['ineffective_corr'], markeredgewidth=1.5, markersize=5,
           label='Ineffective Correction (Experiment)'),
    Line2D([0], [0], color=COLORS['silent_failure'], linewidth=2,
           marker='D', markerfacecolor=COLORS['silent_failure'], markeredgewidth=1.5, markersize=4,
           label='Silent Failure (Experiment only)'),
]
ax_f.legend(handles=legend_f, loc='upper right', fontsize=5.5, frameon=False)
ax_f.set_title('f', loc='left', fontweight='bold', fontsize=12)
ax_f.text(0.5, 1.02, 'Negative Interaction Outcomes', transform=ax_f.transAxes,
          ha='center', fontsize=9, fontweight='normal')

# ── Save ──────────────────────────────────────────────────────────────────────
output_pdf = os.path.join(SCRIPT_DIR, 'Figure4.pdf')
output_png = os.path.join(SCRIPT_DIR, 'Figure4.png')
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved: {output_pdf}")
print(f"Figure saved: {output_png}")
