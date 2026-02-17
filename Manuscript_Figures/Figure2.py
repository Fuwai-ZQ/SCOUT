"""
Figure 2 – Nature Medicine-style 3×3 composite figure.
Rows: M1 (Model Heterogeneity), M2 (Stochastic Inconsistency), M3 (Reasoning Critique).
Columns: Scatter (review rate vs coverage), Efficiency boxplot, Accuracy improvement boxplot.
Unified colorbar; statistical tests use LMM (n≥30) or t-test fallback.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats
import statsmodels.formula.api as smf
import re
import warnings

warnings.filterwarnings("ignore")

# ── Path configuration (relative to this script) ──────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # repository root

CHD_DATA = os.path.join(BASE_DIR, "冠心病分型", "AIM_CHD_SMART_CHD", "回顾性研究")
GOLD_FILE = os.path.join(CHD_DATA, "df_type_gold.xlsx")
WORK_DIR = os.path.join(CHD_DATA, "data_LLM_analyzed", "S1")
DATA_ROOT_DIR = os.path.join(CHD_DATA, "data_LLM_analyzed", "S3")
DIR_RUN1 = os.path.join(CHD_DATA, "data_LLM_analyzed", "S2", "run_1")
DIR_RUN2 = os.path.join(CHD_DATA, "data_LLM_analyzed", "S2", "run_2")

LMM_MIN_GROUPS = 30  # threshold for LMM vs t-test

# ── Nature color scheme ───────────────────────────────────────────────────────
NATURE_COLORS = {
    'scatter_low': '#3B4992', 'scatter_high': '#EE0000',
    'box_efficiency': '#4DBBD5', 'box_accuracy': '#E64B35',
    'baseline': '#666666', 'points': '#8FA4B0', 'text': '#000000'
}

NATURE_CMAP = LinearSegmentedColormap.from_list(
    'nature', ['#3B4992', '#4DBBD5', '#00A087', '#F39B7F', '#E64B35'])


# ── Utility functions ─────────────────────────────────────────────────────────
def clean_text(x):
    if pd.isna(x) or str(x).strip() == "":
        return np.nan
    return str(x).strip().upper()


def robust_read_csv(path):
    for enc in ('utf-8-sig', 'gb18030', 'utf-8'):
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')
            return df
        except Exception:
            continue
    raise IOError(f"Cannot read {path}")


def find_all_files(root_dir, pattern_regex):
    try:
        regex = re.compile(pattern_regex, re.IGNORECASE)
        matches = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if regex.search(file):
                    matches.append(os.path.join(root, file))
        return matches
    except Exception:
        return []


def simplify_name(name):
    name = os.path.basename(name).replace(".csv", "")
    name = re.sub(r"cases_(labeled|checked_results)_", "", name)
    name = re.sub(r"withPrompt_?", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_?withPrompt", "", name, flags=re.IGNORECASE)
    name = re.sub(r"using_", "", name, flags=re.IGNORECASE)
    name = re.sub(r"Ds_?v?", "DS", name, flags=re.IGNORECASE)
    return name.strip("_")


def setup_nature_style():
    """Nature Medicine style: Arial, 5–7 pt fonts, minimal decoration."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
        'font.size': 6, 'axes.linewidth': 0.5,
        'axes.labelsize': 6, 'axes.titlesize': 7,
        'xtick.labelsize': 5, 'ytick.labelsize': 5,
        'xtick.major.width': 0.4, 'ytick.major.width': 0.4,
        'xtick.major.size': 2, 'ytick.major.size': 2,
        'legend.fontsize': 5, 'legend.frameon': False,
        'figure.dpi': 300, 'savefig.dpi': 600,
        'axes.spines.top': True, 'axes.spines.right': True,
        'mathtext.default': 'regular',
    })


# ── Strategy 1: M1 (Dual-Model Heterogeneity) ────────────────────────────────
def process_dual_model():
    print("\nProcessing M1: Dual-Model...")
    if not os.path.exists(GOLD_FILE):
        print("  Gold file not found, skipping M1")
        return None

    df_gold = pd.read_excel(GOLD_FILE)
    df_gold['病案号'] = df_gold['病案号'].astype(str)
    df_gold['Gold_Label'] = df_gold['冠心病分型'].apply(clean_text)
    df_gold = df_gold[['病案号', 'Gold_Label']].dropna()

    model_files = find_all_files(WORK_DIR, r"cases_labeled_.*\.csv$")
    if not model_files:
        print("  No model files found, skipping M1")
        return None

    model_data, model_accs = {}, {}
    for fp in model_files:
        try:
            tmp = robust_read_csv(fp)
            tmp['病案号'] = tmp['病案号'].astype(str)
            tmp['Pred'] = tmp['final_label'].apply(clean_text)
            model_id = simplify_name(fp)
            merged = df_gold.merge(tmp[['病案号', 'Pred']], on='病案号', how='inner')
            if len(merged) == 0:
                continue
            model_accs[model_id] = (merged['Pred'] == merged['Gold_Label']).mean()
            model_data[model_id] = tmp[['病案号', 'Pred']]
        except Exception:
            continue

    valid_main = [m for m, acc in model_accs.items() if acc > 0.90]
    if not valid_main:
        print("  No valid main models, skipping M1")
        return None
    print(f"  Main models: {len(valid_main)}, Auxiliary models: {len(model_data)}")

    results = []
    for main_model in valid_main:
        main_df = model_data[main_model]
        for aux_model in model_data:
            if main_model == aux_model:
                continue
            aux_df = model_data[aux_model]
            merged = df_gold.merge(
                main_df.rename(columns={'Pred': 'Main_Pred'}), on='病案号', how='inner'
            ).merge(
                aux_df.rename(columns={'Pred': 'Aux_Pred'}), on='病案号', how='left')
            if len(merged) == 0:
                continue
            n = len(merged)
            is_main_error = (merged['Main_Pred'] != merged['Gold_Label']).fillna(True)
            is_flagged = (merged['Main_Pred'] != merged['Aux_Pred']).fillna(True)
            both_na = merged['Main_Pred'].isna() & merged['Aux_Pred'].isna()
            is_flagged = is_flagged & ~both_na

            TP = (is_main_error & is_flagged).sum()
            FN = (is_main_error & ~is_flagged).sum()
            FP = (~is_main_error & is_flagged).sum()
            review_rate = (TP + FP) / n
            coverage_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
            baseline_acc = model_accs[main_model]

            results.append({
                'Main_Model': main_model, 'Checker_Model': aux_model,
                'Baseline_Acc': baseline_acc,
                'Review_Rate': review_rate, 'Coverage_Rate': coverage_rate,
                'Hybrid_Acc': 1 - (FN / n),
                'Acc_Improvement': (1 - FN / n) - baseline_acc,
                'Efficiency': coverage_rate / (review_rate + 1e-5)
            })

    if not results:
        return None
    df_result = pd.DataFrame(results)
    print(f"  Valid data points: {len(df_result)}")
    return df_result


# ── Strategy 2: M2 (Stochastic Self-Check) ───────────────────────────────────
def process_self_check():
    print("\nProcessing M2: Self-Check...")
    if not os.path.exists(GOLD_FILE) or not os.path.exists(DIR_RUN2):
        print("  Required files not found, skipping M2")
        return None

    df_gold = pd.read_excel(GOLD_FILE)
    df_gold['病案号'] = df_gold['病案号'].astype(str)
    df_gold['Gold_Label'] = df_gold['冠心病分型'].apply(clean_text)
    df_gold = df_gold[['病案号', 'Gold_Label']].dropna()

    files_run2 = find_all_files(DIR_RUN2, r"cases_labeled_.*\.csv$")
    results = []

    for file2_path in files_run2:
        filename = os.path.basename(file2_path)
        file1_path = os.path.join(DIR_RUN1, filename)
        if not os.path.exists(file1_path):
            continue
        model_name = re.sub(r"cases_labeled_", "", filename).replace(".csv", "")

        try:
            df1 = robust_read_csv(file1_path)
            df2 = robust_read_csv(file2_path)
            if 'final_label' not in df1.columns or 'final_label' not in df2.columns:
                continue
            df1['病案号'] = df1['病案号'].astype(str)
            df1['Pred1'] = df1['final_label'].apply(clean_text)
            df2['病案号'] = df2['病案号'].astype(str)
            df2['Pred2'] = df2['final_label'].apply(clean_text)

            merged = df_gold.merge(df1[['病案号', 'Pred1']], on='病案号', how='inner')
            merged = merged.merge(df2[['病案号', 'Pred2']], on='病案号', how='inner')
            if len(merged) == 0:
                continue

            n = len(merged)
            baseline_acc = (merged['Pred1'] == merged['Gold_Label']).mean()
            if baseline_acc <= 0.90:
                continue

            is_main_error = (merged['Pred1'] != merged['Gold_Label']).fillna(True)
            is_flagged = (merged['Pred1'] != merged['Pred2']).fillna(True)
            both_na = merged['Pred1'].isna() & merged['Pred2'].isna()
            is_flagged = is_flagged & ~both_na

            TP = (is_main_error & is_flagged).sum()
            FN = (is_main_error & ~is_flagged).sum()
            FP = (~is_main_error & is_flagged).sum()
            review_rate = (TP + FP) / n
            coverage_rate = TP / (TP + FN) if (TP + FN) > 0 else 0

            results.append({
                'Model': model_name, 'Baseline_Acc': baseline_acc,
                'Review_Rate': review_rate, 'Coverage_Rate': coverage_rate,
                'Hybrid_Acc': 1 - (FN / n),
                'Acc_Improvement': (1 - FN / n) - baseline_acc,
                'Efficiency': coverage_rate / (review_rate + 1e-5)
            })
        except Exception:
            continue

    if not results:
        return None
    df_result = pd.DataFrame(results)
    print(f"  Valid data points: {len(df_result)}")
    return df_result


# ── Strategy 3: M3 (Reasoning Critique) ──────────────────────────────────────
def process_cot():
    print("\nProcessing M3: Reasoning Critique...")
    if not os.path.exists(GOLD_FILE) or not os.path.exists(DATA_ROOT_DIR):
        print("  Required files not found, skipping M3")
        return None

    df_gold = pd.read_excel(GOLD_FILE)
    df_gold['病案号'] = df_gold['病案号'].astype(str)
    df_gold['Gold_Label'] = df_gold['冠心病分型'].apply(clean_text)
    df_gold = df_gold[['病案号', 'Gold_Label']].dropna()

    all_files = find_all_files(DATA_ROOT_DIR, r"\.csv$")
    results = []
    safe_words = ["不存在", "一致", "无矛盾", "None", "No", "Pass", "Consistent"]

    for filepath in all_files:
        filename = os.path.basename(filepath)
        match_main = re.search(r"cases_checked_results_(.+?)_using_", filename)
        match_checker = re.search(r"_using_(.+?)\.csv", filename)
        if not match_main or not match_checker:
            continue

        main_name = match_main.group(1)
        checker_name = match_checker.group(1)

        try:
            df = robust_read_csv(filepath)
            if 'audit_result' in df.columns:
                df.rename(columns={'audit_result': 'has_contradiction'}, inplace=True)
            if 'original_label' in df.columns:
                df.rename(columns={'original_label': 'final_label'}, inplace=True)
            if 'final_label' not in df.columns:
                continue
            if 'has_contradiction' not in df.columns:
                df['has_contradiction'] = 'NA'

            df['病案号'] = df['病案号'].astype(str)
            df['Main_Pred'] = df['final_label'].apply(clean_text)

            merged = df_gold.merge(
                df[['病案号', 'Main_Pred', 'has_contradiction']], on='病案号', how='inner')
            if len(merged) == 0:
                continue

            n = len(merged)
            baseline_acc = (merged['Main_Pred'] == merged['Gold_Label']).mean()
            if baseline_acc <= 0.90:
                continue

            is_error = (merged['Main_Pred'] != merged['Gold_Label']).fillna(True)

            def check_flagged(val):
                val_str = str(val).strip()
                return not any(s in val_str for s in safe_words)

            is_flagged = merged['has_contradiction'].apply(check_flagged)

            TP = (is_error & is_flagged).sum()
            FN = (is_error & ~is_flagged).sum()
            FP = (~is_error & is_flagged).sum()
            review_rate = (TP + FP) / n
            coverage_rate = TP / (TP + FN) if (TP + FN) > 0 else 0

            results.append({
                'Main_Model': main_name, 'Checker_Model': checker_name,
                'Baseline_Acc': baseline_acc,
                'Review_Rate': review_rate, 'Coverage_Rate': coverage_rate,
                'Hybrid_Acc': 1 - (FN / n),
                'Acc_Improvement': (1 - FN / n) - baseline_acc,
                'Efficiency': coverage_rate / (review_rate + 1e-5)
            })
        except Exception:
            continue

    if not results:
        return None
    df_result = pd.DataFrame(results)
    print(f"  Valid data points: {len(df_result)}")
    return df_result


# ── Statistical analysis ──────────────────────────────────────────────────────
def calc_stats_efficiency(data_efficiency, df_full=None, group_col='Main_Model', reference=1):
    """Efficiency test: LMM if n≥LMM_MIN_GROUPS, else t-test, vs reference=1."""
    values = np.array(data_efficiency)
    n = len(values)
    if n < 2:
        return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'p_value': np.nan, 'label': 'N/A', 'method': 'N/A'}

    values_vs_ref = values - reference
    mean_val = np.mean(values)
    use_ttest = n < LMM_MIN_GROUPS or df_full is None

    if not use_ttest and df_full is not None:
        try:
            df_lmm = df_full.copy()
            df_lmm['Eff_vs_Ref'] = df_lmm['Efficiency'] - reference
            model = smf.mixedlm("Eff_vs_Ref ~ 1", df_lmm, groups=df_lmm[group_col])
            res = model.fit(disp=False)
            ci = res.conf_int().loc['Intercept']
            pval = res.pvalues['Intercept']
            if np.isnan(pval) or np.isnan(ci[0]) or ci[0] == ci[1]:
                raise ValueError("Invalid LMM results")
            ci_lower, ci_upper = ci[0] + reference, ci[1] + reference
            p_str = "P < 0.001" if pval < 0.001 else f"P = {pval:.3f}"
            return {'mean': mean_val, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
                    'p_value': pval, 'method': f'LMM (n={n})',
                    'label': f"Mean: {mean_val:.2f}\n95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]\n{p_str}"}
        except Exception:
            use_ttest = True

    t_res = stats.ttest_1samp(values_vs_ref, 0)
    sem = stats.sem(values_vs_ref)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lower = mean_val - t_crit * sem
    ci_upper = mean_val + t_crit * sem
    p_str = "P < 0.001" if t_res.pvalue < 0.001 else f"P = {t_res.pvalue:.3f}"
    return {'mean': mean_val, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
            'p_value': t_res.pvalue, 'method': f't-test (n={n})',
            'label': f"Mean: {mean_val:.2f}\n95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]\n{p_str}"}


def calc_stats_accuracy(data_acc_improvement, df_full=None, group_col='Main_Model'):
    """Accuracy improvement test: LMM if n≥LMM_MIN_GROUPS, else t-test, vs 0."""
    values = np.array(data_acc_improvement) * 100
    n = len(values)
    if n < 2:
        return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'p_value': np.nan, 'label': 'N/A', 'method': 'N/A'}

    mean_val = np.mean(values)
    use_ttest = n < LMM_MIN_GROUPS or df_full is None

    if not use_ttest and df_full is not None:
        try:
            df_lmm = df_full.copy()
            df_lmm['Acc_Improve_Pct'] = df_lmm['Acc_Improvement'] * 100
            model = smf.mixedlm("Acc_Improve_Pct ~ 1", df_lmm, groups=df_lmm[group_col])
            res = model.fit(disp=False)
            coef = res.params['Intercept']
            ci = res.conf_int().loc['Intercept']
            pval = res.pvalues['Intercept']
            if np.isnan(pval) or np.isnan(ci[0]) or ci[0] == ci[1]:
                raise ValueError("Invalid LMM results")
            p_str = "P < 0.001" if pval < 0.001 else f"P = {pval:.3f}"
            return {'mean': coef, 'ci_lower': ci[0], 'ci_upper': ci[1],
                    'p_value': pval, 'method': f'LMM (n={n})',
                    'label': f"Mean: +{coef:.1f}%\n95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]\n{p_str}"}
        except Exception:
            use_ttest = True

    t_res = stats.ttest_1samp(values, 0)
    sem = stats.sem(values)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lower = mean_val - t_crit * sem
    ci_upper = mean_val + t_crit * sem
    p_str = "P < 0.001" if t_res.pvalue < 0.001 else f"P = {t_res.pvalue:.3f}"
    return {'mean': mean_val, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
            'p_value': t_res.pvalue, 'method': f't-test (n={n})',
            'label': f"Mean: +{mean_val:.1f}%\n95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]\n{p_str}"}


# ── Plot functions ────────────────────────────────────────────────────────────
def plot_scatter_nature(data, ax, panel_label='a', norm=None, show_colorbar=False):
    ax.plot([0, 1], [0, 1], linestyle='--', color=NATURE_COLORS['baseline'],
            linewidth=0.4, zorder=1)
    scatter = ax.scatter(
        data['Review_Rate'], data['Coverage_Rate'], c=data['Hybrid_Acc'],
        cmap=NATURE_CMAP, norm=norm, s=15, alpha=0.85,
        edgecolors='white', linewidths=0.2, zorder=2)

    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.02)
        cbar.ax.set_title('Final\nAccuracy', fontsize=5, pad=2)
        cbar.ax.tick_params(labelsize=4.5, width=0.3, length=1.5)
        cbar.outline.set_linewidth(0.3)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('Review rate', fontsize=6)
    ax.set_ylabel('Error coverage', fontsize=6)
    ax.set_aspect('equal')
    ax.grid(True, color='grey', alpha=0.2, linewidth=0.25)
    ax.text(-0.18, 1.08, panel_label, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='bottom', ha='left')
    return scatter


def plot_box_efficiency_nature(data, stats_label, ax, panel_label='b', stats_position='top'):
    ax.axhline(y=1, linestyle='--', color=NATURE_COLORS['baseline'], linewidth=0.4)
    ax.text(1.65, 1, 'Random\nDeferral', ha='left', va='center', fontsize=4.5,
            color=NATURE_COLORS['baseline'], style='italic')

    bp = ax.boxplot(data['Efficiency'].dropna(), widths=0.5,
                    patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor(NATURE_COLORS['box_efficiency'])
    bp['boxes'][0].set_alpha(0.7); bp['boxes'][0].set_linewidth(0.3)
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('black'); item.set_linewidth(0.3)

    y_data = data['Efficiency'].dropna().values
    x_jitter = np.random.normal(1, 0.08, len(y_data))
    ax.scatter(x_jitter, y_data, s=8, alpha=0.7,
               facecolors=NATURE_COLORS['points'], edgecolors='white', linewidths=0.2)

    ax.set_ylim(0, 10); ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel('Efficiency ratio', fontsize=6)
    ax.set_xticks([]); ax.set_xlim(0.3, 1.7)
    ax.yaxis.grid(True, color='grey', alpha=0.2, linewidth=0.25)

    if stats_label:
        y_pos = 0.7 if stats_position == 'bottom' else 9.3
        va = 'bottom' if stats_position == 'bottom' else 'top'
        ax.text(1, y_pos, stats_label, ha='center', va=va, fontsize=5,
                linespacing=0.85, family='Arial')

    ax.text(-0.18, 1.08, panel_label, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='bottom', ha='left')


def plot_box_accuracy_nature(data, stats_label, ax, panel_label='c'):
    acc_pct = data['Acc_Improvement'].dropna() * 100
    ax.axhline(y=0, linestyle='--', color=NATURE_COLORS['baseline'], linewidth=0.4)

    bp = ax.boxplot(acc_pct, widths=0.5, patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor(NATURE_COLORS['box_accuracy'])
    bp['boxes'][0].set_alpha(0.7); bp['boxes'][0].set_linewidth(0.3)
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('black'); item.set_linewidth(0.3)

    y_data = acc_pct.values
    x_jitter = np.random.normal(1, 0.08, len(y_data))
    ax.scatter(x_jitter, y_data, s=8, alpha=0.7,
               facecolors=NATURE_COLORS['points'], edgecolors='white', linewidths=0.2)

    ax.set_ylim(0, 10); ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel('Accuracy improvement (%)', fontsize=6)
    ax.set_xticks([]); ax.set_xlim(0.3, 1.7)
    ax.yaxis.grid(True, color='grey', alpha=0.2, linewidth=0.25)

    if stats_label:
        ax.text(1, 9.3, stats_label, ha='center', va='top', fontsize=5,
                linespacing=0.85, family='Arial')

    ax.text(-0.18, 1.08, panel_label, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='bottom', ha='left')


# ── Main figure generation ────────────────────────────────────────────────────
def generate_nature_figure():
    print("=" * 60)
    print("Generating Nature-style Figure 2")
    print("=" * 60)

    setup_nature_style()

    df_m1 = process_dual_model()
    df_m2 = process_self_check()
    df_m3 = process_cot()

    has_m1 = df_m1 is not None and len(df_m1) > 0
    has_m2 = df_m2 is not None and len(df_m2) > 0
    has_m3 = df_m3 is not None and len(df_m3) > 0

    if not (has_m1 or has_m2 or has_m3):
        print("\nError: no valid data available")
        return None

    # Unified colorbar range
    all_acc = []
    for df in [df_m1, df_m2, df_m3]:
        if df is not None:
            all_acc.extend(df['Hybrid_Acc'].dropna().tolist())

    if all_acc:
        margin = (max(all_acc) - min(all_acc)) * 0.05
        vmin = max(0, min(all_acc) - margin)
        vmax = min(1, max(all_acc) + margin)
    else:
        vmin, vmax = 0.92, 0.99

    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(3, 3, figsize=(7.08, 7.08))
    row_labels = ['Model\nHeterogeneity', 'Stochastic\nInconsistency', 'Reasoning\nCritique']

    # Row 1: M1
    if has_m1:
        df_m1_indep = df_m1.drop_duplicates(subset=['Main_Model', 'Checker_Model'])
        stats_eff_1 = calc_stats_efficiency(df_m1_indep['Efficiency'].dropna().values,
                                            df_full=df_m1_indep, group_col='Main_Model')
        stats_acc_1 = calc_stats_accuracy(df_m1_indep['Acc_Improvement'].dropna().values,
                                          df_full=df_m1_indep, group_col='Main_Model')
        plot_scatter_nature(df_m1, axes[0, 0], 'a', norm=norm)
        plot_box_efficiency_nature(df_m1, stats_eff_1['label'], axes[0, 1], 'b')
        plot_box_accuracy_nature(df_m1, stats_acc_1['label'], axes[0, 2], 'c')
    else:
        for j in range(3):
            axes[0, j].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[0, j].transAxes)
            axes[0, j].set_xticks([]); axes[0, j].set_yticks([])

    # Row 2: M2
    if has_m2:
        df_m2_indep = df_m2.drop_duplicates(subset=['Model'])
        stats_eff_2 = calc_stats_efficiency(df_m2_indep['Efficiency'].dropna().values,
                                            df_full=df_m2_indep, group_col='Model')
        stats_acc_2 = calc_stats_accuracy(df_m2_indep['Acc_Improvement'].dropna().values,
                                          df_full=df_m2_indep, group_col='Model')
        plot_scatter_nature(df_m2, axes[1, 0], 'd', norm=norm)
        plot_box_efficiency_nature(df_m2, stats_eff_2['label'], axes[1, 1], 'e', stats_position='bottom')
        plot_box_accuracy_nature(df_m2, stats_acc_2['label'], axes[1, 2], 'f')
    else:
        for j in range(3):
            axes[1, j].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[1, j].transAxes)
            axes[1, j].set_xticks([]); axes[1, j].set_yticks([])

    # Row 3: M3
    if has_m3:
        df_m3_indep = df_m3.drop_duplicates(subset=['Main_Model', 'Checker_Model'])
        stats_eff_3 = calc_stats_efficiency(df_m3_indep['Efficiency'].dropna().values,
                                            df_full=df_m3_indep, group_col='Main_Model')
        stats_acc_3 = calc_stats_accuracy(df_m3_indep['Acc_Improvement'].dropna().values,
                                          df_full=df_m3_indep, group_col='Main_Model')
        plot_scatter_nature(df_m3, axes[2, 0], 'g', norm=norm)
        plot_box_efficiency_nature(df_m3, stats_eff_3['label'], axes[2, 1], 'h')
        plot_box_accuracy_nature(df_m3, stats_acc_3['label'], axes[2, 2], 'i')
    else:
        for j in range(3):
            axes[2, j].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[2, j].transAxes)
            axes[2, j].set_xticks([]); axes[2, j].set_yticks([])

    # Row labels
    for i, label in enumerate(row_labels):
        axes[i, 0].text(-0.45, 0.5, label, transform=axes[i, 0].transAxes,
                        fontsize=7, fontweight='bold', va='center', ha='center', rotation=90)

    plt.tight_layout()
    plt.subplots_adjust(left=0.14, right=0.88, wspace=0.35, hspace=0.25)

    # Unified colorbar
    cbar_ax = fig.add_axes([0.91, 0.25, 0.015, 0.5])
    sm = ScalarMappable(cmap=NATURE_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_title('Final\nAccuracy', fontsize=5, pad=3)
    cbar.ax.tick_params(labelsize=4.5, width=0.3, length=1.5)
    cbar.outline.set_linewidth(0.3)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))

    output_png = os.path.join(SCRIPT_DIR, "Figure2.png")
    output_pdf = os.path.join(SCRIPT_DIR, "Figure2.pdf")
    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_pdf, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nFigure saved: {output_png}")
    print(f"Figure saved: {output_pdf}")
    return output_png, output_pdf


if __name__ == "__main__":
    generate_nature_figure()
