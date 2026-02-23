"""
Table 1 – Strategy enumeration and statistical analysis for SCOUT framework.
Computes accuracy, efficiency, review rate, and coverage across all Main × Aux × Checker
combinations using M1 (heterogeneity), M2 (stochasticity), and M3 (reasoning critique).
"""

import os
import pandas as pd
import numpy as np
import re
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ── Path configuration (relative to this script) ──────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # repository root

CHD_DATA = os.path.join(BASE_DIR, "冠心病分型", "AIM_CHD_SMART_CHD", "回顾性研究")
GOLD_FILE = os.path.join(CHD_DATA, "df_type_gold.xlsx")
MAIN_DIR = os.path.join(CHD_DATA, "data_LLM_analyzed", "S1")
CHECK_DIR = os.path.join(CHD_DATA, "data_LLM_analyzed", "S3")
RUN1_DIR = os.path.join(CHD_DATA, "data_LLM_analyzed", "S2", "run_1")
RUN2_DIR = os.path.join(CHD_DATA, "data_LLM_analyzed", "S2", "run_2")

OUTPUT_DIR = SCRIPT_DIR

# ── Threshold for t-test vs LMM ──────────────────────────────────────────────
LMM_MIN_GROUPS = 30


# ── Utility functions ─────────────────────────────────────────────────────────
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


def clean_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return np.nan
    return str(text).strip().upper()


def robust_read_csv(path):
    for enc in ('utf-8-sig', 'gb18030', 'utf-8'):
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')
            return df
        except Exception:
            continue
    raise IOError(f"Cannot read {path}")


def simplify_name(name):
    name = os.path.basename(name).replace(".csv", "")
    name = re.sub(r"cases_(labeled|checked_results)_", "", name)
    name = re.sub(r"withPrompt_?", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_?withPrompt", "", name, flags=re.IGNORECASE)
    name = re.sub(r"using_", "", name, flags=re.IGNORECASE)
    name = re.sub(r"Ds_?v?", "DS", name, flags=re.IGNORECASE)
    return name.strip("_")


def fmt_pct(x, decimals=1):
    return f"{x * 100:.{decimals}f}"


def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def get_independent_obs(df_strat, strategy):
    """De-duplicate to independent observations based on strategy type."""
    if strategy == 'M2 (Self)':
        return df_strat.drop_duplicates(subset=['Main_Model'])
    elif strategy == 'M3 (Rationale)':
        return df_strat.drop_duplicates(subset=['Main_Model', 'Checker_Model'])
    elif strategy in ['M1 (Aux)', 'M1 | M2', 'M1 & M2']:
        return df_strat.drop_duplicates(subset=['Main_Model', 'Aux_Model'])
    else:
        return df_strat.drop_duplicates(subset=['Main_Model', 'Aux_Model', 'Checker_Model'])


# ── 1. Load gold standard ────────────────────────────────────────────────────
print("=== 1. Loading data and computing baseline accuracy ===")
if not os.path.exists(GOLD_FILE):
    raise FileNotFoundError(f"Gold standard file not found: {GOLD_FILE}")

df_gold = pd.read_excel(GOLD_FILE)
df_gold['病案号'] = df_gold['病案号'].astype(str)
df_gold['Gold'] = df_gold['冠心病分型'].apply(clean_text)

initial_count = len(df_gold)
df_gold = df_gold.dropna(subset=['Gold'])
dropped_count = initial_count - len(df_gold)
if dropped_count > 0:
    print(f"Dropped {dropped_count} samples with missing gold labels (remaining: {len(df_gold)})")

# ── 2. Scan all models and compute accuracy ──────────────────────────────────
all_pred_files = find_all_files(MAIN_DIR, r"cases_labeled_.*\.csv")
model_inventory = []
print(f"Scanning {len(all_pred_files)} model files...")

for fp in all_pred_files:
    try:
        tmp = robust_read_csv(fp)
        tmp['病案号'] = tmp['病案号'].astype(str)
        tmp['Pred'] = tmp['final_label'].apply(clean_text)

        chk = df_gold.merge(tmp[['病案号', 'Pred']], on='病案号', how='left')
        chk['Is_Correct'] = (chk['Pred'] == chk['Gold']) & (chk['Pred'].notna())

        acc = chk['Is_Correct'].mean()
        total = len(chk)
        errors = total - chk['Is_Correct'].sum()

        model_inventory.append({
            "name": simplify_name(fp),
            "filename": os.path.basename(fp),
            "path": fp,
            "acc": acc,
            "total": total,
            "errors": errors,
            "df": tmp[['病案号', 'Pred']]
        })
    except Exception as e:
        print(f"Failed to read {os.path.basename(fp)}: {e}")

df_inventory = pd.DataFrame(model_inventory).sort_values("acc", ascending=False)

print("\n========== Initial Accuracy Report ==========")
print(df_inventory[['name', 'acc', 'errors', 'total']].to_string(index=False))
df_inventory.to_excel(os.path.join(OUTPUT_DIR, "Model_Initial_Accuracy.xlsx"), index=False)

# Filter main models (Acc > 90%)
main_candidates = [m for m in model_inventory if m['acc'] > 0.90]
BASELINE_ACC_REF = np.mean([m['acc'] for m in main_candidates])
BASELINE_EFF_REF = 1.0  # full human review

print(f"\nMain models (Acc>90%): {len(main_candidates)}")
print(f"Baseline accuracy reference: {fmt_pct(BASELINE_ACC_REF, 1)}%")
print(f"Baseline efficiency reference: {BASELINE_EFF_REF:.2f} (full review)")
print(f"Available auxiliary models: {len(model_inventory)}")

# ── 3. Preload M2 & M3 resources ─────────────────────────────────────────────
print("\n=== 2. Preloading check resources (M2 & M3) ===")

# M3 (reasoning critique)
m3_resource = {}
check_files = find_all_files(CHECK_DIR, r"cases_checked_.*\.csv")
safe_words = ["不存在", "一致", "无矛盾", "None", "No", "Pass", "Consistent"]

for fp in check_files:
    try:
        fname = os.path.basename(fp)
        match = re.search(r"cases_checked_results_(.+)_using_(.+)\.csv", fname)
        if match:
            main_part = simplify_name(match.group(1))
            checker_part = simplify_name(match.group(2))

            tmp = robust_read_csv(fp)
            if 'audit_result' in tmp.columns:
                tmp.rename(columns={'audit_result': 'has_contradiction'}, inplace=True)

            tmp['病案号'] = tmp['病案号'].astype(str)

            def is_flagged(row):
                c = str(row.get('has_contradiction', ''))
                return not any(s in c for s in safe_words)

            tmp['Flag'] = tmp.apply(is_flagged, axis=1)

            if main_part not in m3_resource:
                m3_resource[main_part] = {}
            m3_resource[main_part][checker_part] = tmp[['病案号', 'Flag']]
    except Exception:
        pass

print(f"Loaded M3 (rationale check) pairs: {sum(len(v) for v in m3_resource.values())}")

# M2 (stochastic self-check)
m2_resource = {}
run2_files = find_all_files(RUN2_DIR, r"cases_labeled_.*\.csv")

for fp in run2_files:
    try:
        name = simplify_name(fp)
        tmp = robust_read_csv(fp)
        tmp['病案号'] = tmp['病案号'].astype(str)
        tmp['Pred'] = tmp['final_label'].apply(clean_text)
        m2_resource[name] = tmp[['病案号', 'Pred']]
    except Exception:
        pass

print(f"Loaded M2 (self-check) files: {len(m2_resource)}")

# ── 4. Full strategy enumeration ─────────────────────────────────────────────
print("\n=== 3. Strategy enumeration (Main x Aux x Checker) ===")
lmm_data = []
detailed_results = {}

for main_model in main_candidates:
    main_name = main_model['name']

    df_base = df_gold.merge(
        main_model['df'].rename(columns={'Pred': 'Main_Pred'}), on='病案号', how='inner')
    df_base['Is_Error'] = (df_base['Main_Pred'] != df_base['Gold']) | (df_base['Main_Pred'].isna())
    total_errors = df_base['Is_Error'].sum()
    total_cases = len(df_base)

    # M2 lookup
    m2_df = m2_resource.get(main_name)
    if m2_df is None:
        for k, v in m2_resource.items():
            if k in main_name or main_name in k:
                m2_df = v
                break

    if m2_df is not None:
        df_base = df_base.merge(m2_df.rename(columns={'Pred': 'Self_Pred'}), on='病案号', how='left')
        df_base['Signal_M2'] = (df_base['Main_Pred'] != df_base['Self_Pred']) | df_base['Self_Pred'].isna()
    else:
        df_base['Signal_M2'] = False

    available_checkers = m3_resource.get(main_name, {})

    for aux_model in model_inventory:
        if aux_model['name'] == main_name:
            continue

        curr = df_base.merge(
            aux_model['df'].rename(columns={'Pred': 'Aux_Pred'}), on='病案号', how='left')
        curr['Signal_M1'] = (curr['Main_Pred'] != curr['Aux_Pred']) | curr['Aux_Pred'].isna()

        checker_iter = available_checkers.items() if available_checkers else [("None", None)]

        for checker_name, m3_df in checker_iter:
            if m3_df is not None:
                curr_m3 = curr.merge(
                    m3_df.rename(columns={'Flag': 'Signal_M3'}), on='病案号', how='left')
                curr_m3['Signal_M3'] = curr_m3['Signal_M3'].fillna(False)
                has_m3 = True
            else:
                curr_m3 = curr.copy()
                curr_m3['Signal_M3'] = False
                has_m3 = False

            S1 = curr_m3['Signal_M1']
            S2 = curr_m3['Signal_M2']
            S3 = curr_m3['Signal_M3']

            strategies = {
                "M1 (Aux)": S1,
                "M2 (Self)": S2,
                "M1 | M2": S1 | S2,
                "M1 & M2": S1 & S2,
            }
            if has_m3:
                strategies.update({
                    "M3 (Rationale)": S3,
                    "M1 | M3": S1 | S3,
                    "M2 | M3": S2 | S3,
                    "All Union": S1 | S2 | S3,
                    "M1 & M3": S1 & S3,
                    "M2 & M3": S2 & S3,
                    "All Intersect": S1 & S2 & S3,
                    "Majority (>=2)": ((S1 & S2) | (S1 & S3) | (S2 & S3)),
                    "(M1 | M2) & M3": (S1 | S2) & S3,
                    "(M1 | M3) & M2": (S1 | S3) & S2,
                    "(M2 | M3) & M1": (S2 | S3) & S1,
                })

            for strat_name, sig_vec in strategies.items():
                review_rate = sig_vec.mean()
                tp = (curr_m3['Is_Error'] & sig_vec).sum()
                coverage = tp / total_errors if total_errors > 0 else 0
                final_acc = 1 - ((total_errors - tp) / total_cases)
                efficiency = coverage / (review_rate + 1e-9)

                lmm_data.append({
                    "Main_Model": main_name,
                    "Aux_Model": aux_model['name'],
                    "Checker_Model": checker_name,
                    "Strategy": strat_name,
                    "Baseline_Acc": main_model['acc'],
                    "Efficiency": efficiency,
                    "Final_Acc": final_acc,
                    "Review_Rate": review_rate,
                    "Coverage": coverage
                })

                key = (main_name, aux_model['name'], checker_name, strat_name)
                detailed_results[key] = {
                    "df": curr_m3.copy(), "S1": S1.copy(), "S2": S2.copy(),
                    "S3": S3.copy(), "sig_vec": sig_vec.copy(),
                    "total_cases": total_cases, "total_errors": total_errors,
                    "tp": tp, "review_rate": review_rate, "coverage": coverage,
                    "final_acc": final_acc, "efficiency": efficiency,
                    "baseline_acc": main_model['acc']
                }

df_res = pd.DataFrame(lmm_data)
df_res = df_res[np.isfinite(df_res['Efficiency'])]
print(f"Complete! Generated {len(df_res)} strategy combinations.")

# ── 5. Statistical analysis (LMM / t-test) ───────────────────────────────────
print("\n=== 4. Running Statistical Analysis (LMM / t-test) ===")
df_res['Group'] = df_res['Main_Model'] + "_" + df_res['Aux_Model'] + "_" + df_res['Checker_Model']
df_res['Acc_vs_Baseline'] = df_res['Final_Acc'] - df_res['Baseline_Acc']
df_res['Eff_vs_Baseline'] = df_res['Efficiency'] - BASELINE_EFF_REF


def run_lmm_acc(df):
    """Test each strategy's accuracy improvement vs 0 using LMM or t-test."""
    from scipy import stats
    results = []

    print("Fitting accuracy models...")
    for strat in df['Strategy'].unique():
        df_strat = df[df['Strategy'] == strat].copy()
        df_indep = get_independent_obs(df_strat, strat)
        values = df_indep['Acc_vs_Baseline'].dropna().values
        n = len(values)

        if n < 2:
            results.append({'Strategy': strat, 'Acc_Coef': values.mean() if n > 0 else np.nan,
                            'Acc_CI_Low': np.nan, 'Acc_CI_High': np.nan,
                            'Acc_P': np.nan, 'Acc_Method': 'N/A'})
            continue

        if n < LMM_MIN_GROUPS:
            t_res = stats.ttest_1samp(values, 0)
            sem = stats.sem(values)
            mean_val = np.mean(values)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            results.append({'Strategy': strat, 'Acc_Coef': mean_val,
                            'Acc_CI_Low': mean_val - t_crit * sem,
                            'Acc_CI_High': mean_val + t_crit * sem,
                            'Acc_P': t_res.pvalue, 'Acc_Method': f't-test (n={n})'})
        else:
            try:
                model = smf.mixedlm("Acc_vs_Baseline ~ 1", df_indep, groups=df_indep["Main_Model"])
                res = model.fit(disp=False)
                ci = res.conf_int().loc['Intercept']
                pval = res.pvalues['Intercept']
                if np.isnan(pval) or np.isnan(ci[0]) or ci[0] == ci[1]:
                    raise ValueError("LMM returned invalid results")
                results.append({'Strategy': strat, 'Acc_Coef': res.params['Intercept'],
                                'Acc_CI_Low': ci[0], 'Acc_CI_High': ci[1],
                                'Acc_P': pval, 'Acc_Method': f'LMM (n={n})'})
            except Exception as e:
                n_groups = df_indep["Main_Model"].nunique()
                print(f"  LMM failed for {strat} accuracy (n={n}, groups={n_groups}): {str(e)[:50]}")
                t_res = stats.ttest_1samp(values, 0)
                sem = stats.sem(values)
                mean_val = np.mean(values)
                t_crit = stats.t.ppf(0.975, df=n - 1)
                results.append({'Strategy': strat, 'Acc_Coef': mean_val,
                                'Acc_CI_Low': mean_val - t_crit * sem,
                                'Acc_CI_High': mean_val + t_crit * sem,
                                'Acc_P': t_res.pvalue,
                                'Acc_Method': f't-test (n={n}, LMM failed)'})

    return pd.DataFrame(results).set_index('Strategy')


def run_lmm_eff(df):
    """Test each strategy's efficiency vs 1 (full review) using LMM or t-test."""
    from scipy import stats
    results = []

    print("Fitting efficiency models...")
    for strat in df['Strategy'].unique():
        df_strat = df[df['Strategy'] == strat].copy()
        df_indep = get_independent_obs(df_strat, strat)
        values = df_indep['Eff_vs_Baseline'].dropna().values
        n = len(values)

        if n < 2:
            results.append({'Strategy': strat, 'Eff_Coef': values.mean() if n > 0 else np.nan,
                            'Eff_CI_Low': np.nan, 'Eff_CI_High': np.nan,
                            'Eff_P': np.nan, 'Eff_Method': 'N/A'})
            continue

        if n < LMM_MIN_GROUPS:
            t_res = stats.ttest_1samp(values, 0)
            sem = stats.sem(values)
            mean_val = np.mean(values)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            results.append({'Strategy': strat, 'Eff_Coef': mean_val,
                            'Eff_CI_Low': mean_val - t_crit * sem,
                            'Eff_CI_High': mean_val + t_crit * sem,
                            'Eff_P': t_res.pvalue, 'Eff_Method': f't-test (n={n})'})
        else:
            try:
                model = smf.mixedlm("Eff_vs_Baseline ~ 1", df_indep, groups=df_indep["Main_Model"])
                res = model.fit(disp=False)
                ci = res.conf_int().loc['Intercept']
                pval = res.pvalues['Intercept']
                if np.isnan(pval) or np.isnan(ci[0]) or ci[0] == ci[1]:
                    raise ValueError("LMM returned invalid results")
                results.append({'Strategy': strat, 'Eff_Coef': res.params['Intercept'],
                                'Eff_CI_Low': ci[0], 'Eff_CI_High': ci[1],
                                'Eff_P': pval, 'Eff_Method': f'LMM (n={n})'})
            except Exception as e:
                n_groups = df_indep["Main_Model"].nunique()
                print(f"  LMM failed for {strat} efficiency (n={n}, groups={n_groups}): {str(e)[:50]}")
                t_res = stats.ttest_1samp(values, 0)
                sem = stats.sem(values)
                mean_val = np.mean(values)
                t_crit = stats.t.ppf(0.975, df=n - 1)
                results.append({'Strategy': strat, 'Eff_Coef': mean_val,
                                'Eff_CI_Low': mean_val - t_crit * sem,
                                'Eff_CI_High': mean_val + t_crit * sem,
                                'Eff_P': t_res.pvalue,
                                'Eff_Method': f't-test (n={n}, LMM failed)'})

    return pd.DataFrame(results).set_index('Strategy')


stats_acc = run_lmm_acc(df_res)
stats_eff = run_lmm_eff(df_res)

# ── 6. Generate final report ─────────────────────────────────────────────────
print("\n=== 5. Generating final report ===")

summary = df_res.groupby("Strategy").agg({
    "Final_Acc": "mean", "Efficiency": "mean", "Review_Rate": "mean",
    "Coverage": "mean", "Acc_vs_Baseline": "mean", "Eff_vs_Baseline": "mean"
}).round(4)

final_report = summary.join(stats_acc, how='left').join(stats_eff, how='left')
final_report = final_report.sort_values("Final_Acc", ascending=False).reset_index()


def format_nature_report(df):
    formatted = df.copy()
    formatted['Accuracy (%)'] = (df['Final_Acc'] * 100).round(1)
    formatted['Δ Accuracy (%)'] = (df['Acc_vs_Baseline'] * 100).round(1)
    formatted['Acc 95% CI'] = df.apply(
        lambda r: f"[{r['Acc_CI_Low'] * 100:.1f}, {r['Acc_CI_High'] * 100:.1f}]"
        if pd.notna(r.get('Acc_CI_Low')) else "", axis=1)
    formatted['Acc P'] = df['Acc_P'].apply(lambda x: fmt_p(x) if pd.notna(x) else "")
    formatted['Acc Method'] = df['Acc_Method']
    formatted['Efficiency'] = df['Efficiency'].round(2)
    formatted['Δ Efficiency'] = df['Eff_vs_Baseline'].round(2)
    # CI is based on (Efficiency - 1); add 1 back for reporting
    formatted['Eff 95% CI'] = df.apply(
        lambda r: f"[{r['Eff_CI_Low'] + 1:.2f}, {r['Eff_CI_High'] + 1:.2f}]"
        if pd.notna(r.get('Eff_CI_Low')) else "", axis=1)
    formatted['Eff P'] = df['Eff_P'].apply(lambda x: fmt_p(x) if pd.notna(x) else "")
    formatted['Eff Method'] = df['Eff_Method']
    formatted['Review Rate (%)'] = (df['Review_Rate'] * 100).round(1)
    formatted['Coverage (%)'] = (df['Coverage'] * 100).round(1)

    pub_cols = [
        'Strategy',
        'Accuracy (%)', 'Δ Accuracy (%)', 'Acc 95% CI', 'Acc P', 'Acc Method',
        'Efficiency', 'Δ Efficiency', 'Eff 95% CI', 'Eff P', 'Eff Method',
        'Review Rate (%)', 'Coverage (%)'
    ]
    return formatted[pub_cols]


pub_report = format_nature_report(final_report)

print("\n" + "=" * 100)
print("PUBLICATION-READY TABLE")
print(f"Accuracy improvement: relative to each model's own baseline")
print(f"Efficiency baseline reference: {BASELINE_EFF_REF:.2f} (full human review)")
print("=" * 100)
print(pub_report.to_string(index=False))

out_path = os.path.join(OUTPUT_DIR, "Table1_Strategy_Analysis.xlsx")
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    pub_report.to_excel(writer, sheet_name='Publication', index=False)
    final_report.to_excel(writer, sheet_name='Full_Data', index=False)
print(f"\nReport saved to: {out_path}")

df_res.to_csv(os.path.join(OUTPUT_DIR, "Table1_All_Combinations.csv"), index=False, encoding='utf-8-sig')

# ── 7. Specific strategy output ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== 6. Specific strategy output ===")
print("=" * 70)

TARGET_MAIN = "DS3.1_withThinking"
TARGET_AUX = "DS3.1_noPrompt_withThinking"
TARGET_CHECKER = "Qwen32b_noThinking"
TARGET_STRATEGY = "All Union"
target_key = (TARGET_MAIN, TARGET_AUX, TARGET_CHECKER, TARGET_STRATEGY)

if target_key in detailed_results:
    result = detailed_results[target_key]
    df_detail = result['df']
    S1, S2, S3 = result['S1'], result['S2'], result['S3']
    sig_vec = result['sig_vec']

    print(f"\nTarget: {TARGET_MAIN} + {TARGET_AUX} + {TARGET_CHECKER} ({TARGET_STRATEGY})")
    print("-" * 50)
    print(f"Total samples:       {result['total_cases']}")
    print(f"Baseline accuracy:   {fmt_pct(result['baseline_acc'], 1)}%")
    print(f"Review Rate:         {fmt_pct(result['review_rate'], 1)}%")
    print(f"Coverage:            {fmt_pct(result['coverage'], 1)}%")
    print(f"Captured errors:     {result['tp']} / {result['total_errors']}")
    print(f"Final Accuracy:      {fmt_pct(result['final_acc'], 1)}%")
    print(f"Efficiency:          {result['efficiency']:.2f}")
    print("-" * 50)

    print("\nSignal activation:")
    print(f"  M1 (Aux):       {S1.sum():4d} ({fmt_pct(S1.mean(), 1)}%)")
    print(f"  M2 (Self):      {S2.sum():4d} ({fmt_pct(S2.mean(), 1)}%)")
    print(f"  M3 (Rationale): {S3.sum():4d} ({fmt_pct(S3.mean(), 1)}%)")
    print(f"  Union:          {sig_vec.sum():4d} ({fmt_pct(sig_vec.mean(), 1)}%)")

    print("\nError capture:")
    errors_mask = df_detail['Is_Error']
    print(f"  M1: {(errors_mask & S1).sum():3d} / {errors_mask.sum()}")
    print(f"  M2: {(errors_mask & S2).sum():3d} / {errors_mask.sum()}")
    print(f"  M3: {(errors_mask & S3).sum():3d} / {errors_mask.sum()}")
    print(f"  Union: {(errors_mask & sig_vec).sum():3d} / {errors_mask.sum()}")

    silent_cases = df_detail[errors_mask & (~sig_vec)]
    if len(silent_cases) > 0:
        print(f"\nMissed errors: {len(silent_cases)}")
        print(silent_cases[['病案号', 'Main_Pred', 'Gold']].head(10).to_string(index=False))
    else:
        print("\nNo missed errors.")
else:
    print(f"\nTarget combination not found: {target_key}")
    print("Available examples:")
    for i, k in enumerate(list(detailed_results.keys())[:5]):
        print(f"  {k}")
