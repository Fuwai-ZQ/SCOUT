"""
SCOUT framework evaluation: joint accuracy analysis for liver cancer screening.
Computes review rate, error coverage, efficiency ratio, and final accuracy
across three verification strategies (S1: model heterogeneity, S2: stochastic
inconsistency, S3: reasoning critique).
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== File Configuration ====================

CT_REPORTS_FILE = "ct_reports_final.csv"
GOLD_FILE = "liver_cancer_classification_result.csv"
MAIN_FILE = "liver_cancer_prediction_with_reasoning.csv"
AUDIT1_FILE = "liver_cancer_prediction_simple.csv"
AUDIT2_FILE = "liver_cancer_prediction_with_reasoning2.csv"
AUDIT3_FILE = "liver_reasoning_audit_results.csv"

OUTPUT_DETAIL = "scout_joint_analysis_detail.csv"
OUTPUT_STATS = "scout_statistics_summary.csv"
OUTPUT_REPORT = "scout_analysis_report.txt"
OUTPUT_SILENT_CASES = "scout_silent_failure_cases.csv"
OUTPUT_METRICS_TABLE = "scout_metrics_comparison.csv"
OUTPUT_FULL_DATA = "scout_full_data.csv"


# ==================== Utilities ====================

def read_csv_auto(filepath, **kwargs):
    """Try multiple encodings to read a CSV file."""
    for enc in ['utf-8-sig', 'utf-8', 'gb18030', 'gbk', 'iso-8859-1', 'latin-1']:
        try:
            df = pd.read_csv(filepath, encoding=enc, **kwargs)
            print(f"  {os.path.basename(filepath)} (encoding: {enc}, rows: {len(df)})")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot read {filepath}")


def standardize_datetime(dt_str):
    """Normalize datetime strings to 'YYYY-MM-DD HH:MM' format."""
    if pd.isna(dt_str):
        return None
    dt_str = str(dt_str).strip()
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M',
                '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M']:
        try:
            return pd.to_datetime(dt_str, format=fmt).strftime('%Y-%m-%d %H:%M')
        except Exception:
            continue
    try:
        return pd.to_datetime(dt_str).strftime('%Y-%m-%d %H:%M')
    except Exception:
        return str(dt_str)


def to_binary(category):
    """Convert 4-class category to binary (high-risk vs low-risk)."""
    if pd.isna(category):
        return np.nan
    category = str(category).lower().strip()
    if category in ['no_focal_lesion', 'benign_lesion']:
        return 'low_risk'
    elif category in ['high_risk', 'confirmed_liver_cancer']:
        return 'high_risk'
    return np.nan


def calculate_metrics(y_true, y_pred, pos_label='high_risk'):
    """Calculate binary classification metrics."""
    y_true_bin = (y_true == pos_label).astype(int)
    y_pred_bin = (y_pred == pos_label).astype(int)

    TP = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
    TN = ((y_true_bin == 0) & (y_pred_bin == 0)).sum()
    FP = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
    FN = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
    total = TP + TN + FP + FN

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    npv = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    accuracy = (TP + TN) / total if total > 0 else np.nan

    denom = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    mcc = ((TP * TN) - (FP * FN)) / denom if denom > 0 else np.nan

    return {
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
        'Sensitivity': sensitivity, 'Specificity': specificity,
        'PPV': ppv, 'NPV': npv, 'Accuracy': accuracy, 'MCC': mcc,
    }


def fmt(val, decimals=4):
    """Format a numeric value, returning 'N/A' for NaN."""
    return f"{val:.{decimals}f}" if not np.isnan(val) else 'N/A'


# ==================== Main Pipeline ====================

print("=" * 70)
print("SCOUT Joint Accuracy Analysis — Liver Cancer Screening")
print("=" * 70)

# --- Load data ---
print("\nLoading data...")
ct_reports = read_csv_auto(CT_REPORTS_FILE)
gold = read_csv_auto(GOLD_FILE)
df_main = read_csv_auto(MAIN_FILE)
df_audit1 = read_csv_auto(AUDIT1_FILE)
df_audit2 = read_csv_auto(AUDIT2_FILE)
df_audit3 = read_csv_auto(AUDIT3_FILE)

# --- Standardize dates and build composite key ---
print("\nStandardizing dates...")
for name, df in [('ct_reports', ct_reports), ('gold', gold), ('main', df_main),
                 ('audit1', df_audit1), ('audit2', df_audit2), ('audit3', df_audit3)]:
    df['exam_date_std'] = df['exam_date'].apply(standardize_datetime)
    df['combined_key'] = df['patient_id'].astype(str) + '_' + df['exam_date_std'].astype(str)

# --- Deduplicate by composite key ---
print("\nDeduplicating by composite key...")
ct_reports_full = ct_reports.copy()
ct_reports = ct_reports.drop_duplicates(subset=['combined_key'], keep='first')
gold = gold.drop_duplicates(subset=['combined_key'], keep='first')
df_main = df_main.drop_duplicates(subset=['combined_key'], keep='first')
df_audit1 = df_audit1.drop_duplicates(subset=['combined_key'], keep='first')
df_audit2 = df_audit2.drop_duplicates(subset=['combined_key'], keep='first')
df_audit3 = df_audit3.drop_duplicates(subset=['combined_key'], keep='first')

print(f"After dedup: ct_reports={len(ct_reports)}, gold={len(gold)}, main={len(df_main)}")

# --- Merge all datasets ---
print("\nMerging datasets...")
ct_base = ct_reports[['combined_key']].copy()

gold_select = gold[['combined_key', 'patient_id', 'exam_date',
                     'liver_lesion_category']].copy()
gold_select.columns = ['combined_key', 'patient_id', 'exam_date', 'gold_category']

if 'reasoning_detail' in df_main.columns:
    main_select = df_main[['combined_key', 'predicted_category', 'reasoning_detail']].copy()
    main_select.columns = ['combined_key', 'pred_main', 'reasoning_detail']
else:
    main_select = df_main[['combined_key', 'predicted_category']].copy()
    main_select.columns = ['combined_key', 'pred_main']
    main_select['reasoning_detail'] = ''

audit1_select = df_audit1[['combined_key', 'predicted_category']].copy()
audit1_select.columns = ['combined_key', 'pred_audit1']

audit2_select = df_audit2[['combined_key', 'predicted_category']].copy()
audit2_select.columns = ['combined_key', 'pred_audit2']

audit3_select = df_audit3[['combined_key', 'is_logically_sound']].copy()
audit3_select.columns = ['combined_key', 'audit_pass']

merged = ct_base.merge(gold_select, on='combined_key', how='inner')
merged = merged.merge(main_select, on='combined_key', how='inner')
merged = merged.merge(audit1_select, on='combined_key', how='inner')
merged = merged.merge(audit2_select, on='combined_key', how='inner')
merged = merged.merge(audit3_select, on='combined_key', how='left')
merged['audit_pass'] = merged['audit_pass'].fillna(True)

print(f"Merged: {len(merged)} cases")

# --- Binarize categories ---
print("\nBinarizing 4-class -> 2-class...")
for col_src, col_dst in [('gold_category', 'gold_binary'),
                          ('pred_main', 'pred_main_binary'),
                          ('pred_audit1', 'pred_audit1_binary'),
                          ('pred_audit2', 'pred_audit2_binary')]:
    merged[col_dst] = merged[col_src].apply(to_binary)

merged = merged.dropna(subset=['gold_binary', 'pred_main_binary',
                                'pred_audit1_binary', 'pred_audit2_binary'])
print(f"Valid samples: {len(merged)}")

# --- Compute verification signals ---
print("\nComputing verification signals...")
merged['Is_Error'] = merged['pred_main_binary'] != merged['gold_binary']
merged['Signal_S1'] = merged['pred_main_binary'] != merged['pred_audit1_binary']
merged['Signal_S2'] = merged['pred_main_binary'] != merged['pred_audit2_binary']
merged['audit_pass_bool'] = merged['audit_pass'].apply(
    lambda x: str(x).lower() in ['true', '1', 'yes'] if pd.notna(x) else True
)
merged['Signal_S3'] = ~merged['audit_pass_bool']
merged['Flag_Union'] = merged['Signal_S1'] | merged['Signal_S2'] | merged['Signal_S3']

merged['Is_Positive_Prediction'] = merged['pred_main_binary'] == 'high_risk'
merged['Needs_Review'] = merged['Flag_Union'] | merged['Is_Positive_Prediction']

merged['Joint_Correct'] = (~merged['Is_Error']) | (merged['Is_Error'] & merged['Needs_Review'])
merged['Joint_Correct_Inconsistent_Only'] = (
    (~merged['Is_Error']) | (merged['Is_Error'] & merged['Flag_Union'])
)

merged['pred_reviewed_binary'] = merged.apply(
    lambda r: r['gold_binary'] if r['Needs_Review'] else r['pred_main_binary'], axis=1
)
merged['pred_reviewed_binary_inconsistent_only'] = merged.apply(
    lambda r: r['gold_binary'] if r['Flag_Union'] else r['pred_main_binary'], axis=1
)

# ==================== Results ====================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

N = len(merged)
n_errors = int(merged['Is_Error'].sum())
original_acc = 1 - (n_errors / N) if N > 0 else np.nan

n_flagged = int(merged['Flag_Union'].sum())
rr_flag_only = n_flagged / N if N > 0 else np.nan

n_review = int(merged['Needs_Review'].sum())
rr = n_review / N if N > 0 else np.nan

errors_by_flag = int((merged['Is_Error'] & merged['Flag_Union']).sum())
errors_by_review = int((merged['Is_Error'] & merged['Needs_Review']).sum())
ec = errors_by_review / n_errors if n_errors > 0 else np.nan
er = ec / rr if rr > 0 else np.nan

ec_flag = errors_by_flag / n_errors if n_errors > 0 else np.nan
er_flag = ec_flag / rr_flag_only if rr_flag_only > 0 else np.nan

final_acc = int(merged['Joint_Correct'].sum()) / N if N > 0 else np.nan
final_acc_flag = int(merged['Joint_Correct_Inconsistent_Only'].sum()) / N if N > 0 else np.nan

print(f"\nTotal cases: {N}")
print(f"Errors: {n_errors}")
print(f"Original accuracy (Mmain): {fmt(original_acc * 100, 2)}%")

print(f"\n--- SCOUT Screening (Review = Inconsistent OR Positive) ---")
print(f"Positive predictions: {int(merged['Is_Positive_Prediction'].sum())}")
print(f"Inconsistency flags (Union): {n_flagged}")
print(f"Cases needing review: {n_review}")
print(f"Review rate: {fmt(rr * 100, 2)}%")
print(f"Error coverage: {fmt(ec * 100, 2)}%")
print(f"Efficiency ratio: {fmt(er)}")
print(f"Final accuracy: {fmt(final_acc * 100, 2)}%")

print(f"\n--- Comparison: Inconsistency-only ---")
print(f"Review rate: {fmt(rr_flag_only * 100, 2)}%")
print(f"Error coverage: {fmt(ec_flag * 100, 2)}%")
print(f"Efficiency ratio: {fmt(er_flag)}")
print(f"Final accuracy: {fmt(final_acc_flag * 100, 2)}%")

# --- Signal breakdown ---
print(f"\n--- Signal Activation ---")
for sig in ['Signal_S1', 'Signal_S2', 'Signal_S3', 'Flag_Union']:
    n = int(merged[sig].sum())
    pct = merged[sig].mean() * 100
    print(f"{sig}: {n} ({pct:.2f}%)")

errors_only = merged[merged['Is_Error']]
if len(errors_only) > 0:
    print(f"\n--- Error Capture by Signal ---")
    for sig in ['Signal_S1', 'Signal_S2', 'Signal_S3', 'Flag_Union']:
        n = int(errors_only[sig].sum())
        pct = errors_only[sig].mean() * 100
        print(f"{sig}: {n}/{len(errors_only)} ({pct:.2f}%)")

    only_s1 = int((errors_only['Signal_S1'] & ~errors_only['Signal_S2'] & ~errors_only['Signal_S3']).sum())
    only_s2 = int((~errors_only['Signal_S1'] & errors_only['Signal_S2'] & ~errors_only['Signal_S3']).sum())
    only_s3 = int((~errors_only['Signal_S1'] & ~errors_only['Signal_S2'] & errors_only['Signal_S3']).sum())
    multi = int(errors_only['Flag_Union'].sum()) - only_s1 - only_s2 - only_s3
    print(f"\nUnique contributions: S1-only={only_s1}, S2-only={only_s2}, S3-only={only_s3}, multi={multi}")

# --- Metrics comparison ---
print(f"\n{'=' * 70}")
print("Metrics Comparison: Original vs Post-Review")
print("=" * 70)

m_orig = calculate_metrics(merged['gold_binary'], merged['pred_main_binary'])
m_rev = calculate_metrics(merged['gold_binary'], merged['pred_reviewed_binary'])

metric_names = ['TP', 'TN', 'FP', 'FN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'MCC']
metrics_comparison = pd.DataFrame({
    'Metric': metric_names,
    'Original': [m_orig[k] if isinstance(m_orig[k], int) else fmt(m_orig[k]) for k in metric_names],
    'Post-Review': [m_rev[k] if isinstance(m_rev[k], int) else fmt(m_rev[k]) for k in metric_names],
})
print(metrics_comparison.to_string(index=False))
metrics_comparison.to_csv(OUTPUT_METRICS_TABLE, index=False, encoding='utf-8-sig')
print(f"\nSaved: {OUTPUT_METRICS_TABLE}")

# --- Silent failure analysis ---
print("\nSilent failure analysis...")
silent_cases = merged[(merged['Is_Error']) & (~merged['Flag_Union'])]

if len(silent_cases) > 0:
    print(f"Found {len(silent_cases)} silent failure case(s)")
    ct_reports_full['exam_date_std'] = ct_reports_full['exam_date'].apply(standardize_datetime)
    ct_reports_full['combined_key'] = (ct_reports_full['patient_id'].astype(str) + '_'
                                       + ct_reports_full['exam_date_std'].astype(str))
    silent_keys = set(silent_cases['combined_key'])
    silent_full = ct_reports_full[ct_reports_full['combined_key'].isin(silent_keys)].copy()

    silent_export = silent_cases[[
        'combined_key', 'patient_id', 'exam_date',
        'gold_category', 'gold_binary', 'pred_main', 'pred_main_binary',
        'reasoning_detail', 'pred_audit1_binary', 'pred_audit2_binary', 'audit_pass_bool',
    ]].merge(
        silent_full.drop_duplicates(subset=['combined_key']),
        on='combined_key', how='left', suffixes=('', '_full')
    )
    silent_export.to_csv(OUTPUT_SILENT_CASES, index=False, encoding='utf-8-sig')
    print(f"Saved: {OUTPUT_SILENT_CASES}")
else:
    print("No silent failures detected.")
    pd.DataFrame({'note': ['No silent failure cases']}).to_csv(
        OUTPUT_SILENT_CASES, index=False, encoding='utf-8-sig')

# --- Save full data ---
print("\nSaving full data table...")
full_data = merged.copy()
full_data['original_correct'] = ~full_data['Is_Error']
full_data['reviewed_correct'] = full_data['pred_reviewed_binary'] == full_data['gold_binary']
full_data['error_direction'] = full_data.apply(
    lambda r: f"{r['gold_binary']}->{r['pred_main_binary']}" if r['Is_Error'] else 'correct', axis=1
)

output_cols = [
    'combined_key', 'patient_id', 'exam_date',
    'gold_category', 'gold_binary', 'pred_main', 'pred_main_binary', 'reasoning_detail',
    'pred_audit1', 'pred_audit1_binary', 'pred_audit2', 'pred_audit2_binary',
    'audit_pass', 'audit_pass_bool',
    'Signal_S1', 'Signal_S2', 'Signal_S3', 'Flag_Union',
    'Is_Error', 'Joint_Correct', 'pred_reviewed_binary',
    'original_correct', 'reviewed_correct', 'error_direction',
]
output_cols = [c for c in output_cols if c in full_data.columns]
full_data[output_cols].to_csv(OUTPUT_FULL_DATA, index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_FULL_DATA} ({len(full_data)} rows, {len(output_cols)} cols)")

# --- Save detail ---
detail_cols = [
    'combined_key', 'patient_id', 'exam_date',
    'gold_category', 'gold_binary', 'pred_main', 'pred_main_binary', 'reasoning_detail',
    'pred_audit1', 'pred_audit1_binary', 'pred_audit2', 'pred_audit2_binary',
    'audit_pass', 'audit_pass_bool',
    'Is_Positive_Prediction', 'Is_Error',
    'Signal_S1', 'Signal_S2', 'Signal_S3', 'Flag_Union', 'Needs_Review',
    'Joint_Correct', 'Joint_Correct_Inconsistent_Only', 'pred_reviewed_binary',
]
detail_cols = [c for c in detail_cols if c in merged.columns]
merged[detail_cols].to_csv(OUTPUT_DETAIL, index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_DETAIL}")

# --- Save statistics summary ---
stats_rows = [
    ('Baseline cases (ct_reports)', len(ct_reports)),
    ('Valid samples', N),
    ('Original errors', n_errors),
    ('Original accuracy', f"{original_acc * 100:.2f}%"),
    ('Positive predictions', int(merged['Is_Positive_Prediction'].sum())),
    ('Inconsistency flags (Union)', n_flagged),
    ('Cases needing review', n_review),
    ('Review rate', f"{rr * 100:.2f}%"),
    ('Error coverage', f"{ec * 100:.2f}%"),
    ('Efficiency ratio', fmt(er)),
    ('Final accuracy', f"{final_acc * 100:.2f}%"),
    ('Silent failures', len(silent_cases)),
    ('S1 activations', int(merged['Signal_S1'].sum())),
    ('S2 activations', int(merged['Signal_S2'].sum())),
    ('S3 activations', int(merged['Signal_S3'].sum())),
    ('Original Sensitivity', fmt(m_orig['Sensitivity'])),
    ('Original Specificity', fmt(m_orig['Specificity'])),
    ('Reviewed Sensitivity', fmt(m_rev['Sensitivity'])),
    ('Reviewed Specificity', fmt(m_rev['Specificity'])),
]
pd.DataFrame(stats_rows, columns=['Metric', 'Value']).to_csv(
    OUTPUT_STATS, index=False, encoding='utf-8-sig')
print(f"Saved: {OUTPUT_STATS}")

# --- Save text report ---
with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("SCOUT Joint Accuracy Analysis Report — Liver Cancer Screening\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")

    f.write("[Method]\n")
    f.write(f"Composite key: patient_id + exam_date\n")
    f.write(f"Binarization: low_risk={{no_focal_lesion, benign_lesion}}, "
            f"high_risk={{high_risk, confirmed_liver_cancer}}\n")
    f.write(f"Review scope: Inconsistent (Flag_Union) OR Positive prediction\n\n")

    f.write("[Core Metrics]\n")
    f.write(f"Valid samples: {N}\n")
    f.write(f"Original accuracy: {fmt(original_acc * 100, 2)}%\n")
    f.write(f"Review rate: {fmt(rr * 100, 2)}%\n")
    f.write(f"Error coverage: {fmt(ec * 100, 2)}%\n")
    f.write(f"Efficiency ratio: {fmt(er)}\n")
    f.write(f"Final accuracy: {fmt(final_acc * 100, 2)}%\n")
    f.write(f"Silent failures: {len(silent_cases)}\n\n")

    f.write("[Metrics Comparison]\n")
    f.write(metrics_comparison.to_string(index=False) + "\n\n")

    f.write("[Output Files]\n")
    for i, path in enumerate([OUTPUT_DETAIL, OUTPUT_STATS, OUTPUT_METRICS_TABLE,
                               OUTPUT_SILENT_CASES, OUTPUT_FULL_DATA], 1):
        f.write(f"  {i}. {path}\n")

print(f"Saved: {OUTPUT_REPORT}")
print("\nAnalysis complete.")
