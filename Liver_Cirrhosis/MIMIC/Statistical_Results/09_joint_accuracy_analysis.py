#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCOUT framework evaluation for liver cancer screening (MIMIC-IV cohort).

Computes SCOUT metrics (review rate, error coverage, efficiency ratio,
final accuracy) by triangulating three verification strategies (S1/S2/S3)
against a gold-standard classification.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# File paths
# =============================================================================

GOLD_FILE   = "liver_cancer_classification_result_filtered.csv"
MAIN_FILE   = "liver_cancer_prediction_optimized.csv"          # Mmain (optimized prompt)
AUDIT1_FILE = "liver_cancer_prediction_baseline.csv"           # S1: model heterogeneity
AUDIT2_FILE = "liver_cancer_prediction_optimized_rerun.csv"    # S2: stochastic inconsistency
AUDIT3_FILE = "liver_reasoning_audit_results.csv"              # S3: reasoning critique

OUTPUT_DETAIL        = "outputs/MIMIC_joint_analysis_detail.csv"
OUTPUT_STATS         = "outputs/MIMIC_statistics_summary.csv"
OUTPUT_REPORT        = "outputs/MIMIC_analysis_report.txt"
OUTPUT_SILENT_CASES  = "outputs/MIMIC_missed_cases.csv"
OUTPUT_METRICS_TABLE = "outputs/MIMIC_metrics_comparison.csv"
OUTPUT_FULL_DATA     = "outputs/MIMIC_full_data.csv"


# =============================================================================
# Helpers
# =============================================================================

def read_csv_auto_encoding(filepath, **kwargs):
    """Try common encodings until one succeeds."""
    for enc in ['utf-8-sig', 'utf-8', 'gb18030', 'gbk', 'iso-8859-1']:
        try:
            df = pd.read_csv(filepath, encoding=enc, **kwargs)
            print(f"  {os.path.basename(filepath)} ({enc}, n={len(df)})")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot read {filepath}")


def to_binary(category):
    """Map four-class label to binary risk level."""
    if pd.isna(category):
        return np.nan
    cat = str(category).lower().strip()
    if cat in ('no_focal_lesion', 'benign_lesion'):
        return 'low_risk'
    elif cat in ('high_risk', 'confirmed_liver_cancer'):
        return 'high_risk'
    return np.nan


def calculate_metrics(y_true, y_pred, pos_label='high_risk'):
    """Compute binary classification metrics."""
    yt = (y_true == pos_label).astype(int)
    yp = (y_pred == pos_label).astype(int)

    TP = ((yt == 1) & (yp == 1)).sum()
    TN = ((yt == 0) & (yp == 0)).sum()
    FP = ((yt == 0) & (yp == 1)).sum()
    FN = ((yt == 1) & (yp == 0)).sum()
    total = TP + TN + FP + FN

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    npv = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    accuracy = (TP + TN) / total if total > 0 else np.nan
    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = ((TP * TN) - (FP * FN)) / denom if denom > 0 else np.nan

    return {
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
        'Sensitivity': sensitivity, 'Specificity': specificity,
        'PPV': ppv, 'NPV': npv,
        'Accuracy': accuracy, 'MCC': mcc,
    }


# =============================================================================
# Load & prepare
# =============================================================================

print("=" * 70)
print("  SCOUT Evaluation — Liver Cancer Screening (MIMIC-IV)")
print("  Composite key: hadm_id + note_id")
print("=" * 70)
print("\nReading data ...")

gold      = read_csv_auto_encoding(GOLD_FILE)
df_main   = read_csv_auto_encoding(MAIN_FILE)
df_audit1 = read_csv_auto_encoding(AUDIT1_FILE)
df_audit2 = read_csv_auto_encoding(AUDIT2_FILE)
df_audit3 = read_csv_auto_encoding(AUDIT3_FILE)

print("\nCreating composite keys ...")

for df in [gold, df_main, df_audit1, df_audit2]:
    df['combined_key'] = df['hadm_id'].astype(str) + '_' + df['note_id'].astype(str)

df_audit3['hadm_id'] = df_audit3['hadm_id'].astype(str)

print(f"Gold standard rows: {len(gold)}, unique keys: {gold['combined_key'].nunique()}")
print(f"Main model rows: {len(df_main)}, unique keys: {df_main['combined_key'].nunique()}")
print(f"S1 (baseline) rows: {len(df_audit1)}, unique keys: {df_audit1['combined_key'].nunique()}")
print(f"S2 (rerun) rows: {len(df_audit2)}, unique keys: {df_audit2['combined_key'].nunique()}")
print(f"S3 (audit) rows: {len(df_audit3)}, unique hadm_id: {df_audit3['hadm_id'].nunique()}")

print("\nDeduplicating (keep first) ...")
gold      = gold.drop_duplicates(subset=['combined_key'], keep='first')
df_main   = df_main.drop_duplicates(subset=['combined_key'], keep='first')
df_audit1 = df_audit1.drop_duplicates(subset=['combined_key'], keep='first')
df_audit2 = df_audit2.drop_duplicates(subset=['combined_key'], keep='first')
df_audit3 = df_audit3.drop_duplicates(subset=['hadm_id'], keep='first')

print(f"Gold: {len(gold)} | Main: {len(df_main)} | S1: {len(df_audit1)} "
      f"| S2: {len(df_audit2)} | S3: {len(df_audit3)}")

# =============================================================================
# Merge
# =============================================================================

print("\nMerging ...")

gold_select = gold[['combined_key', 'hadm_id', 'note_id', 'liver_lesion_category']].copy()
gold_select.columns = ['combined_key', 'hadm_id', 'note_id', 'gold_category']
gold_select['hadm_id'] = gold_select['hadm_id'].astype(str)

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

audit3_select = df_audit3[['hadm_id', 'is_logically_sound']].copy()
audit3_select.columns = ['hadm_id', 'audit_pass']

n_gold = len(gold_select)
merged = gold_select.merge(main_select, on='combined_key', how='inner')
n_after_main = len(merged)

merged = merged.merge(audit1_select, on='combined_key', how='inner')
n_after_s1 = len(merged)

merged = merged.merge(audit2_select, on='combined_key', how='inner')
n_after_s2 = len(merged)

merged = merged.merge(audit3_select, on='hadm_id', how='left')
merged['audit_pass'] = merged['audit_pass'].fillna(True)
n_after_s3 = len(merged)

print(f"  Merged: {len(merged)} cases")

print("\n" + "=" * 70)
print("  Data Cleaning Report")
print("=" * 70)
print(f"Gold (deduplicated):  {n_gold}")
print(f"After main merge:     {n_after_main}  (dropped {n_gold - n_after_main})")
print(f"After S1 merge:       {n_after_s1}  (dropped {n_after_main - n_after_s1})")
print(f"After S2 merge:       {n_after_s2}  (dropped {n_after_s1 - n_after_s2})")
print(f"After S3 merge:       {n_after_s3}  (dropped {n_after_s2 - n_after_s3})")
print(f"Final sample size:    {len(merged)}")
print("=" * 70)

# =============================================================================
# Binarize & compute signals
# =============================================================================

print("\nBinarizing four-class → two-class ...")
merged['gold_binary']       = merged['gold_category'].apply(to_binary)
merged['pred_main_binary']  = merged['pred_main'].apply(to_binary)
merged['pred_audit1_binary'] = merged['pred_audit1'].apply(to_binary)
merged['pred_audit2_binary'] = merged['pred_audit2'].apply(to_binary)

merged = merged.dropna(subset=['gold_binary', 'pred_main_binary',
                                'pred_audit1_binary', 'pred_audit2_binary'])
print(f"  Valid cases: {len(merged)}")

print("\nComputing verification signals ...")
merged['Is_Error']   = merged['pred_main_binary'] != merged['gold_binary']
merged['Signal_M1']  = merged['pred_main_binary'] != merged['pred_audit1_binary']
merged['Signal_M2']  = merged['pred_main_binary'] != merged['pred_audit2_binary']
merged['audit_pass_bool'] = merged['audit_pass'].apply(
    lambda x: str(x).lower() in ('true', '1', 'yes') if pd.notna(x) else True
)
merged['Signal_M3']  = ~merged['audit_pass_bool']
merged['Flag_Union'] = merged['Signal_M1'] | merged['Signal_M2'] | merged['Signal_M3']

merged['Is_Positive_Prediction'] = merged['pred_main_binary'] == 'high_risk'
merged['Needs_Review'] = merged['Flag_Union'] | merged['Is_Positive_Prediction']

merged['Joint_Correct'] = (~merged['Is_Error']) | (merged['Is_Error'] & merged['Needs_Review'])
merged['Joint_Correct_Inconsistent_Only'] = (~merged['Is_Error']) | (merged['Is_Error'] & merged['Flag_Union'])


def get_reviewed_prediction(row):
    return row['gold_binary'] if row['Needs_Review'] else row['pred_main_binary']


def get_reviewed_prediction_inconsistent_only(row):
    return row['gold_binary'] if row['Flag_Union'] else row['pred_main_binary']


merged['pred_reviewed_binary'] = merged.apply(get_reviewed_prediction, axis=1)
merged['pred_reviewed_binary_inconsistent_only'] = merged.apply(
    get_reviewed_prediction_inconsistent_only, axis=1)

# =============================================================================
# SCOUT metrics
# =============================================================================

print("\n" + "=" * 70)
print("  SCOUT Evaluation Results")
print("=" * 70)

total_cases  = len(merged)
total_errors = merged['Is_Error'].sum()
original_acc = 1 - (total_errors / total_cases) if total_cases > 0 else np.nan

total_flagged_inconsistent = merged['Flag_Union'].sum()
review_rate_inconsistent_only = total_flagged_inconsistent / total_cases if total_cases > 0 else np.nan

total_needs_review = merged['Needs_Review'].sum()
review_rate = total_needs_review / total_cases if total_cases > 0 else np.nan

errors_flagged_by_inconsistency = (merged['Is_Error'] & merged['Flag_Union']).sum()
errors_captured_by_review = (merged['Is_Error'] & merged['Needs_Review']).sum()
coverage = errors_captured_by_review / total_errors if total_errors > 0 else np.nan

tp = errors_captured_by_review
fn = total_errors - errors_captured_by_review
fp = total_needs_review - errors_captured_by_review

efficiency = coverage / review_rate if review_rate > 0 else np.nan

tp_inc = errors_flagged_by_inconsistency
fn_inc = total_errors - errors_flagged_by_inconsistency
fp_inc = total_flagged_inconsistent - errors_flagged_by_inconsistency
coverage_inc = errors_flagged_by_inconsistency / total_errors if total_errors > 0 else np.nan
efficiency_inc = coverage_inc / review_rate_inconsistent_only if review_rate_inconsistent_only > 0 else np.nan

joint_correct_count = merged['Joint_Correct'].sum()
final_acc = joint_correct_count / total_cases if total_cases > 0 else np.nan

print(f"\nTotal valid cases: {total_cases}")
print(f"Initial errors: {total_errors}")
print(f"Baseline accuracy (Mmain): {original_acc * 100:.2f}%")

print(f"\n{'=' * 50}")
print("SCOUT Screening Metrics (review = inconsistency OR positive prediction)")
print("=" * 50)
print(f"Positive predictions (Mmain): {merged['Is_Positive_Prediction'].sum()}")
print(f"Inconsistency flags (Flag_Union): {total_flagged_inconsistent}")
print(f"Total cases for review (Needs_Review): {total_needs_review}")
print(f"Review rate: {review_rate * 100:.2f}%")
print(f"Error coverage: {coverage * 100:.2f}%")
print(f"TP (captured errors): {tp}")
print(f"FN (silent failures): {fn}")
print(f"FP (correct but reviewed): {fp}")
print(f"Efficiency ratio (coverage / review rate): {efficiency:.4f}")

print(f"\n{'-' * 50}")
print("Comparison: inconsistency-only review")
print("-" * 50)
print(f"Inconsistency-only flags: {total_flagged_inconsistent}")
print(f"Inconsistency-only review rate: {review_rate_inconsistent_only * 100:.2f}%")
print(f"Inconsistency-only coverage: {coverage_inc * 100:.2f}%")
print(f"Inconsistency-only efficiency: {efficiency_inc:.4f}")
print(f"TP={tp_inc}, FN={fn_inc}, FP={fp_inc}")

joint_correct_inc = merged['Joint_Correct_Inconsistent_Only'].sum()
final_acc_inc = joint_correct_inc / total_cases if total_cases > 0 else np.nan

print(f"\n{'=' * 50}")
print("Final Accuracy Comparison")
print("=" * 50)
print(f"Final accuracy (SCOUT screening): {final_acc * 100:.2f}%")
print(f"Final accuracy (inconsistency only): {final_acc_inc * 100:.2f}%")

# Signal trigger rates
print(f"\n{'-' * 70}")
print("  Signal Trigger Rates")
print("-" * 70)
if total_cases > 0:
    print(f"S1 (model heterogeneity):     {merged['Signal_M1'].sum()} ({merged['Signal_M1'].mean()*100:.2f}%)")
    print(f"S2 (stochastic inconsistency): {merged['Signal_M2'].sum()} ({merged['Signal_M2'].mean()*100:.2f}%)")
    print(f"S3 (reasoning critique):       {merged['Signal_M3'].sum()} ({merged['Signal_M3'].mean()*100:.2f}%)")
    print(f"Union (S1|S2|S3):              {merged['Flag_Union'].sum()} ({merged['Flag_Union'].mean()*100:.2f}%)")

# Error capture breakdown
print(f"\n{'-' * 70}")
print("  Error Capture Source Analysis")
print("-" * 70)
errors_only = merged[merged['Is_Error']]
if len(errors_only) > 0:
    print(f"Total errors: {len(errors_only)}")
    print(f"S1 captured: {errors_only['Signal_M1'].sum()} / {len(errors_only)} ({errors_only['Signal_M1'].mean()*100:.2f}%)")
    print(f"S2 captured: {errors_only['Signal_M2'].sum()} / {len(errors_only)} ({errors_only['Signal_M2'].mean()*100:.2f}%)")
    print(f"S3 captured: {errors_only['Signal_M3'].sum()} / {len(errors_only)} ({errors_only['Signal_M3'].mean()*100:.2f}%)")
    print(f"Union captured: {errors_only['Flag_Union'].sum()} / {len(errors_only)} ({errors_only['Flag_Union'].mean()*100:.2f}%)")

    only_m1 = (errors_only['Signal_M1'] & ~errors_only['Signal_M2'] & ~errors_only['Signal_M3']).sum()
    only_m2 = (~errors_only['Signal_M1'] & errors_only['Signal_M2'] & ~errors_only['Signal_M3']).sum()
    only_m3 = (~errors_only['Signal_M1'] & ~errors_only['Signal_M2'] & errors_only['Signal_M3']).sum()
    multi = errors_only['Flag_Union'].sum() - only_m1 - only_m2 - only_m3

    print(f"\nExclusive contributions:")
    print(f"  S1 only: {only_m1}")
    print(f"  S2 only: {only_m2}")
    print(f"  S3 only: {only_m3}")
    print(f"  Multi-signal: {multi}")
else:
    print("No errors found.")
    only_m1 = only_m2 = only_m3 = multi = 0

# =============================================================================
# Metrics comparison table
# =============================================================================

print("\n" + "=" * 70)
print("  Metrics Comparison (Original vs. Post-Review)")
print("=" * 70)

metrics_original = calculate_metrics(merged['gold_binary'], merged['pred_main_binary'])
metrics_reviewed = calculate_metrics(merged['gold_binary'], merged['pred_reviewed_binary'])


def fmt(v):
    return f"{v:.4f}" if not np.isnan(v) else 'N/A'


def calc_change(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return 'N/A'
    if np.isnan(a) or np.isnan(b):
        return 'N/A'
    d = b - a
    return f"+{d:.4f}" if d > 0 else (f"{d:.4f}" if d < 0 else "0")


metric_names = ['TP', 'TN', 'FP', 'FN',
                'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'MCC']

metrics_comparison = pd.DataFrame({
    'Metric': metric_names,
    'Original': [metrics_original[k] if k in ('TP', 'TN', 'FP', 'FN')
                 else fmt(metrics_original[k]) for k in metric_names],
    'Post_Review': [metrics_reviewed[k] if k in ('TP', 'TN', 'FP', 'FN')
                    else fmt(metrics_reviewed[k]) for k in metric_names],
    'Change': [calc_change(metrics_original[k], metrics_reviewed[k]) for k in metric_names],
})

print(metrics_comparison.to_string(index=False))

metrics_comparison.to_csv(OUTPUT_METRICS_TABLE, index=False, encoding='utf-8-sig')
print(f"\nMetrics table saved: {OUTPUT_METRICS_TABLE}")

# =============================================================================
# Silent failure analysis
# =============================================================================

print("\nSilent failure analysis ...")
silent_cases = merged[(merged['Is_Error']) & (~merged['Flag_Union'])]

if len(silent_cases) > 0:
    print(f"Found {len(silent_cases)} silent failure(s)")
    silent_export = silent_cases[[
        'combined_key', 'hadm_id', 'note_id',
        'gold_category', 'gold_binary',
        'pred_main', 'pred_main_binary', 'reasoning_detail',
        'pred_audit1_binary', 'pred_audit2_binary', 'audit_pass_bool',
    ]].copy()
    silent_export.to_csv(OUTPUT_SILENT_CASES, index=False, encoding='utf-8-sig')
    print(f"  Saved: {OUTPUT_SILENT_CASES}")
else:
    print("No silent failures detected.")
    pd.DataFrame({'note': ['No silent failure cases found']}).to_csv(
        OUTPUT_SILENT_CASES, index=False, encoding='utf-8-sig')

# =============================================================================
# Full data export
# =============================================================================

print("\nExporting full data table ...")
full_data = merged.copy()
full_data['original_correct'] = ~full_data['Is_Error']
full_data['reviewed_correct'] = full_data['pred_reviewed_binary'] == full_data['gold_binary']
full_data['error_direction'] = full_data.apply(
    lambda r: f"{r['gold_binary']}->{r['pred_main_binary']}" if r['Is_Error'] else 'correct',
    axis=1,
)

output_columns = [
    'combined_key', 'hadm_id', 'note_id',
    'gold_category', 'gold_binary',
    'pred_main', 'pred_main_binary', 'reasoning_detail',
    'pred_audit1', 'pred_audit1_binary',
    'pred_audit2', 'pred_audit2_binary',
    'audit_pass', 'audit_pass_bool',
    'Signal_M1', 'Signal_M2', 'Signal_M3', 'Flag_Union',
    'Is_Error', 'Joint_Correct',
    'pred_reviewed_binary',
    'original_correct', 'reviewed_correct', 'error_direction',
]
output_columns = [c for c in output_columns if c in full_data.columns]
full_data[output_columns].to_csv(OUTPUT_FULL_DATA, index=False, encoding='utf-8-sig')
print(f"  Saved: {OUTPUT_FULL_DATA}  ({len(full_data)} rows x {len(output_columns)} cols)")

# Detail export
detail_cols = [
    'combined_key', 'hadm_id', 'note_id',
    'gold_category', 'gold_binary',
    'pred_main', 'pred_main_binary', 'reasoning_detail',
    'pred_audit1', 'pred_audit1_binary',
    'pred_audit2', 'pred_audit2_binary',
    'audit_pass', 'audit_pass_bool',
    'Is_Positive_Prediction', 'Is_Error',
    'Signal_M1', 'Signal_M2', 'Signal_M3',
    'Flag_Union', 'Needs_Review',
    'Joint_Correct', 'Joint_Correct_Inconsistent_Only',
    'pred_reviewed_binary',
]
merged[detail_cols].to_csv(OUTPUT_DETAIL, index=False, encoding='utf-8-sig')
print(f"  Saved: {OUTPUT_DETAIL}")

# =============================================================================
# Statistics summary CSV
# =============================================================================

only_m1_s = (errors_only['Signal_M1'] & ~errors_only['Signal_M2'] & ~errors_only['Signal_M3']).sum() if len(errors_only) > 0 else 0
only_m2_s = (~errors_only['Signal_M1'] & errors_only['Signal_M2'] & ~errors_only['Signal_M3']).sum() if len(errors_only) > 0 else 0
only_m3_s = (~errors_only['Signal_M1'] & ~errors_only['Signal_M2'] & errors_only['Signal_M3']).sum() if len(errors_only) > 0 else 0

stats_df = pd.DataFrame({
    'Metric': [
        'Gold standard sample size', 'Valid sample size', 'Initial errors', 'Baseline accuracy',
        'Positive predictions (Mmain)', 'Inconsistency flags (Flag_Union)',
        'Cases for review (Needs_Review)', 'Review rate (SCOUT)',
        'Error coverage (SCOUT)', 'Efficiency ratio (SCOUT)', 'Final accuracy (SCOUT)',
        'TP (captured errors)', 'FN (silent failures)', 'FP (correct but reviewed)',
        'Review rate (inconsistency only)', 'Coverage (inconsistency only)',
        'Efficiency (inconsistency only)', 'Final accuracy (inconsistency only)',
        'Silent failure count',
        'S1 trigger count', 'S1 trigger rate',
        'S2 trigger count', 'S2 trigger rate',
        'S3 trigger count', 'S3 trigger rate',
        'Union trigger count', 'Union trigger rate',
        'S1 error capture count', 'S1 error capture rate',
        'S2 error capture count', 'S2 error capture rate',
        'S3 error capture count', 'S3 error capture rate',
        'S1-only exclusive capture', 'S2-only exclusive capture', 'S3-only exclusive capture',
        'Original Sensitivity', 'Original Specificity', 'Original PPV', 'Original NPV', 'Original MCC',
        'Post-review Sensitivity', 'Post-review Specificity', 'Post-review PPV', 'Post-review NPV', 'Post-review MCC',
    ],
    'Value': [
        n_gold, total_cases, total_errors,
        f"{original_acc * 100:.2f}%",
        merged['Is_Positive_Prediction'].sum(), total_flagged_inconsistent,
        total_needs_review, f"{review_rate * 100:.2f}%",
        f"{coverage * 100:.2f}%", f"{efficiency:.4f}", f"{final_acc * 100:.2f}%",
        tp, fn, fp,
        f"{review_rate_inconsistent_only * 100:.2f}%", f"{coverage_inc * 100:.2f}%",
        f"{efficiency_inc:.4f}", f"{final_acc_inc * 100:.2f}%",
        len(silent_cases),
        merged['Signal_M1'].sum(), f"{merged['Signal_M1'].mean()*100:.2f}%",
        merged['Signal_M2'].sum(), f"{merged['Signal_M2'].mean()*100:.2f}%",
        merged['Signal_M3'].sum(), f"{merged['Signal_M3'].mean()*100:.2f}%",
        merged['Flag_Union'].sum(), f"{merged['Flag_Union'].mean()*100:.2f}%",
        errors_only['Signal_M1'].sum() if len(errors_only) > 0 else 0,
        f"{errors_only['Signal_M1'].mean()*100:.2f}%" if len(errors_only) > 0 else "N/A",
        errors_only['Signal_M2'].sum() if len(errors_only) > 0 else 0,
        f"{errors_only['Signal_M2'].mean()*100:.2f}%" if len(errors_only) > 0 else "N/A",
        errors_only['Signal_M3'].sum() if len(errors_only) > 0 else 0,
        f"{errors_only['Signal_M3'].mean()*100:.2f}%" if len(errors_only) > 0 else "N/A",
        only_m1_s, only_m2_s, only_m3_s,
        fmt(metrics_original['Sensitivity']),
        fmt(metrics_original['Specificity']),
        fmt(metrics_original['PPV']),
        fmt(metrics_original['NPV']),
        fmt(metrics_original['MCC']),
        fmt(metrics_reviewed['Sensitivity']),
        fmt(metrics_reviewed['Specificity']),
        fmt(metrics_reviewed['PPV']),
        fmt(metrics_reviewed['NPV']),
        fmt(metrics_reviewed['MCC']),
    ]
})
stats_df.to_csv(OUTPUT_STATS, index=False, encoding='utf-8-sig')
print(f"Statistics summary saved: {OUTPUT_STATS}")

# =============================================================================
# Summary table (console)
# =============================================================================

print("\n" + "=" * 70)
print("  Summary Table")
print("=" * 70)
print(f"\n{'Metric':<50} {'Value':>15}")
print("-" * 67)
for label, val in [
    ("Baseline accuracy (Mmain)", f"{original_acc * 100:.2f}%"),
    ("Positive predictions (Mmain)", str(merged['Is_Positive_Prediction'].sum())),
    ("Inconsistency flags (Flag_Union)", str(total_flagged_inconsistent)),
    ("Cases for review (Needs_Review)", str(total_needs_review)),
    ("Review rate", f"{review_rate * 100:.2f}%"),
    ("Error coverage", f"{coverage * 100:.2f}%"),
    ("Efficiency ratio", f"{efficiency:.4f}"),
    ("Final accuracy (SCOUT)", f"{final_acc * 100:.2f}%"),
]:
    print(f"  {label:<48} {val:>15}")
print("=" * 70)

# =============================================================================
# Text report
# =============================================================================

with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("  SCOUT Evaluation Report — Liver Cancer Screening (MIMIC-IV)\n")
    f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")

    f.write("[Methods]\n" + "-" * 70 + "\n")
    f.write("1. Composite key: hadm_id + note_id (S3 uses hadm_id only)\n")
    f.write("2. Binarization: low_risk = {no_focal_lesion, benign_lesion}; "
            "high_risk = {high_risk, confirmed_liver_cancer}\n")
    f.write("3. Review scope: inconsistency (Flag_Union) OR positive prediction (high_risk)\n")
    f.write("4. Efficiency ratio = error coverage / review rate\n\n")

    f.write("[Source Files]\n" + "-" * 70 + "\n")
    f.write(f"Gold standard: {GOLD_FILE}\n")
    f.write(f"Main model:    {MAIN_FILE}\n")
    f.write(f"S1 (baseline): {AUDIT1_FILE}\n")
    f.write(f"S2 (rerun):    {AUDIT2_FILE}\n")
    f.write(f"S3 (audit):    {AUDIT3_FILE}\n\n")

    f.write("[Data Cleaning Report]\n" + "-" * 70 + "\n")
    f.write(f"Gold (deduplicated):  {n_gold}\n")
    f.write(f"After main merge:     {n_after_main}  (dropped {n_gold - n_after_main})\n")
    f.write(f"After S1 merge:       {n_after_s1}  (dropped {n_after_main - n_after_s1})\n")
    f.write(f"After S2 merge:       {n_after_s2}  (dropped {n_after_s1 - n_after_s2})\n")
    f.write(f"After S3 merge:       {n_after_s3}  (dropped {n_after_s2 - n_after_s3})\n")
    f.write(f"Final valid sample:   {total_cases}\n\n")

    f.write("[SCOUT Screening Results]\n" + "-" * 70 + "\n")
    f.write(f"Valid sample size: {total_cases}\n")
    f.write(f"Initial errors: {total_errors}\n")
    f.write(f"Baseline accuracy (Mmain): {original_acc * 100:.2f}%\n")
    f.write(f"Positive predictions: {merged['Is_Positive_Prediction'].sum()}\n")
    f.write(f"Inconsistency flags (Flag_Union): {total_flagged_inconsistent}\n")
    f.write(f"Cases for review (Needs_Review): {total_needs_review}\n")
    f.write(f"Review rate: {review_rate * 100:.2f}%\n")
    f.write(f"Error coverage: {coverage * 100:.2f}%\n")
    f.write(f"TP: {tp}, FN: {fn}, FP: {fp}\n")
    f.write(f"Efficiency ratio: {efficiency:.4f}\n")
    f.write(f"Final accuracy: {final_acc * 100:.2f}%\n\n")

    f.write("[Inconsistency-Only Comparison]\n" + "-" * 70 + "\n")
    f.write(f"Review rate: {review_rate_inconsistent_only * 100:.2f}%\n")
    f.write(f"Coverage: {coverage_inc * 100:.2f}%\n")
    f.write(f"Efficiency: {efficiency_inc:.4f}\n")
    f.write(f"Final accuracy: {final_acc_inc * 100:.2f}%\n\n")

    f.write("[Signal Trigger Rates]\n" + "-" * 70 + "\n")
    if total_cases > 0:
        f.write(f"S1 (model heterogeneity):      {merged['Signal_M1'].sum()} ({merged['Signal_M1'].mean()*100:.2f}%)\n")
        f.write(f"S2 (stochastic inconsistency): {merged['Signal_M2'].sum()} ({merged['Signal_M2'].mean()*100:.2f}%)\n")
        f.write(f"S3 (reasoning critique):       {merged['Signal_M3'].sum()} ({merged['Signal_M3'].mean()*100:.2f}%)\n")
        f.write(f"Union:                         {merged['Flag_Union'].sum()} ({merged['Flag_Union'].mean()*100:.2f}%)\n")
        f.write(f"Positive prediction:           {merged['Is_Positive_Prediction'].sum()} ({merged['Is_Positive_Prediction'].mean()*100:.2f}%)\n")
        f.write(f"Needs_Review:                  {merged['Needs_Review'].sum()} ({merged['Needs_Review'].mean()*100:.2f}%)\n\n")

    f.write("[Error Capture Source Analysis]\n" + "-" * 70 + "\n")
    if len(errors_only) > 0:
        f.write(f"Total errors: {len(errors_only)}\n")
        f.write(f"S1 captured: {errors_only['Signal_M1'].sum()} / {len(errors_only)} ({errors_only['Signal_M1'].mean()*100:.2f}%)\n")
        f.write(f"S2 captured: {errors_only['Signal_M2'].sum()} / {len(errors_only)} ({errors_only['Signal_M2'].mean()*100:.2f}%)\n")
        f.write(f"S3 captured: {errors_only['Signal_M3'].sum()} / {len(errors_only)} ({errors_only['Signal_M3'].mean()*100:.2f}%)\n")
        f.write(f"Union captured: {errors_only['Flag_Union'].sum()} / {len(errors_only)} ({errors_only['Flag_Union'].mean()*100:.2f}%)\n")
        f.write(f"\nExclusive contributions:\n")
        f.write(f"  S1 only: {only_m1}\n  S2 only: {only_m2}\n  S3 only: {only_m3}\n  Multi-signal: {multi}\n\n")
    else:
        f.write("No errors found.\n\n")

    f.write("[Metrics Comparison]\n" + "-" * 70 + "\n")
    f.write(metrics_comparison.to_string(index=False) + "\n\n")

    f.write("[Output Files]\n" + "-" * 70 + "\n")
    f.write(f"1. {OUTPUT_DETAIL}\n2. {OUTPUT_STATS}\n3. {OUTPUT_METRICS_TABLE}\n")
    f.write(f"4. {OUTPUT_SILENT_CASES}\n5. {OUTPUT_FULL_DATA}\n")
    f.write("=" * 70 + "\n")

print(f"\nReport saved: {OUTPUT_REPORT}")
print("\nAnalysis complete.")
