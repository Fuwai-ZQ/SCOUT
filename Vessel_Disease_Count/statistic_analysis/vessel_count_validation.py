# -*- coding: utf-8 -*-
"""
Diseased Vessel Counting — Validation & Metric Computation
Evaluates the SCOUT union strategy (S1 ∪ S2 ∪ S3) against gold-standard labels.
"""

import pandas as pd
import numpy as np

# ========================== Data Loading ==========================

path_main = "disease_vessels_DeepSeek_Aliyun_MultiThread1.csv"   # Mmain run 1
path_s2   = "disease_vessels_DeepSeek_Aliyun_MultiThread2.csv"   # Mmain run 2 (S2)
path_s1   = "disease_vessels_DeepSeek_Aliyun_WeakPrompt.csv"     # Maux (S1)
path_s3   = "disease_vessels_Audit_Results.csv"                  # Mcheck (S3)
path_gold = "df_clinical_gold.xlsx"

df_main = pd.read_csv(path_main, encoding='utf-8-sig')
df_s2   = pd.read_csv(path_s2, encoding='utf-8-sig')
df_s1   = pd.read_csv(path_s1, encoding='utf-8-sig')
df_s3   = pd.read_csv(path_s3, encoding='utf-8-sig')
df_gold = pd.read_excel(path_gold)

print(f"Mmain run 1: {len(df_main)} cases")
print(f"Mmain run 2 (S2): {len(df_s2)} cases")
print(f"Maux (S1): {len(df_s1)} cases")
print(f"Mcheck (S3): {len(df_s3)} cases")
print(f"Gold standard: {len(df_gold)} cases")


# ========================== Helpers ==========================

def clean_label(x):
    """Normalize label to integer or NaN."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper()
    if x in ('', 'NA', 'NULL', 'NAN', 'NONE'):
        return np.nan
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return np.nan


SAFE_KEYWORDS = ["不存在", "一致", "无矛盾", "None", "No", "Pass", "Consistent", "PASS"]


def is_safe(text):
    """Check if audit result indicates PASS (no issue found)."""
    if pd.isna(text) or str(text).strip() == '':
        return False
    text_lower = str(text).lower()
    return any(kw.lower() in text_lower for kw in SAFE_KEYWORDS)


# ========================== Preprocessing ==========================

# Gold standard
df_gold_clean = df_gold[['病案号', '病变支数']].copy()
df_gold_clean.columns = ['hadm_id', 'gold_label']
df_gold_clean['hadm_id'] = df_gold_clean['hadm_id'].astype(str)
df_gold_clean['gold_label'] = df_gold_clean['gold_label'].apply(clean_label)
df_gold_clean = df_gold_clean.dropna(subset=['gold_label'])
print(f"\nGold standard (valid): {len(df_gold_clean)} cases")

# Mmain prediction (run 1)
df_main_clean = df_main[['病案号', 'disease_vessel_count']].copy()
df_main_clean.columns = ['hadm_id', 'main_pred']
df_main_clean['hadm_id'] = df_main_clean['hadm_id'].astype(str)
df_main_clean['main_pred'] = df_main_clean['main_pred'].apply(clean_label)

# S1: Maux (baseline prompt)
df_s1_clean = df_s1[['病案号', 'disease_vessel_count']].copy()
df_s1_clean.columns = ['hadm_id', 's1_pred']
df_s1_clean['hadm_id'] = df_s1_clean['hadm_id'].astype(str)
df_s1_clean['s1_pred'] = df_s1_clean['s1_pred'].apply(clean_label)

# S2: Mmain run 2 (stochastic inconsistency)
df_s2_clean = df_s2[['病案号', 'disease_vessel_count']].copy()
df_s2_clean.columns = ['hadm_id', 's2_pred']
df_s2_clean['hadm_id'] = df_s2_clean['hadm_id'].astype(str)
df_s2_clean['s2_pred'] = df_s2_clean['s2_pred'].apply(clean_label)

# S3: Reasoning critique
df_s3_clean = df_s3[['病案号', 'audit_status']].copy()
df_s3_clean.columns = ['hadm_id', 'audit_result']
df_s3_clean['hadm_id'] = df_s3_clean['hadm_id'].astype(str)
df_s3_clean['signal_s3'] = ~df_s3_clean['audit_result'].apply(is_safe)

# ========================== Merge ==========================

data = df_gold_clean.merge(df_main_clean, on='hadm_id', how='inner')
data = data.merge(df_s1_clean, on='hadm_id', how='inner')
data = data.merge(df_s2_clean, on='hadm_id', how='inner')
data = data.merge(df_s3_clean, on='hadm_id', how='inner')
print(f"Merged sample size: {len(data)} cases")

# ========================== Signal Computation ==========================

data['is_error'] = (data['main_pred'] != data['gold_label']) | data['main_pred'].isna()
data['signal_s1'] = (data['main_pred'] != data['s1_pred']).fillna(False)
data['signal_s2'] = (data['main_pred'] != data['s2_pred']).fillna(False)
data['signal_s3'] = data['signal_s3'].fillna(False)
data['flag_union'] = data['signal_s1'] | data['signal_s2'] | data['signal_s3']

# ========================== Metrics ==========================

n_total = len(data)
n_errors = int(data['is_error'].sum())
original_acc = 1 - (n_errors / n_total)

review_rate = data['flag_union'].mean()
tp = int((data['is_error'] & data['flag_union']).sum())
error_coverage = tp / n_errors if n_errors > 0 else 0
final_acc = 1 - ((n_errors - tp) / n_total)
efficiency_ratio = error_coverage / (review_rate + 1e-5)

# ========================== Report ==========================

print("\n" + "=" * 62)
print("  Diseased Vessel Counting — Validation Report")
print("  Strategy: S1 (Model Heterogeneity) ∪ S2 (Stochastic")
print("            Inconsistency) ∪ S3 (Reasoning Critique)")
print("=" * 62)
print(f"Total cases:     {n_total}")
print(f"Initial errors:  {n_errors}  (baseline accuracy: {original_acc * 100:.2f}%)")
print("-" * 62)
print(f"Review rate:     {review_rate:.4f}  ({review_rate * 100:.2f}%)")
print(f"Error coverage:  {error_coverage:.4f}  ({error_coverage * 100:.2f}%)")
print(f"TP (captured):   {tp} / {n_errors}")
print("-" * 62)
print(f"Efficiency ratio: {efficiency_ratio:.4f}")
print(f"Final accuracy:   {final_acc:.4f}  ({final_acc * 100:.2f}%)")
print("=" * 62)

# Per-signal breakdown
print("\n--- Per-Signal Trigger Rates ---")
for name, col in [("S1 (Model Heterogeneity)", "signal_s1"),
                  ("S2 (Stochastic Inconsistency)", "signal_s2"),
                  ("S3 (Reasoning Critique)", "signal_s3"),
                  ("Union (S1∪S2∪S3)", "flag_union")]:
    n_triggered = int(data[col].sum())
    print(f"{name}: {n_triggered}  ({data[col].mean() * 100:.2f}%)")

# Error capture by signal
print("\n--- Error Capture by Signal (among actual errors) ---")
errors_only = data[data['is_error']]
for name, col in [("S1", "signal_s1"), ("S2", "signal_s2"),
                  ("S3", "signal_s3"), ("Union", "flag_union")]:
    print(f"{name}: {int(errors_only[col].sum())} / {len(errors_only)}")

# Silent errors (missed by all signals)
silent = data[(data['is_error']) & (~data['flag_union'])]
if len(silent) > 0:
    print(f"\n>>> WARNING: {len(silent)} silent error(s) found")
    display_cols = ['hadm_id', 'main_pred', 'gold_label', 's1_pred', 's2_pred', 'audit_result']
    print(silent[display_cols].to_string(index=False))
else:
    print("\n>>> No silent errors detected.")

# ========================== Save Results ==========================

output_path = "outputs/disease_vessels_validation_result.csv"
data.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\nResults saved to: {output_path}")

if len(silent) > 0:
    silent_path = "outputs/disease_vessels_silent_errors.csv"
    silent.to_csv(silent_path, index=False, encoding='utf-8-sig')
    print(f"Silent errors saved to: {silent_path}")
