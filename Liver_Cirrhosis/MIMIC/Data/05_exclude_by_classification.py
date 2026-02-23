"""
Exclude records based on liver lesion classification results.

Exclusion criteria:
  1. liver_lesion_category == 'uncertain'
  2. prior_cancer_history == 'yes'
"""

import pandas as pd
import os

print("=" * 70)
print("Exclude Records by Classification")
print("=" * 70)

ORIGINAL_FILE       = "path/to/ct_reports_filtered.csv"
CLASSIFICATION_FILE = "path/to/liver_cancer_classification_result.csv"
OUTPUT_FILE         = "path/to/ct_reports_final.csv"
ID_COLUMN           = "hadm_id"

for f in [ORIGINAL_FILE, CLASSIFICATION_FILE]:
    if not os.path.exists(f):
        print(f"File not found: {f}")
        exit(1)

try:
    df_original = pd.read_csv(ORIGINAL_FILE, encoding="utf-8-sig")
except Exception:
    df_original = pd.read_csv(ORIGINAL_FILE, encoding="utf-8")

try:
    df_cls = pd.read_csv(CLASSIFICATION_FILE, encoding="utf-8-sig")
except Exception:
    df_cls = pd.read_csv(CLASSIFICATION_FILE, encoding="utf-8")

print(f"\nOriginal data: {len(df_original)} records")
print(f"Classification data: {len(df_cls)} records")

# Current distribution
print("\n" + "-" * 50)
print("Classification distribution:")
print("-" * 50)
print("\n[Liver lesion category]")
print(df_cls['liver_lesion_category'].value_counts().to_string())
print("\n[Prior cancer history]")
print(df_cls['prior_cancer_history'].value_counts().to_string())

# Identify records to exclude
mask_uncertain = df_cls['liver_lesion_category'] == 'uncertain'
mask_prior_yes = df_cls['prior_cancer_history'] == 'yes'
mask_exclude   = mask_uncertain | mask_prior_yes

exclude_records = df_cls[mask_exclude]
exclude_ids = set(exclude_records[ID_COLUMN].astype(str).tolist())

print(f"\nUncertain category: {mask_uncertain.sum()}")
print(f"Prior cancer = yes: {mask_prior_yes.sum()}")

overlap = set(df_cls.loc[mask_uncertain, ID_COLUMN].astype(str)) & \
          set(df_cls.loc[mask_prior_yes, ID_COLUMN].astype(str))
if overlap:
    print(f"Overlap IDs: {overlap}")
print(f"Unique IDs to exclude: {len(exclude_ids)}")

# Filter
df_original['_id_str'] = df_original[ID_COLUMN].astype(str)
df_filtered = df_original[~df_original['_id_str'].isin(exclude_ids)].copy()
df_filtered.drop(columns=['_id_str'], inplace=True)
excluded_count = len(df_original) - len(df_filtered)

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"  Original:              {len(df_original):>6}")
print(f"  Uncertain excluded:    {mask_uncertain.sum():>6}")
print(f"  Prior cancer excluded: {mask_prior_yes.sum():>6}")
print(f"  Actually excluded:     {excluded_count:>6}")
print(f"  Retained:              {len(df_filtered):>6}")

df_filtered.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\nSaved: {OUTPUT_FILE}")

# Exclusion detail
print("\n" + "-" * 50)
print("Excluded records:")
print("-" * 50)
for _, row in exclude_records.iterrows():
    reasons = []
    if row['liver_lesion_category'] == 'uncertain':
        reasons.append("uncertain category")
    if row['prior_cancer_history'] == 'yes':
        reasons.append("prior cancer = yes")
    print(f"\n  {ID_COLUMN}: {row[ID_COLUMN]}")
    print(f"  Reason: {', '.join(reasons)}")
    print(f"  Category: {row['liver_lesion_category']}  |  Prior: {row['prior_cancer_history']}")
    if 'key_findings' in row and pd.notna(row['key_findings']):
        print(f"  Key findings: {row['key_findings']}")
