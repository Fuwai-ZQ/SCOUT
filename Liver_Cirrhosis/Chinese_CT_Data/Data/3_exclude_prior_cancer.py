"""
Step 3: Exclude patients with prior cancer history (prior_cancer_history == 'yes')
from the deduplicated cohort based on LLM classification results.
Handles duplicate IDs in classification with configurable dedup strategies.
"""

import pandas as pd
import os
from collections import Counter

# ==================== Configuration ====================

ORIGINAL_FILE = "portal_cohort_deduped.xlsx"
CLASSIFICATION_FILE = "liver_cancer_classification_result.csv"
OUTPUT_FILE = "ct_reports_final.csv"
ID_COLUMN = "patient_id"

# Dedup strategy for classification results: 'conservative', 'majority', 'latest'
DEDUP_STRATEGY = 'conservative'

# ==================== Helper ====================

def get_unique_classification(df, strategy='conservative'):
    """Deduplicate classification results per patient_id."""
    result_list = []

    for tid in df[ID_COLUMN].unique():
        subset = df[df[ID_COLUMN] == tid]

        if len(subset) == 1:
            row = subset.iloc[0].to_dict()
        else:
            if strategy == 'conservative':
                # If any record says 'yes', mark as 'yes'
                prior_values = subset['prior_cancer_history'].tolist()
                if 'yes' in prior_values:
                    prior_result = 'yes'
                elif 'uncertain' in prior_values:
                    prior_result = 'uncertain'
                else:
                    prior_result = 'no'

                # Take the most severe lesion category
                lesion_values = subset['liver_lesion_category'].tolist()
                severity_order = [
                    'confirmed_liver_cancer', 'high_risk', 'suspicious_lesion',
                    'benign_lesion', 'no_focal_lesion', 'uncertain'
                ]
                lesion_result = next(
                    (s for s in severity_order if s in lesion_values),
                    lesion_values[0]
                )

                row = subset.iloc[0].to_dict()
                row['prior_cancer_history'] = prior_result
                row['liver_lesion_category'] = lesion_result
                row['_dedup_note'] = f'merged from {len(subset)} records (conservative)'

            elif strategy == 'majority':
                row = subset.iloc[0].to_dict()
                row['prior_cancer_history'] = Counter(
                    subset['prior_cancer_history']).most_common(1)[0][0]
                row['liver_lesion_category'] = Counter(
                    subset['liver_lesion_category']).most_common(1)[0][0]
                row['_dedup_note'] = f'merged from {len(subset)} records (majority)'

            elif strategy == 'latest':
                if 'exam_date' in subset.columns:
                    row = subset.sort_values('exam_date', ascending=False).iloc[0].to_dict()
                    row['_dedup_note'] = 'latest record'
                else:
                    row = subset.iloc[-1].to_dict()
                    row['_dedup_note'] = 'last record (no date column)'
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        result_list.append(row)

    return pd.DataFrame(result_list)


# ==================== Main ====================

def main():
    for f in [ORIGINAL_FILE, CLASSIFICATION_FILE]:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            return

    # Load data
    if ORIGINAL_FILE.endswith(('.xlsx', '.xls')):
        df_original = pd.read_excel(ORIGINAL_FILE)
    else:
        try:
            df_original = pd.read_csv(ORIGINAL_FILE, encoding="utf-8-sig")
        except Exception:
            df_original = pd.read_csv(ORIGINAL_FILE, encoding="utf-8")

    try:
        df_classification = pd.read_csv(CLASSIFICATION_FILE, encoding="utf-8-sig")
    except Exception:
        df_classification = pd.read_csv(CLASSIFICATION_FILE, encoding="utf-8")

    # Data quality check
    print(f"Original data:      {len(df_original)} rows, "
          f"{df_original[ID_COLUMN].nunique()} unique IDs")
    print(f"Classification data: {len(df_classification)} rows, "
          f"{df_classification[ID_COLUMN].nunique()} unique IDs")

    # Check for inconsistent duplicate classifications
    dup_ids = df_classification[
        df_classification[ID_COLUMN].duplicated(keep=False)
    ][ID_COLUMN].unique()
    inconsistent = [
        tid for tid in dup_ids
        if df_classification[df_classification[ID_COLUMN] == tid]['prior_cancer_history'].nunique() > 1
    ]
    if inconsistent:
        print(f"Warning: {len(inconsistent)} IDs have inconsistent prior_cancer_history across duplicates")

    # Deduplicate classification results
    df_class_unique = get_unique_classification(df_classification, strategy=DEDUP_STRATEGY)
    print(f"\nClassification after dedup: {len(df_class_unique)} (strategy: {DEDUP_STRATEGY})")
    print(f"\nLesion category distribution:\n{df_class_unique['liver_lesion_category'].value_counts().to_string()}")
    print(f"\nPrior cancer history distribution:\n{df_class_unique['prior_cancer_history'].value_counts().to_string()}")

    # Exclude patients with prior cancer history
    exclude_records = df_class_unique[df_class_unique['prior_cancer_history'] == 'yes']
    exclude_ids = set(exclude_records[ID_COLUMN].astype(str).tolist())
    print(f"\nExcluding {len(exclude_ids)} IDs with prior_cancer_history == 'yes'")

    # Deduplicate original data and apply exclusion
    df_original_unique = df_original.drop_duplicates(subset=[ID_COLUMN], keep='first')
    df_original_unique['_id_str'] = df_original_unique[ID_COLUMN].astype(str)
    df_filtered = df_original_unique[~df_original_unique['_id_str'].isin(exclude_ids)].copy()
    df_filtered = df_filtered.drop(columns=['_id_str'])

    excluded_count = len(df_original_unique) - len(df_filtered)

    print(f"\nSummary:")
    print(f"  Original (deduped):  {len(df_original_unique)}")
    print(f"  Excluded:            {excluded_count}")
    print(f"  Final:               {len(df_filtered)}")

    df_filtered.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {OUTPUT_FILE}")

    df_class_unique.to_csv("classification_deduped.csv", index=False, encoding='utf-8-sig')
    print(f"Saved: classification_deduped.csv")

    # Print excluded record details (first 20)
    print(f"\nExcluded records (showing up to 20):")
    for _, row in exclude_records.head(20).iterrows():
        kf = str(row.get('key_findings', ''))[:100] if pd.notna(row.get('key_findings')) else ''
        print(f"  {row[ID_COLUMN]} | {row['liver_lesion_category']} | findings: {kf}")

    if len(exclude_records) > 20:
        print(f"  ... {len(exclude_records) - 20} more not shown")


if __name__ == "__main__":
    main()
