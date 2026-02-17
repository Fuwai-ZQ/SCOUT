"""
Filter parsed CT reports by:
  1. Requiring non-empty indication, comparison, findings, impression.
  2. Excluding records containing cancer/treatment-related keywords
     in indication, comparison, or findings.
"""

import pandas as pd
import re
from collections import defaultdict

print("=" * 70)
print("CT Report Filtering")
print("=" * 70)

df = pd.read_csv("path/to/ct_reports_parsed.csv")
print(f"\nRaw data: {len(df)} records")

# =============================================================================
# Step 1: Drop records with missing required fields
# =============================================================================
required_fields = ['indication', 'comparison', 'findings', 'impression']

print("\n" + "-" * 50)
print("Step 1: Required-field completeness check")
print("-" * 50)

for field in required_fields:
    non_empty = df[field].notna() & (df[field].str.strip() != '')
    print(f"  {field}: {non_empty.sum()} non-empty")

mask_non_empty = pd.Series([True] * len(df))
for field in required_fields:
    mask_non_empty &= df[field].notna() & (df[field].astype(str).str.strip() != '')

df_step1 = df[mask_non_empty].copy()
excluded_step1 = len(df) - len(df_step1)
print(f"\n  Kept (all fields present): {len(df_step1)}")
print(f"  Excluded (missing fields): {excluded_step1}")

# =============================================================================
# Step 2: Define exclusion keywords
# =============================================================================

# A. Cancer & malignancy
cancer_keywords = [
    r'\bHCC\b', r'\bHepatoma\b', r'\bCarcinoma\b',
    r'\bMalignan', r'\bMetasta', r'\bNeoplasm',
    r'\bTumor\b', r'\bTumour\b', r'\bCholangiocarcinoma\b',
    r'\bLesion.*suspicious', r'\bsuspicious.*lesion',
    r'\bCancer\b', r'\bOncolog',
]

# B. Interventional & locoregional therapy
intervention_keywords = [
    r'\bTACE\b', r'\bChemoemboli', r'\bEmboli[zs]ation\b',
    r'\bAblation\b', r'\bRFA\b', r'\bMWA\b',
    r'\bLipiodol\b', r'\bEthiodol\b', r'\bCryoablation\b',
    r'\bRadiofrequency\b', r'\bMicrowave\b',
]

# C. Chemo-/radiotherapy & systemic treatment
chemo_radio_keywords = [
    r'\bChemo', r'\bRadiation\b', r'\bRadiotherapy\b',
    r'\bY-90\b', r'\bYttrium\b',
    r'\bSorafenib\b', r'\bNexavar\b', r'\bLenvatinib\b',
    r'\bDoxorubicin\b', r'\bCisplatin\b',
    r'\bImmunotherapy\b', r'\bResection\b', r'\bHepate?ctomy\b',
    r'\bTargeted\s+therap',
]

all_keywords = cancer_keywords + intervention_keywords + chemo_radio_keywords

# =============================================================================
# Step 3: Keyword-based exclusion
# =============================================================================
print("\n" + "-" * 50)
print("Step 2: Keyword exclusion")
print("-" * 50)

check_fields = ['indication', 'comparison', 'findings']

keyword_hits = defaultdict(int)
category_hits = {
    'Cancer & malignancy': 0,
    'Interventional therapy': 0,
    'Chemo-/radiotherapy': 0,
}

exclusion_reasons = []


def check_text(text, keywords_list):
    if pd.isna(text) or not str(text).strip():
        return []
    text = str(text)
    return [kw for kw in keywords_list if re.search(kw, text, re.IGNORECASE)]


def categorize_keyword(kw):
    if kw in cancer_keywords:
        return 'Cancer & malignancy'
    if kw in intervention_keywords:
        return 'Interventional therapy'
    if kw in chemo_radio_keywords:
        return 'Chemo-/radiotherapy'
    return 'Other'


excluded_indices = []
for idx, row in df_step1.iterrows():
    found_keywords = set()
    found_in_fields = {}

    for field in check_fields:
        found = check_text(row[field], all_keywords)
        if found:
            found_keywords.update(found)
            found_in_fields[field] = found

    if found_keywords:
        excluded_indices.append(idx)
        for kw in found_keywords:
            keyword_hits[kw] += 1
            category_hits[categorize_keyword(kw)] += 1
        exclusion_reasons.append({
            'index': idx,
            'keywords': list(found_keywords),
            'fields': found_in_fields,
        })

df_step2 = df_step1.drop(index=excluded_indices).copy()
excluded_step2 = len(excluded_indices)

print(f"\n  Checked fields: {check_fields}")
print(f"  Excluded by keywords: {excluded_step2}")
print(f"  Retained: {len(df_step2)}")

# =============================================================================
# Statistics
# =============================================================================
print("\n" + "=" * 70)
print("Exclusion statistics")
print("=" * 70)

print("\n[By category] (one record may trigger multiple)")
for cat, count in sorted(category_hits.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

print("\n[By keyword] (top 20)")
for kw, count in sorted(keyword_hits.items(), key=lambda x: -x[1])[:20]:
    display_kw = kw.replace(r'\b', '').replace(r'.*', '*')
    print(f"  {display_kw}: {count}")

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"  Raw data:              {len(df):>6}")
print(f"  Excluded (empty):      {excluded_step1:>6}")
print(f"  Excluded (keywords):   {excluded_step2:>6}")
print(f"  ─────────────────────────")
print(f"  Retained:              {len(df_step2):>6}")
print(f"  Total excluded:        {len(df) - len(df_step2):>6} "
      f"({(len(df) - len(df_step2)) / len(df) * 100:.1f}%)")

# =============================================================================
# Save
# =============================================================================
output_file = "path/to/ct_reports_filtered.csv"
df_step2.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\nFiltered data saved: {output_file}")

exclusion_df = pd.DataFrame([
    {
        'original_index': r['index'],
        'matched_keywords': ', '.join(r['keywords']),
        'matched_fields': str(r['fields']),
    }
    for r in exclusion_reasons
])
exclusion_file = "path/to/excluded_records_detail.csv"
exclusion_df.to_csv(exclusion_file, index=False, encoding='utf-8-sig')
print(f"Exclusion detail saved: {exclusion_file}")
