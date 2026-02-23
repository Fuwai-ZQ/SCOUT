"""
Step 1: Deduplicate the raw cohort by patient_id, keeping the first record.
"""

import pandas as pd

INPUT_FILE = "portal_cohort_raw.xlsx"
OUTPUT_FILE = "portal_cohort_deduped.xlsx"
ID_COLUMN = "patient_id"

df = pd.read_excel(INPUT_FILE)

print(f"Before dedup: {len(df)} rows, {df[ID_COLUMN].nunique()} unique IDs, "
      f"{len(df) - df[ID_COLUMN].nunique()} duplicates")

df_unique = df.drop_duplicates(subset=[ID_COLUMN], keep='first')

print(f"After dedup:  {len(df_unique)} rows")

df_unique.to_excel(OUTPUT_FILE, index=False)
print(f"Saved: {OUTPUT_FILE}")
