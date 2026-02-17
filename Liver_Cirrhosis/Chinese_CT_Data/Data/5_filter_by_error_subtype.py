"""
Step 5: Remove cases from ct_reports_final.csv whose error_subtype in the
inconsistency analysis is 'key_finding_not_described' or 'external_info_required'.
These cases have information gaps that make accurate classification impossible
from the CT findings alone.
"""

import pandas as pd

ct_reports = pd.read_csv("ct_reports_final.csv", encoding='utf-8-sig')
inconsistency = pd.read_csv("inconsistency_analysis_result.csv", encoding='utf-8-sig')

print(f"ct_reports_final.csv:           {len(ct_reports)} rows")
print(f"inconsistency_analysis_result:  {len(inconsistency)} rows")

# Identify patient_ids to remove
to_delete = inconsistency[
    inconsistency['error_subtype'].isin(['key_finding_not_described', 'external_info_required'])
]
delete_ids = set(to_delete['patient_id'].tolist())

n_knfd = (inconsistency['error_subtype'] == 'key_finding_not_described').sum()
n_eir = (inconsistency['error_subtype'] == 'external_info_required').sum()
print(f"\nIDs to remove: {len(delete_ids)} "
      f"(key_finding_not_described: {n_knfd}, external_info_required: {n_eir})")

# Filter
ct_reports_filtered = ct_reports[~ct_reports['patient_id'].isin(delete_ids)]

print(f"\nAfter filtering: {len(ct_reports_filtered)} rows "
      f"(removed {len(ct_reports) - len(ct_reports_filtered)})")

ct_reports_filtered.to_csv("ct_reports_final.csv", index=False, encoding='utf-8-sig')
print("Saved: ct_reports_final.csv")

print(f"\nRemoved patient_ids:")
for tid in sorted(delete_ids):
    print(f"  {tid}")
