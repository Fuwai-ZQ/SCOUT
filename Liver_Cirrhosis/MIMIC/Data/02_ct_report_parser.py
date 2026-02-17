"""
Structured parser for radiology CT reports from the MIMIC-IV database.

Extracts standard report sections (Examination, Indication, Technique,
Findings, Impression, etc.) and optionally parses Findings sub-sections.
"""

import pandas as pd
import re
from typing import Optional, Dict, List, Tuple


class CTReportParser:
    """Extracts structured sections from free-text CT reports."""

    SECTION_PATTERNS = {
        'examination':    r'^(?:EXAMINATION|EXAM|STUDY)\s*:',
        'indication':     r'^(?:INDICATION(?:\s+OF\s+THE\s+STUDY)?|CLINICAL\s+(?:INDICATION|HISTORY)|HISTORY|REASON\s+FOR\s+EXAM(?:INATION)?)\s*:',
        'technique':      r'^(?:TECHNIQUE|PROTOCOL)\s*:',
        'dose':           r'^(?:DOSE|RADIATION\s+DOSE)\s*:',
        'comparison':     r'^(?:COMPARISONS?|PRIOR\s+STUD(?:Y|IES))\s*:',
        'findings':       r'^FINDINGS\s*:',
        'findings_alt':   r'^CT\s+(?:OF\s+THE\s+)?(?:ABDOMEN|PELVIS|ABD)[^:]*:',
        'impression':     r'^(?:IMPRESSION|CONCLUSION|OPINION|SUMMARY)\s*:',
        'recommendation': r'^RECOMMENDATION(?:\(S\))?\s*:',
        'notification':   r'^(?:NOTIFICATION|CRITICAL\s+RESULTS?|COMMUNICATION)\s*:',
    }

    FINDINGS_SUBSECTIONS = [
        'LOWER CHEST', 'CHEST', 'LUNGS', 'ABDOMEN',
        'HEPATOBILIARY', 'LIVER', 'PANCREAS', 'SPLEEN',
        'ADRENALS', 'ADRENAL GLANDS', 'URINARY', 'KIDNEYS',
        'GASTROINTESTINAL', 'GI', 'PERITONEUM', 'PELVIS',
        'REPRODUCTIVE ORGANS', 'LYMPH NODES',
        'VASCULAR', 'VESSELS',
        'BONES', 'OSSEOUS STRUCTURES', 'BONE WINDOWS',
        'SOFT TISSUES',
    ]

    def __init__(self):
        self.section_regex = {
            name: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for name, pattern in self.SECTION_PATTERNS.items()
        }
        all_patterns = '|'.join(self.SECTION_PATTERNS.values())
        self.all_headers_regex = re.compile(all_patterns, re.IGNORECASE | re.MULTILINE)

    def parse(self, text: str) -> Dict[str, Optional[str]]:
        """Parse a single CT report into section fields."""
        result = {k: None for k in [
            'examination', 'indication', 'technique', 'dose',
            'comparison', 'findings', 'impression',
            'recommendation', 'notification',
        ]}

        if pd.isna(text) or not str(text).strip():
            return result

        text = str(text).replace('\r\n', '\n').replace('\r', '\n')

        # Locate all section headers
        found_sections: List[Tuple[int, int, str]] = []
        for field_name, regex in self.section_regex.items():
            for match in regex.finditer(text):
                found_sections.append((match.start(), match.end(), field_name))
        found_sections.sort(key=lambda x: x[0])

        # Deduplicate overlapping matches (keep the longer one)
        deduped = []
        for sec in found_sections:
            if not deduped or sec[0] >= deduped[-1][1]:
                deduped.append(sec)
            elif sec[1] > deduped[-1][1]:
                deduped[-1] = sec
        found_sections = deduped

        # Extract content between consecutive headers
        for i, (start, header_end, field_name) in enumerate(found_sections):
            content_end = found_sections[i + 1][0] if i + 1 < len(found_sections) else len(text)
            content = text[header_end:content_end].strip()
            actual_field = 'findings' if field_name == 'findings_alt' else field_name
            if content and (result[actual_field] is None or len(content) > len(result[actual_field] or '')):
                result[actual_field] = content

        return result

    def extract_findings_subsections(self, findings_text: str) -> Dict[str, str]:
        """Parse organ-level sub-sections within Findings."""
        if not findings_text:
            return {}

        pattern = r'^(' + '|'.join(self.FINDINGS_SUBSECTIONS) + r')\s*:'
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        matches = list(regex.finditer(findings_text))

        result = {}
        for i, match in enumerate(matches):
            section_name = match.group(1).upper().replace(' ', '_')
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(findings_text)
            content = findings_text[start:end].strip()
            if content:
                result[section_name] = content
        return result


def parse_batch(df: pd.DataFrame,
                text_column: str = 'text',
                extract_subsections: bool = False,
                verbose: bool = True) -> pd.DataFrame:
    """Parse CT reports in batch and return a structured DataFrame."""
    parser = CTReportParser()
    if verbose:
        print(f"Parsing {len(df)} records ...")

    parsed_results = []
    for idx, text in enumerate(df[text_column]):
        result = parser.parse(text)
        if extract_subsections and result.get('findings'):
            subsections = parser.extract_findings_subsections(result['findings'])
            result.update({f'findings_{k.lower()}': v for k, v in subsections.items()})
        parsed_results.append(result)
        if verbose and (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(df)}")

    if verbose:
        print("Parsing complete.")

    parsed_df = pd.DataFrame(parsed_results)
    id_columns = [col for col in df.columns if col != text_column]
    result_df = pd.concat([df[id_columns].reset_index(drop=True), parsed_df], axis=1)
    result_df['original_text'] = df[text_column].values
    return result_df


def analyze_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Report section-level extraction statistics."""
    fields = ['examination', 'indication', 'technique', 'dose',
              'comparison', 'findings', 'impression', 'recommendation', 'notification']
    stats = []
    for field in fields:
        if field in df.columns:
            non_empty = (df[field].fillna('').str.len() > 0).sum()
            avg_len = df[field].dropna().str.len().mean() if non_empty > 0 else 0
            stats.append({
                'field': field,
                'non_empty': non_empty,
                'success_rate': f"{non_empty / len(df) * 100:.1f}%",
                'avg_length': f"{avg_len:.0f} chars"
            })
    return pd.DataFrame(stats)


def show_sample(df: pd.DataFrame, idx: int = 0, max_length: int = 400):
    """Print parsed fields for a single record."""
    print(f"\n{'=' * 70}")
    print(f"Record {idx + 1}:")
    print('=' * 70)
    row = df.iloc[idx]
    for field in ['examination', 'indication', 'technique', 'dose',
                  'comparison', 'findings', 'impression', 'recommendation', 'notification']:
        if field in df.columns and pd.notna(row[field]) and row[field]:
            content = str(row[field])
            if len(content) > max_length:
                content = content[:max_length] + f"\n... ({len(row[field])} chars total)"
            print(f"\n[{field.upper()}]")
            print(content)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    INPUT_FILE  = "path/to/cirrhosis_plain_ct.csv"
    OUTPUT_FILE = "path/to/ct_reports_parsed.csv"
    TEXT_COLUMN = "text"

    print(f"Reading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"  {len(df)} records loaded")

    result_df = parse_batch(df, text_column=TEXT_COLUMN, extract_subsections=False)

    print("\nExtraction quality:")
    stats = analyze_quality(result_df)
    print(stats.to_string(index=False))

    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {OUTPUT_FILE}")

    for i in [0, 1, 9]:
        show_sample(result_df, i, max_length=300)
