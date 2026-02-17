"""
Liver cancer risk prediction — optimized prompt with chain-of-thought reasoning.

Uses guideline-derived clinical knowledge and explicit reasoning directives
to classify CT findings into four risk categories (Strategy S1 main model).
"""

import os
import json
import time
import csv
import pandas as pd
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =============================================================================
# Configuration
# =============================================================================

API_KEY    = os.getenv("DASHSCOPE_API_KEY", "sk-XXX")
BASE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v3.1"

INPUT_CSV_FILE  = "path/to/ct_reports_final.csv"
OUTPUT_CSV_FILE = "path/to/liver_cancer_prediction_with_reasoning.csv"

ID_COLUMN        = "hadm_id"
MAX_WORKERS      = 100
CALLS_PER_MINUTE = 99
WINDOW_SECONDS   = 65

file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# =============================================================================
# Prompt (guideline-enhanced with CoT reasoning)
# =============================================================================

SYSTEM_PROMPT = """You are a Senior Hepatic Imaging Specialist focused on diagnosing complications in cirrhotic patients through non-contrast CT imaging. Your task is to sensitively identify deceptive malignant lesions while maintaining high specificity (avoiding false positives).

## CORE LOGIC UPGRADES
1. **Distinguish "Description" from "Conclusion"**: "Cyst-like" in imaging description ≠ "Cyst" in conclusion. If the conclusion is indeterminate, treat as high risk.
2. **Beware of "Mixed Features"**: Benign features (e.g., calcification, cystic change) occurring WITHIN a solid nodule are actually signs of malignancy.
3. **Heed "Radiologist Hesitation"**: Any description containing "indeterminate", "recommend contrast-enhanced imaging", or "?" should be escalated to high risk in cirrhotic patients.

## INTERPRETATION CRITERIA (REFINED VERSION)

### I. Absolute High Risk (Confirmed/High Risk)
1. **Definitive Malignant Keywords**: Liver cancer, HCC, hepatocellular carcinoma, MT, mass/occupying lesion, malignant tumor, new lesion.
2. **Typical Malignant Morphology**:
   - Mass with portal vein tumor thrombus (filling defect).
   - Heterogeneous hypodense lesion (suggesting necrosis/hemorrhage).
   - Ill-defined borders, lobulated contour, or infiltrative growth pattern.
3. **[KEY IMPROVEMENT] Malignant Features Masquerading as Benign**:
   - **Impure Calcification**: Described as "nodular mixed-density lesion with internal calcification" (not simple calcific focus).
   - **Impure Cystic Lesion**: Described as "cystic-solid", "thickened cyst wall", "internal septations", or "cyst-like hypodensity, nature indeterminate".
   - **Dangerous Diffuse Disease**: Described as "diffuse nodules" WITH concurrent "portal vein tumor thrombus/filling defect" OR "new-onset large volume ascites".

### II. Typical Benign (Benign Lesion) - Must Strictly Meet the Following Conditions
**Only classify as benign (confidently exclude malignancy) if COMPLETELY meeting these criteria:**

1. **Confirmed Cyst**:
   - Report conclusion explicitly states "hepatic cyst".
   - OR described as "no wall", "water density" AND "sharply demarcated/well-defined borders".
   - **Note**: If only described as "cyst-like" but conclusion states "nature indeterminate", this does NOT qualify as benign—classify as high risk.

2. **Simple Calcific Focus**:
   - Described as "high-density calcific focus", "intrahepatic calcification point".
   - **Exclude**: Predominantly hypodense nodule with only internal punctate calcification (classify as high risk).

3. **Hemangioma**:
   - Report explicitly mentions "hemangioma" or "probable hemangioma".

4. **Old/Post-treatment Changes**:
   - Explicitly mentions "post-surgical changes", "lipiodol deposition" (high density), "no change after treatment".

### III. No Focal Lesion
- Only cirrhotic background changes (irregular liver surface, altered proportions, regenerative nodules).
- **Note**: If report mentions "diffuse hypodense nodules" without definitive benign diagnosis, AND concurrent portal vein trunk obscuration or filling defect, escalate to high risk; otherwise may be considered background.

---

## CHAIN-OF-THOUGHT (CoT) REASONING REQUIREMENTS
In `diagnostic_reasoning`, please perform the following checks:
1. **Semantic Check**: Is this "cyst" (diagnosis) or "cyst-like" (description)? If the latter without definitive conclusion, lean toward high risk.
2. **Purity Check**: Is this "pure calcification" or "nodule containing calcification"? The latter is high risk.
3. **Temporal Check**: Is this "unchanged from prior" (benign tendency) or "new/enlarged" (malignant tendency)?
4. **Conclusion Verification**: Does the radiologist's final impression include "nature indeterminate", "recommend further workup"? If yes, MUST classify as High Risk.

## OUTPUT FORMAT (JSON)
{
  "cirrhosis_signs": "Summary of cirrhotic morphology and portal hypertension signs",
  "plain_ct_findings": {
    "focal_lesion_detected": "Yes/No",
    "lesion_description": "Detailed description of any lesions found",
    "suspicious_indicators": ["List specific warning signs, e.g.: nature indeterminate, ill-defined borders, nodule with calcification"]
  },
  "diagnostic_reasoning": "Step-by-step reasoning following the semantic/purity/temporal/conclusion checks",
  "risk_assessment": {
    "category": "One of: confirmed_liver_cancer / high_risk / benign_lesion / no_focal_lesion",
    "confidence_score": "0-100",
    "key_evidence": "The most critical text segment supporting the category"
  }
}

Please output JSON only, without any markdown code blocks or additional text.
"""

# =============================================================================
# Utilities
# =============================================================================

class ThreadSafeRateLimiter:
    """Sliding-window rate limiter safe for concurrent use."""

    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.ts = deque()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            while self.ts and (now - self.ts[0]) >= self.window_seconds:
                self.ts.popleft()
            if len(self.ts) >= self.max_calls:
                sleep_s = self.window_seconds - (now - self.ts[0]) + 0.1
                if sleep_s > 0:
                    time.sleep(sleep_s)
                now = time.time()
                while self.ts and (now - self.ts[0]) >= self.window_seconds:
                    self.ts.popleft()
            self.ts.append(time.time())


def parse_json_from_content(content: str):
    """Extract JSON object from raw LLM output."""
    text = content.strip()
    if "```json" in text:
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    elif text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        if text.endswith("```"):
            text = text[:-3].strip()
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
    return None


def call_model(indication, comparison, findings, limiter):
    """Send report sections to the LLM and return structured prediction."""
    user_content = f"""Please analyze the following CT report information and predict whether the patient has hepatic malignancy:

## Clinical Indication
{indication}

## Comparison
{comparison}

## Imaging Findings
{findings}

Please output your analysis results in the required JSON format."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    retries, wait = 3, 5
    for attempt in range(retries):
        limiter.wait()
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                extra_body={"enable_thinking": True},
                stream=True, stream_options={"include_usage": True},
                temperature=0.6, top_p=0.95,
            )
            full_answer, full_reasoning = "", ""
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    full_reasoning += delta.reasoning_content
                if hasattr(delta, 'content') and delta.content:
                    full_answer += delta.content

            data = parse_json_from_content(full_answer)
            if data is None:
                return {
                    "parse_error": True, "raw_response": full_answer,
                    "thinking_trace": full_reasoning,
                    "risk_assessment": {"category": "PARSE_ERROR",
                                        "confidence_score": "", "key_evidence": ""},
                }
            data["thinking_trace"] = full_reasoning
            data["raw_response"] = full_answer
            return data

        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(wait); wait *= 2

    return {"timeout_error": True,
            "risk_assessment": {"category": "TIMEOUT_ERROR",
                                "confidence_score": "", "key_evidence": ""}}


def safe_append_row(path, fieldnames, row):
    """Thread-safe CSV append."""
    with file_lock:
        exists = os.path.isfile(path) and os.path.getsize(path) > 0
        with open(path, "a" if exists else "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                w.writeheader()
            w.writerow(row)


def process_single_task(row, limiter, fieldnames):
    """Classify one record and write result with full reasoning detail."""
    case_id    = str(row.get(ID_COLUMN, "")).strip()
    indication = str(row.get("indication", "")).strip()
    comparison = str(row.get("comparison", "")).strip()
    findings   = str(row.get("findings", "")).strip()

    try:
        result = call_model(indication, comparison, findings, limiter)
        ct_findings    = result.get("plain_ct_findings", {})
        risk           = result.get("risk_assessment", {})

        # Assemble structured reasoning detail
        parts = []
        if result.get("cirrhosis_signs"):
            parts.append(f"[Cirrhosis Background] {result['cirrhosis_signs']}")

        ct_parts = []
        if ct_findings.get("focal_lesion_detected"):
            ct_parts.append(f"Focal Lesion: {ct_findings['focal_lesion_detected']}")
        if ct_findings.get("lesion_description"):
            ct_parts.append(f"Description: {ct_findings['lesion_description']}")
        indicators = ct_findings.get("suspicious_indicators")
        if indicators:
            ct_parts.append(f"Warning Signs: {', '.join(indicators) if isinstance(indicators, list) else indicators}")
        if ct_parts:
            parts.append(f"[CT Findings] {'; '.join(ct_parts)}")

        if result.get("diagnostic_reasoning"):
            parts.append(f"[Diagnostic Reasoning] {result['diagnostic_reasoning']}")

        risk_parts = []
        if risk.get("confidence_score"):
            risk_parts.append(f"Confidence: {risk['confidence_score']}")
        if risk.get("key_evidence"):
            risk_parts.append(f"Key Evidence: {risk['key_evidence']}")
        if risk_parts:
            parts.append(f"[Risk Assessment] {'; '.join(risk_parts)}")

        out_row = {
            ID_COLUMN: case_id,
            "note_id": row.get("note_id", ""),
            "subject_id": row.get("subject_id", ""),
            "predicted_category": risk.get("category", ""),
            "reasoning_detail": "\n".join(parts),
            "model_thinking": (result.get("thinking_trace", "") or "")[:5000],
        }
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)

        print(f"  {case_id} | {risk.get('category', 'N/A')} | conf={risk.get('confidence_score', 'N/A')}")
        return True

    except Exception as e:
        print(f"  FAILED {case_id}: {e}")
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, {
            ID_COLUMN: case_id, "note_id": row.get("note_id", ""),
            "subject_id": row.get("subject_id", ""),
            "predicted_category": "ERROR",
            "reasoning_detail": f"ERROR: {type(e).__name__} - {e}",
            "model_thinking": "",
        })
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Liver Cancer Risk Prediction — Optimized Prompt with CoT Reasoning")
    print(f"Model: {MODEL_NAME}  |  Input: {INPUT_CSV_FILE}")
    print("=" * 70)

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"Input file not found: {INPUT_CSV_FILE}"); return

    try:
        df = pd.read_csv(INPUT_CSV_FILE, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(INPUT_CSV_FILE, encoding="utf-8")
    print(f"Loaded {len(df)} records")

    fieldnames = [ID_COLUMN, "note_id", "subject_id",
                  "predicted_category", "reasoning_detail", "model_thinking"]

    processed_ids = set()
    if os.path.exists(OUTPUT_CSV_FILE) and os.path.getsize(OUTPUT_CSV_FILE) > 0:
        try:
            done = pd.read_csv(OUTPUT_CSV_FILE, usecols=[ID_COLUMN], encoding="utf-8-sig")
            processed_ids = set(done[ID_COLUMN].astype(str).str.strip())
            print(f"Resuming — {len(processed_ids)} already processed")
        except Exception as e:
            print(f"Checkpoint error: {e}")

    tasks = [
        row for _, row in df.iterrows()
        if str(row.get(ID_COLUMN, "")).strip() not in ("", "nan")
        and str(row.get("findings", "")).strip() not in ("", "nan")
        and len(str(row.get("findings", "")).strip()) >= 20
        and str(row.get(ID_COLUMN, "")).strip() not in processed_ids
    ]
    print(f"Pending: {len(tasks)} | Workers: {MAX_WORKERS} | RPM: {CALLS_PER_MINUTE}")

    if not tasks:
        print("Nothing to process."); return

    limiter = ThreadSafeRateLimiter(CALLS_PER_MINUTE, WINDOW_SECONDS)
    ok, err, t0 = 0, 0, time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(process_single_task, r, limiter, fieldnames) for r in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                ok += fut.result(); err += not fut.result()
            except Exception as e:
                err += 1; print(f"  Exception: {e}")
            if i % 10 == 0 or i == len(futures):
                rate = i / (time.time() - t0) * 60
                print(f"Progress: {i}/{len(futures)} | ok={ok} err={err} | {rate:.1f}/min")

    print(f"\nDone — {len(tasks)} total, {ok} ok, {err} err, "
          f"{(time.time() - t0) / 60:.1f} min")

    if os.path.exists(OUTPUT_CSV_FILE):
        try:
            df_res = pd.read_csv(OUTPUT_CSV_FILE, encoding="utf-8-sig")
            print("\nCategory distribution:")
            print(df_res['predicted_category'].value_counts().to_string())
        except Exception as e:
            print(f"Stats error: {e}")


if __name__ == "__main__":
    main()
