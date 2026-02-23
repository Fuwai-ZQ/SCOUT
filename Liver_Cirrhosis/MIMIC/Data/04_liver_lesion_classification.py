"""
Liver lesion classification via LLM (impression-based).

Classifies CT impression text into four liver lesion categories
and assesses prior cancer history using concurrent API calls.
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
MODEL_NAME = "deepseek-v3.2"

INPUT_CSV_FILE  = "path/to/ct_reports_filtered.csv"
OUTPUT_CSV_FILE = "liver_cancer_classification_result.csv"

ID_COLUMN   = "hadm_id"
TEXT_COLUMN  = "impression"

MAX_WORKERS      = 5
CALLS_PER_MINUTE = 99
WINDOW_SECONDS   = 65
ENABLE_THINKING  = False

file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# =============================================================================
# Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an experienced hepatic pathologist and radiologist. Analyze the CT report impression and classify the liver findings.

## Task 1: Liver Lesion Category (liver_lesion_category)

Classify into one of four categories:

1. **confirmed_liver_cancer**
   - Explicit HCC, cholangiocarcinoma, or hepatic malignancy diagnosis
   - Liver metastasis
   - Terms such as "carcinoma", "malignancy", "malignant" applied to hepatic lesions

2. **high_risk**
   - Suspicious mass or indeterminate nodule requiring further evaluation
   - Phrasing such as "concerning for", "cannot exclude malignancy", "indeterminate"
   - LI-RADS 4 or 5 lesions
   - Nodules showing arterial enhancement or washout

3. **benign_lesion**
   - Confirmed cyst, hemangioma, FNH, lipoma, or other benign entity
   - Stable, well-characterized benign lesion

4. **no_focal_lesion**
   - No focal hepatic lesion described
   - Only diffuse findings (cirrhosis, steatosis, congestion)
   - Explicit "no focal lesion" or equivalent

## Task 2: Prior Cancer History (prior_cancer_history)

- **yes**: Evidence of prior cancer (any site)
- **no**: No indication of prior cancer
- **uncertain**: Suggestive but unconfirmed clues

## Output Format

Respond with JSON only (no markdown, no extra text):
{"liver_lesion_category": "<category>", "prior_cancer_history": "yes/no/uncertain", "confidence": "high/medium/low", "key_findings": "<brief summary in English, <=50 words>"}

## Examples

Input: "1. Hepatocellular carcinoma in segment 6. 2. Cirrhosis with portal hypertension."
Output: {"liver_lesion_category": "confirmed_liver_cancer", "prior_cancer_history": "no", "confidence": "high", "key_findings": "HCC in segment 6"}

Input: "1. 2.3 cm arterially enhancing lesion with washout, concerning for HCC (LI-RADS 4). 2. Cirrhosis."
Output: {"liver_lesion_category": "high_risk", "prior_cancer_history": "no", "confidence": "high", "key_findings": "LI-RADS 4 lesion suspicious for HCC"}

Input: "1. Simple hepatic cyst. 2. Cirrhosis with ascites."
Output: {"liver_lesion_category": "benign_lesion", "prior_cancer_history": "no", "confidence": "high", "key_findings": "Simple hepatic cyst"}

Input: "1. Cirrhotic liver without focal lesion. 2. Splenomegaly and ascites."
Output: {"liver_lesion_category": "no_focal_lesion", "prior_cancer_history": "no", "confidence": "high", "key_findings": "Cirrhosis without focal lesion"}

Input: "1. Post right hepatectomy for HCC, no recurrence. 2. Cirrhosis."
Output: {"liver_lesion_category": "no_focal_lesion", "prior_cancer_history": "yes", "confidence": "high", "key_findings": "Post hepatectomy, no recurrence"}
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
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
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


def call_model_for_case(case_text: str, limiter: ThreadSafeRateLimiter):
    """Send impression text to the LLM and return parsed classification."""
    user_content = (
        "Analyze the following CT report impression and classify the liver findings:\n\n"
        + case_text
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    retries, wait = 3, 5
    for attempt in range(retries):
        limiter.wait()
        try:
            extra_body = {"enable_thinking": True} if ENABLE_THINKING else {}
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=ENABLE_THINKING,
                temperature=0.3,
                top_p=0.7,
                **({"extra_body": extra_body, "stream_options": {"include_usage": True}}
                   if ENABLE_THINKING else {})
            )

            full_answer, full_reasoning = "", ""
            if ENABLE_THINKING:
                for chunk in completion:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        full_reasoning += delta.reasoning_content
                    if hasattr(delta, 'content') and delta.content:
                        full_answer += delta.content
            else:
                full_answer = completion.choices[0].message.content

            json_data = parse_json_from_content(full_answer)
            if json_data is None:
                return {
                    "liver_lesion_category": "PARSE_ERROR",
                    "prior_cancer_history": "PARSE_ERROR",
                    "confidence": "", "key_findings": "",
                    "_thinking_trace": full_reasoning,
                    "raw_response": full_answer, "_parse_error": True,
                }
            json_data["_thinking_trace"] = full_reasoning
            return json_data

        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(wait)
            wait *= 2

    return {"liver_lesion_category": "TIMEOUT_ERROR",
            "prior_cancer_history": "TIMEOUT_ERROR", "_timeout_error": True}


def safe_append_row(path: str, fieldnames: list, row: dict):
    """Thread-safe CSV append."""
    with file_lock:
        exists = os.path.isfile(path) and os.path.getsize(path) > 0
        with open(path, "a" if exists else "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def process_single_task(row, limiter, fieldnames):
    """Classify one record and write result."""
    case_id  = str(row.get(ID_COLUMN, "")).strip()
    raw_text = str(row.get(TEXT_COLUMN, "")).strip()

    try:
        result = call_model_for_case(raw_text, limiter)
        out_row = {
            ID_COLUMN: case_id,
            "note_id": row.get("note_id", ""),
            "subject_id": row.get("subject_id", ""),
            TEXT_COLUMN: raw_text,
            "liver_lesion_category": result.get("liver_lesion_category", ""),
            "prior_cancer_history": result.get("prior_cancer_history", ""),
            "confidence": result.get("confidence", ""),
            "key_findings": result.get("key_findings", ""),
            "reasoning_content": result.get("_thinking_trace", ""),
            "raw_response": result.get("raw_response", ""),
        }
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        print(f"  {case_id} | {result.get('liver_lesion_category', 'N/A')} | "
              f"prior_cancer={result.get('prior_cancer_history', 'N/A')}")
        return True

    except Exception as e:
        print(f"  FAILED {case_id}: {e}")
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, {
            ID_COLUMN: case_id, "note_id": row.get("note_id", ""),
            "subject_id": row.get("subject_id", ""),
            TEXT_COLUMN: raw_text, "liver_lesion_category": "ERROR",
            "prior_cancer_history": "ERROR", "confidence": "",
            "key_findings": f"ERROR: {type(e).__name__}",
            "reasoning_content": "", "raw_response": str(e),
        })
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Liver Lesion Classification (impression-based)")
    print(f"Model: {MODEL_NAME}  |  Input: {INPUT_CSV_FILE}")
    print("=" * 70)

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"Input file not found: {INPUT_CSV_FILE}")
        return

    try:
        df = pd.read_csv(INPUT_CSV_FILE, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(INPUT_CSV_FILE, encoding="utf-8")
    print(f"Loaded {len(df)} records")

    fieldnames = [
        ID_COLUMN, "note_id", "subject_id", TEXT_COLUMN,
        "liver_lesion_category", "prior_cancer_history",
        "confidence", "key_findings", "reasoning_content", "raw_response",
    ]

    # Checkpoint resume
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV_FILE) and os.path.getsize(OUTPUT_CSV_FILE) > 0:
        try:
            df_done = pd.read_csv(OUTPUT_CSV_FILE, usecols=[ID_COLUMN], encoding="utf-8-sig")
            processed_ids = set(df_done[ID_COLUMN].astype(str).str.strip())
            print(f"Resuming — {len(processed_ids)} already processed")
        except Exception as e:
            print(f"Checkpoint read error: {e}; starting fresh")

    tasks = [
        row for _, row in df.iterrows()
        if str(row.get(ID_COLUMN, "")).strip() not in ("", "nan")
        and str(row.get(TEXT_COLUMN, "")).strip() not in ("", "nan")
        and len(str(row.get(TEXT_COLUMN, "")).strip()) >= 10
        and str(row.get(ID_COLUMN, "")).strip() not in processed_ids
    ]
    print(f"Pending: {len(tasks)} | Workers: {MAX_WORKERS} | RPM: {CALLS_PER_MINUTE}")

    if not tasks:
        print("Nothing to process.")
        return

    limiter = ThreadSafeRateLimiter(CALLS_PER_MINUTE, WINDOW_SECONDS)
    success, fail = 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(process_single_task, r, limiter, fieldnames) for r in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                success += fut.result()
                fail += not fut.result()
            except Exception as e:
                fail += 1
                print(f"  Task exception: {e}")
            if i % 20 == 0 or i == len(futures):
                rate = i / (time.time() - t0) * 60
                print(f"Progress: {i}/{len(futures)} ({i / len(futures) * 100:.1f}%) "
                      f"| ok={success} err={fail} | {rate:.1f}/min")

    elapsed = time.time() - t0
    print(f"\nDone — {len(futures)} total, {success} ok, {fail} err, {elapsed / 60:.1f} min")

    if os.path.exists(OUTPUT_CSV_FILE):
        try:
            df_res = pd.read_csv(OUTPUT_CSV_FILE, encoding="utf-8-sig")
            print("\nCategory distribution:")
            print(df_res['liver_lesion_category'].value_counts().to_string())
            print("\nPrior cancer history:")
            print(df_res['prior_cancer_history'].value_counts().to_string())
        except Exception as e:
            print(f"Stats error: {e}")


if __name__ == "__main__":
    main()
