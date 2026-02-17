"""
Liver cancer risk prediction — baseline prompt (Strategy S1 auxiliary).

Uses a minimal prompt without clinical guidelines or chain-of-thought
requirements to classify CT findings into four risk categories.
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
OUTPUT_CSV_FILE = "path/to/liver_cancer_prediction_baseline.csv"

ID_COLUMN        = "hadm_id"
MAX_WORKERS      = 6
CALLS_PER_MINUTE = 99
WINDOW_SECONDS   = 65

file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# =============================================================================
# Prompt
# =============================================================================

SYSTEM_PROMPT = """
Task: Classify the liver findings from the provided non-contrast (plain) CT report of a cirrhotic patient into one of the following categories.

Categories:
1. confirmed_liver_cancer: Explicit malignancy or mass with portal vein tumor thrombus (PVTT).
2. high_risk: Any hypodense lesion, ill-defined margin, or suspicious mass not confirmed as a cyst.
3. benign_lesion: Confirmed benign findings (e.g., simple cyst, siderotic nodules).
4. no_focal_lesion: Cirrhotic changes only; no focal mass or lesion described.

Output Format (JSON only):
{
  "category": "confirmed_liver_cancer / high_risk / benign_lesion / no_focal_lesion"
}
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
    """Send report sections to the LLM and return parsed prediction."""
    user_content = (
        f"Report Data:\nIndication: {indication}\n"
        f"Comparison: {comparison}\nFindings: {findings}\n\n"
        "Please output the JSON classification based on the findings above."
    )
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
                return {"category": "PARSE_ERROR", "thinking_trace": full_reasoning,
                        "raw_response": full_answer, "parse_error": True}
            data["thinking_trace"] = full_reasoning
            data["raw_response"] = full_answer
            return data

        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(wait); wait *= 2

    return {"category": "TIMEOUT_ERROR", "timeout_error": True}


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
    """Classify one record and write result."""
    case_id    = str(row.get(ID_COLUMN, "")).strip()
    indication = str(row.get("indication", "")).strip()
    comparison = str(row.get("comparison", "")).strip()
    findings   = str(row.get("findings", "")).strip()

    try:
        result   = call_model(indication, comparison, findings, limiter)
        category = result.get("category", "N/A")
        out_row  = {
            ID_COLUMN: case_id,
            "note_id": row.get("note_id", ""),
            "subject_id": row.get("subject_id", ""),
            "predicted_category": category,
            "reasoning_detail": f"Classified as: {category}",
            "model_thinking": (result.get("thinking_trace", "") or "")[:5000],
        }
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        print(f"  {case_id} | {category}")
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
    print("Liver Cancer Risk Prediction — Baseline Prompt")
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

    # Checkpoint resume
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
