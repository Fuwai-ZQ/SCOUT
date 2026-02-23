# -*- coding: utf-8 -*-
###############################################################################
# 06_primary_model_inference.py
# Mmain: DeepSeek-V3.1 with chain-of-thought (CoT) optimized prompt
# Also used for Strategy 2 (stochastic inconsistency) via independent re-run
# Reference: Methods - "Configuration selection and model implementation"
###############################################################################

import os
import json
import time
import csv
import re
import threading
import pandas as pd
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =============================================================================
# Configuration
# =============================================================================

API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-XXX")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v3.1"  # Mmain

INPUT_CSV = "LLM_Input_with_HADM_ID.csv"

# --- Change output filename per run ---
# Run 1 (Mmain prediction):  "primary_model_predictions.csv"
# Run 2 (S2 re-sampling):    "s2_stochastic_rerun.csv"
OUTPUT_CSV = "primary_model_predictions.csv"

MAX_WORKERS = 10
CALLS_PER_MINUTE = 60
WINDOW_SECONDS = 65

file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# =============================================================================
# System prompt — CoT-optimized with guideline-derived clinical knowledge
# =============================================================================

SYSTEM_PROMPT = """
You are a clinical adjudicator who classifies coronary heart disease strictly according to given rules.
You must judge only based on this prompt and the given rules, and must not call upon external knowledge; if rules conflict with common sense, the rules prevail.

I. Allowed Final Labels (Select only one)

* STEMI
* NSTEMI
* UA
* CCS
* Insufficient Information

II. Specific Rules for Subtypes

A. STEMI
<!-- [REDACTED: Detailed STEMI diagnostic criteria. Removed for copyright protection.] -->

B. NSTEMI (When STEMI conditions are not met)
<!-- [REDACTED: Detailed NSTEMI diagnostic criteria. Removed for copyright protection.] -->

C. UA (When not meeting STEMI, NSTEMI)
<!-- [REDACTED: Detailed UA diagnostic criteria. Removed for copyright protection.] -->

D. CCS (When not meeting STEMI, NSTEMI, UA)
<!-- [REDACTED: Detailed CCS diagnostic criteria. Removed for copyright protection.] -->

III. Evidence and Synonyms
<!-- [REDACTED: Synonym mappings and conversion rules. Removed for copyright protection.] -->

IV. Missing Value Imputation and Default Inference
<!-- [REDACTED: Rules for handling missing data. Removed for copyright protection.] -->

V. Overall Decision Flow and Priorities

1. Classification needs to be judged top-down in this priority: STEMI → NSTEMI → UA → CCS → Insufficient Information.
<!-- [REDACTED: Detailed priority cascade rules. Removed for copyright protection.] -->

VI. Output Format (Must be single paragraph JSON)
You must output only one JSON string, no extra text, format as follows:

{
  "evidence": {
    "admission_date": "<YYYY-MM-DD or null>",
    "symptoms": "<Key symptoms + Time>",
    "ecg": "<ST elevation Y/N; Leads; Time>",
    "troponin": "<Elevated Y/N; Value; Time; Non-ischemic cause noted Y/N>",
    "revascularization": "<Thrombolysis/PCI/CABG and Time; Post-op symptoms recurrence Y/N>",
    "timing_notes": "<Admission Day(A) - Event Day(B) = X days>",
    "inference": "<Whether 'most likely state' inference used and basis>",
    "missing_or_conflicts": "<Key information missing or conflicting points>"
  },
  "rationale_short": "No more than 80 words, briefly summarize time window and key evidence.",
  "final_label": "<STEMI|NSTEMI|UA|CCS|Insufficient Information>"
}
"""


# =============================================================================
# Utilities
# =============================================================================

class ThreadSafeRateLimiter:
    """Sliding-window rate limiter (thread-safe)."""

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
    """Extract JSON object from model output, handling markdown fences."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text).strip()
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


def call_model(case_text: str, limiter: ThreadSafeRateLimiter) -> dict:
    """Single inference call to Mmain with streaming + reasoning trace."""
    user_content = (
        "Below is the clinical text for a patient. Please classify the coronary "
        "heart disease strictly according to the system prompt rules and output "
        f"the required JSON:\n\n{case_text}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    max_retries, wait_s = 3, 5
    for _ in range(max_retries):
        limiter.wait()
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                extra_body={"enable_thinking": True},
                stream=True,
                stream_options={"include_usage": True},
                temperature=0.6,
                top_p=0.95,
            )
            full_answer, full_reasoning = "", ""
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    full_reasoning += delta.reasoning_content
                if hasattr(delta, "content") and delta.content:
                    full_answer += delta.content

            parsed = parse_json_from_content(full_answer)
            if parsed is None:
                return {"final_label": "PARSE_ERROR", "_thinking_trace": full_reasoning}
            parsed["_thinking_trace"] = full_reasoning
            return parsed

        except Exception as e:
            print(f"  [Retry] {e}")
            time.sleep(wait_s)
            wait_s *= 2

    return {"final_label": "TIMEOUT_ERROR"}


def safe_append_row(path: str, fieldnames: list, row: dict):
    """Thread-safe single-row CSV append with checkpoint support."""
    with file_lock:
        exists = os.path.isfile(path) and os.path.getsize(path) > 0
        with open(path, "a" if exists else "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def process_single_case(row, limiter: ThreadSafeRateLimiter) -> bool:
    """Process one case: call model → write result."""
    hadm_id = str(row.get("hadm_id", "")).strip()
    case_text = row.get("final_text", "")
    print(f"  Processing: {hadm_id}")

    try:
        result = call_model(case_text, limiter)
        out = {
            "hadm_id": hadm_id,
            "final_label": result.get("final_label", "UNKNOWN"),
            "rationale_short": result.get("rationale_short", ""),
            "evidence_json": json.dumps(result.get("evidence", {}), ensure_ascii=False),
            "reasoning_content": result.get("_thinking_trace", ""),
        }
        fieldnames = ["hadm_id", "final_label", "rationale_short", "evidence_json", "reasoning_content"]
        safe_append_row(OUTPUT_CSV, fieldnames, out)
        print(f"  Done: {hadm_id} -> {out['final_label']}")
        return True
    except Exception as e:
        print(f"  Error [{hadm_id}]: {e}")
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Input file not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    # Resume from checkpoint
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        try:
            done = pd.read_csv(OUTPUT_CSV, usecols=["hadm_id"], encoding="utf-8-sig")
            processed_ids = set(done["hadm_id"].astype(str).str.strip())
            print(f"[Resume] {len(processed_ids)} already processed.")
        except Exception:
            pass

    tasks = [
        row for _, row in df.iterrows()
        if str(row.get("hadm_id", "")).strip() not in processed_ids
        and str(row.get("hadm_id", "")).strip() not in ("", "nan")
        and len(str(row.get("final_text", ""))) >= 10
    ]
    print(f"Starting {len(tasks)} tasks with {MAX_WORKERS} threads...")

    limiter = ThreadSafeRateLimiter(CALLS_PER_MINUTE, WINDOW_SECONDS)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_case, row, limiter) for row in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(futures)}")

    print(f"\nComplete. Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
