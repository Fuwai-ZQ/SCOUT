# -*- coding: utf-8 -*-
###############################################################################
# 07_auxiliary_model_inference.py
# Strategy 1 (S1): Model heterogeneity — Maux with baseline prompt
# Maux = DeepSeek-V3.1 (same architecture, different prompt design)
# Reference: "Prompt heterogeneity was achieved by contrasting a baseline prompt
#   with an optimized prompt augmented by guideline-derived clinical knowledge"
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
MODEL_NAME = "deepseek-v3.1"  # Maux (same model, baseline prompt)

INPUT_CSV = "LLM_Input_with_HADM_ID.csv"
OUTPUT_CSV = "auxiliary_model_predictions.csv"

MAX_WORKERS = 5
CALLS_PER_MINUTE = 60
WINDOW_SECONDS = 65

file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# =============================================================================
# System prompt — Baseline (task definition + format requirements only)
# =============================================================================

SYSTEM_PROMPT = """
You are a cardiologist.
Please read the given summary of a patient with coronary heart disease and judge the classification of coronary heart disease for this patient based on your medical knowledge.

Please select the most appropriate one from the following 5 labels (only one can be selected):
- STEMI
- NSTEMI
- UA
- CCS
- Insufficient Information

**Output Format Requirements**:
1. Please engage in deep thinking first, analyzing key evidence in the case (such as ECG, cardiac enzymes, medical history characteristics).
2. Finally, output only a JSON object, formatted as follows:
{
  "final_label": "Your diagnosis result"
}
"""


# =============================================================================
# Utilities (shared pattern with 06_primary_model_inference.py)
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
    """Single inference call to Maux (baseline prompt) with reasoning trace."""
    user_content = "Case Summary:\n" + str(case_text)
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
                return {
                    "final_label": "PARSE_ERROR",
                    "_thinking_trace": full_reasoning,
                    "raw_response": full_answer,
                }
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
    """Process one case: call Maux → write result."""
    hadm_id = str(row.get("hadm_id", "")).strip()
    case_text = row.get("final_text", "")
    print(f"  Processing: {hadm_id}")

    try:
        result = call_model(case_text, limiter)
        out = {
            "hadm_id": hadm_id,
            "final_label": result.get("final_label", "UNKNOWN"),
            "reasoning_content": result.get("_thinking_trace", ""),
            "raw_response_content": result.get("raw_response", ""),
        }
        fieldnames = ["hadm_id", "final_label", "reasoning_content", "raw_response_content"]
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
