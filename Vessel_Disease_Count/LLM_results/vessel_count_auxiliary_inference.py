# -*- coding: utf-8 -*-
"""
Diseased Vessel Counting — Auxiliary Model Inference (Baseline Prompt)
Corresponds to Maux in the SCOUT framework (Strategy 1: Model Heterogeneity).
Uses DeepSeek-V3.1 via Aliyun DashScope API with a minimal prompt.
"""

import os
import json
import time
import csv
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

# ========================== Configuration ==========================

API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-XXX")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v3.1"

INPUT_CSV_FILE = "df_angiography_report.csv"
OUTPUT_CSV_FILE = "disease_vessels_DeepSeek_Aliyun_WeakPrompt.csv"

ID_COLUMN = "hadm_id"
TEXT_COLUMN = "angiography_report_raw"

MAX_WORKERS = 50
CALLS_PER_MINUTE = 160
WINDOW_SECONDS = 65

file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ========================== Baseline Prompt ==========================

SYSTEM_PROMPT = """
You are a cardiovascular specialist.
Read the provided coronary angiography/intervention report and determine the number of diseased coronary vessels (0, 1, 2, or 3) based on the following criteria.

Core criteria (must be strictly followed):
1. Severity threshold: left main (LM) stenosis ≥50%; other vessels (LAD, LCX, RCA and their branches) stenosis ≥70%.
2. Left main (LM) rule: if LM has a significant lesion, it counts as two-vessel disease (i.e., both LAD and LCX are considered involved).
3. Counting rule: only count three main vessels (LAD, LCX, RCA). Multiple lesions in the same vessel count as 1. Sub-threshold stenoses must NOT be included.

Output requirements:
Output a JSON object directly. Do NOT include any reasoning, Markdown formatting (e.g., ```json), or extra text.

Output format example:
{"disease_vessel_count": 2}
"""

OUTPUT_FIELDS = ["disease_vessel_count"]


# ========================== Utilities ==========================

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
    """Extract JSON object from model response."""
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


def call_model(case_text: str, limiter: ThreadSafeRateLimiter):
    """Call DashScope API for vessel counting (baseline prompt)."""
    user_content = f"Coronary report:\n{case_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    max_retries, wait_seconds = 3, 5
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
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    full_reasoning += delta.reasoning_content
                if hasattr(delta, 'content') and delta.content:
                    full_answer += delta.content

            json_data = parse_json_from_content(full_answer)
            if json_data is None:
                return {
                    "disease_vessel_count": "PARSE_ERROR",
                    "_thinking_trace": full_reasoning,
                    "raw_response": full_answer,
                    "_parse_error": True
                }
            json_data["_thinking_trace"] = full_reasoning
            return json_data

        except Exception as e:
            print(f"  Warning: {e}")
            time.sleep(wait_seconds)
            wait_seconds *= 2

    return {"disease_vessel_count": "TIMEOUT_ERROR", "_timeout_error": True}


def safe_append_row(path: str, fieldnames: list, row: dict):
    """Thread-safe CSV row append."""
    with file_lock:
        file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
        with open(path, "a" if file_exists else "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def process_single_task(row: pd.Series, limiter: ThreadSafeRateLimiter) -> bool:
    """Process one case: call model and write result."""
    case_id = str(row.get(ID_COLUMN, "")).strip()
    raw_text = str(row.get(TEXT_COLUMN, "")).strip()
    print(f"Processing: {case_id}...")

    try:
        result = call_model(raw_text, limiter)
        thinking_trace = result.get("_thinking_trace", "")
        out_row = {
            ID_COLUMN: case_id,
            TEXT_COLUMN: raw_text,
            "reasoning_content": thinking_trace,
            "disease_vessel_count": result.get("disease_vessel_count", ""),
            "raw_response_content": result.get("raw_response", "")
        }
        fieldnames = [ID_COLUMN, TEXT_COLUMN, "disease_vessel_count", "reasoning_content", "raw_response_content"]
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        print(f"  Done: {case_id} | vessel_count={result.get('disease_vessel_count', 'N/A')}")
        return True

    except Exception as e:
        print(f"  Failed: {case_id}: {e}")
        out_row = {
            ID_COLUMN: case_id, TEXT_COLUMN: raw_text,
            "reasoning_content": f"ERROR: {type(e).__name__}",
            "disease_vessel_count": "ERROR",
            "raw_response_content": str(e)
        }
        fieldnames = [ID_COLUMN, TEXT_COLUMN, "disease_vessel_count", "reasoning_content", "raw_response_content"]
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        return False


# ========================== Main ==========================

def main():
    print("=" * 60)
    print("Diseased Vessel Counting — Auxiliary Model (Baseline Prompt)")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"Input file not found: {INPUT_CSV_FILE}")
        return

    try:
        if INPUT_CSV_FILE.endswith(".xlsx"):
            df = pd.read_excel(INPUT_CSV_FILE)
        else:
            df = pd.read_csv(INPUT_CSV_FILE, encoding="utf-8-sig")
    except Exception as e:
        print(f"Failed to read input: {e}")
        return

    print(f"Total records: {len(df)}")

    # Resume from checkpoint
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV_FILE) and os.path.getsize(OUTPUT_CSV_FILE) > 0:
        try:
            df_done = pd.read_csv(OUTPUT_CSV_FILE, usecols=[ID_COLUMN], encoding="utf-8-sig")
            processed_ids = set(df_done[ID_COLUMN].astype(str).str.strip())
            print(f"[Checkpoint] Already processed: {len(processed_ids)}")
        except Exception as e:
            print(f"Checkpoint read error: {e}, starting fresh.")

    tasks = []
    for _, row in df.iterrows():
        case_id = str(row.get(ID_COLUMN, "")).strip()
        raw_text = str(row.get(TEXT_COLUMN, "")).strip()
        if not case_id or case_id == "nan":
            continue
        if not raw_text or raw_text.lower() == "nan" or len(raw_text) < 10:
            continue
        if case_id in processed_ids:
            continue
        tasks.append(row)

    print(f"Pending tasks: {len(tasks)}, workers: {MAX_WORKERS}")

    limiter = ThreadSafeRateLimiter(max_calls=CALLS_PER_MINUTE, window_seconds=WINDOW_SECONDS)
    success_count, fail_count = 0, 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_task, row, limiter) for row in tasks]
        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            try:
                if future.result():
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                print(f"  Task exception: {e}")
            if i % 10 == 0:
                print(f"Progress: {i}/{total} (success={success_count}, fail={fail_count})")

    print(f"\nCompleted: {total} total, {success_count} success, {fail_count} fail")
    print(f"Output: {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    main()
