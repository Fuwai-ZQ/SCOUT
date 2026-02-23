# -*- coding: utf-8 -*-
"""CAD subtype classification via DashScope API (minimal prompt).
Supports concurrent execution, checkpoint resume, and incremental saving."""

import os
import json
import time
import csv
import pandas as pd
import re
import threading
import signal
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# --- Configuration ---

# DashScope API config
API_KEY = os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v3.1"  # Options: deepseek-v3.2,deepseek-v3.1, qwen3-235b-a22b, qwen3-32b, qwen3-8b

INPUT_XLSX_FILE = "cases.xlsx"
OUTPUT_CSV_FILE = "cases_labeled_DS3.1_noPrompt_noThinking.csv"

# Column names in input xlsx (Chinese headers)
ID_COLUMN = "case_id"
HISTORY_COLUMN = "case_features"

MAX_WORKERS = 10  
CALLS_PER_MINUTE = 200  
WINDOW_SECONDS = 65
SAVE_EVERY_N = 5  

file_lock = threading.Lock()
shutdown_flag = threading.Event()
processed_count = 0
count_lock = threading.Lock()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- System Prompt (Minimal) ---

SYSTEM_PROMPT = """

你是一名心血管内科医生。
请阅读给定的冠心病患者病历摘要，根据你的医学知识判断该患者的冠心病分型。

请在以下 5 个标签中选择一个最合适的（只能选其一）：
- STEMI
- NSTEMI
- UA
- CCS
- 信息不足
**输出格式要求**：
1. 请先进行深度思考，分析病例中的关键证据（如心电图、心肌酶、病史特点）。
2. 最后仅输出一个 JSON 对象，格式如下：
{
  "final_label": "你的诊断结果"
}
"""

# --- Utilities ---

class ThreadSafeRateLimiter:
    """Thread-safe sliding-window rate limiter."""

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
    """Extract JSON from LLM response text."""
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
        
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except:
                pass
    return None

def build_case_text(history: str) -> str:
    """Return the case text as-is."""
    return history

def call_model_for_classification(case_text: str, limiter: ThreadSafeRateLimiter):
    """Call API for CAD subtype classification."""

    user_content = (
        "下面是一位冠心病患者的病历关键信息，请严格按照系统提示中的规则进行分型并按要求输出JSON：\n\n"
        + case_text
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    MAX_RETRIES = 5
    wait_seconds = 5

    for attempt in range(MAX_RETRIES):
        if shutdown_flag.is_set():
            raise InterruptedError("Shutdown signal received")
            
        limiter.wait()
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                extra_body={"enable_thinking": False},  
                stream=True,
                stream_options={"include_usage": True},
                temperature=0.1,
                top_p=0.95,
                max_tokens=8192,
            )

            full_answer = ""
            full_reasoning = ""

            for chunk in completion:
                if shutdown_flag.is_set():
                    raise InterruptedError("Shutdown signal received")
                    
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
                    "parse_error": True,
                    "raw_response": full_answer,
                    "thinking_trace": full_reasoning,
                    "final_label": "PARSE_ERROR",
                    "rationale_short": "JSON parse failed",
                    "evidence": {}
                }

            json_data["thinking_trace"] = full_reasoning
            json_data["raw_response"] = full_answer
            return json_data

        except InterruptedError:
            raise
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(wait_seconds)
            wait_seconds *= 2
            continue

    return {
        "timeout_error": True,
        "final_label": "TIMEOUT_ERROR",
        "rationale_short": "API call timeout",
        "evidence": {},
        "thinking_trace": ""
    }

def safe_append_row(path: str, fieldnames: list, row: dict):
    """Thread-safe CSV row append."""
    with file_lock:
        file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
        with open(path, "a" if file_exists else "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()

def process_single_task(row: pd.Series, limiter: ThreadSafeRateLimiter, fieldnames: list) -> bool:
    """Process a single classification task."""
    global processed_count
    
    if shutdown_flag.is_set():
        return False
        
    case_id = str(row.get(ID_COLUMN, "")).strip()
    history = str(row.get(HISTORY_COLUMN, "")).strip()

    try:
        
        case_text = build_case_text(history)
        
        result = call_model_for_classification(case_text, limiter)

        final_label = result.get("final_label", "")
        rationale_short = result.get("rationale_short", "")
        evidence = result.get("evidence", {})
        thinking_trace = result.get("thinking_trace", "")

        out_row = {
            ID_COLUMN: case_id,
            HISTORY_COLUMN: history,
            "final_label": final_label,
            "rationale_short": rationale_short,
            "reasoning_content": thinking_trace[:10000] if thinking_trace else "",
            "evidence_json": json.dumps(evidence, ensure_ascii=False) if evidence else "",
        }

        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)

        with count_lock:
            processed_count += 1
            
        print(f"  OK {case_id} | Label: {final_label} | Thinking: {len(thinking_trace)} chars")
        return True

    except InterruptedError:
        print(f"  INTERRUPTED: {case_id}")
        return False
    except Exception as e:
        print(f"  FAILED {case_id}: {e}")

        out_row = {
            ID_COLUMN: case_id,
            HISTORY_COLUMN: history,
            "final_label": "ERROR",
            "rationale_short": f"ERROR: {type(e).__name__} - {str(e)}",
            "reasoning_content": "",
            "evidence_json": "",
        }

        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        return False

def signal_handler(signum, frame):
    """Handle interrupt signal for graceful shutdown."""
    print("\n\nShutdown signal received, exiting safely...")
    print("   Waiting for current tasks to finish...")
    shutdown_flag.set()

# --- Main ---

def main():
    global processed_count
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 70)
    print("CAD Classification - Minimal Prompt")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Input: {INPUT_XLSX_FILE}")
    print(f"Output: {OUTPUT_CSV_FILE}")
    print(f"Thinking: enabled")
    print("=" * 70)
    print("Press Ctrl+C to interrupt safely")
    print("=" * 70)

    if not os.path.exists(INPUT_XLSX_FILE):
        print(f"ERROR: Input file not found: {INPUT_XLSX_FILE}")
        return

    try:
        df = pd.read_excel(INPUT_XLSX_FILE)
        print(f"Loaded xlsx file")
    except Exception as e:
        print(f"ERROR: Failed to read xlsx: {e}")
        return

    print(f"Total records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    fieldnames = [
        ID_COLUMN, HISTORY_COLUMN,
        "final_label",
        "rationale_short",
        "reasoning_content",
        "evidence_json"
    ]

    # Checkpoint resume
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV_FILE) and os.path.getsize(OUTPUT_CSV_FILE) > 0:
        encodings_to_try = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'iso-8859-1', 'latin-1']
        df_done = None

        for enc in encodings_to_try:
            try:
                df_done = pd.read_csv(OUTPUT_CSV_FILE, usecols=[ID_COLUMN], encoding=enc)
                print(f"[Resume] Encoding: {enc}")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                print(f"WARN: Checkpoint read error: {e}")
                break

        if df_done is not None:
            processed_ids = set(df_done[ID_COLUMN].astype(str).str.strip())
            print(f"[Resume] Already done: {len(processed_ids)}")
        else:
            print(f"WARN: Cannot read checkpoint, starting fresh")

    tasks = []
    skip_invalid_id = 0
    skip_invalid_text = 0
    skip_already_done = 0

    for idx, row in df.iterrows():
        case_id = str(row.get(ID_COLUMN, "")).strip()
        history = str(row.get(HISTORY_COLUMN, "")).strip()

        if not case_id or case_id == "nan":
            skip_invalid_id += 1
            continue
        if not history or history.lower() == "nan":
            skip_invalid_text += 1
            continue
        if case_id in processed_ids:
            skip_already_done += 1
            continue

        tasks.append(row)

    print(f"\nData filtering summary:")
    print(f"   Total records:      {len(df)}")
    print(f"   Invalid ID:         {skip_invalid_id}")
    print(f"   Invalid text:       {skip_invalid_text}")
    print(f"   Already processed:  {skip_already_done}")
    print(f"   Pending:            {len(tasks)}")
    print(f"\nPending: {len(tasks)} tasks")
    print(f"   Workers: {MAX_WORKERS} | Rate limit: {CALLS_PER_MINUTE}/min")
    print("-" * 70)

    if not tasks:
        print("No pending tasks")
        return

    limiter = ThreadSafeRateLimiter(CALLS_PER_MINUTE, WINDOW_SECONDS)
    success_count = 0
    fail_count = 0
    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_single_task, row, limiter, fieldnames): row for row in tasks}

            for i, future in enumerate(as_completed(futures), 1):
                if shutdown_flag.is_set():
                    print("\nCancelling remaining tasks...")
                    for f in futures:
                        f.cancel()
                    break
                    
                try:
                    if future.result():
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    print(f"  ERROR: {e}")

                if i % 5 == 0 or i == len(futures):
                    elapsed = time.time() - start_time
                    rate = i / elapsed * 60 if elapsed > 0 else 0
                    print(f"Progress: {i}/{len(futures)} | OK: {success_count} | "
                          f"Fail: {fail_count} | Rate: {rate:.1f}/min")
                          
    except KeyboardInterrupt:
        print("\nInterrupted, saving progress...")

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 70)
    if shutdown_flag.is_set():
        print("Interrupted, progress saved!")
    else:
        print("Done!")
    print(f"   Total:     {len(tasks)}")
    print(f"   Success:   {success_count}")
    print(f"   Failed:    {fail_count}")
    print(f"   Elapsed:   {elapsed_total / 60:.1f} min")
    print(f"   Output:    {OUTPUT_CSV_FILE}")
    print("=" * 70)

    if os.path.exists(OUTPUT_CSV_FILE):
        try:
            df_result = None
            for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'iso-8859-1', 'latin-1']:
                try:
                    df_result = pd.read_csv(OUTPUT_CSV_FILE, encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue

            if df_result is not None:
                print("\nClassification distribution:")
                print("-" * 40)
                print(df_result['final_label'].value_counts().to_string())
        except Exception as e:
            print(f"Stats error: {e}")

if __name__ == "__main__":
    main()
