# -*- coding: utf-8 -*-
"""Chain-of-thought audit using OpenAI GPT API.
Checks rationale–evidence consistency."""

import os
import json
import time
import csv
import pandas as pd
import re
import threading
import signal
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# --- Configuration ---

# OpenAI API config
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")  
BASE_URL = "https://api.openai.com/v1"  

MODEL_NAME = "gpt-5.1-2025-11-13"

# Whether to use o1/o3 reasoning models
USE_REASONING_MODEL = False  

# Input: CSV from classification step (with rationale + evidence)
INPUT_CSV_FILE = "cases_labeled_Qwen235b_withPrompt_withThinking.csv"
# Output: audit results
OUTPUT_CHECK_FILE = "cases_checked_results_ChatGPT.csv"

MAX_WORKERS = 1  
CALLS_PER_MINUTE = 60  
WINDOW_SECONDS = 65
SAVE_EVERY_N = 5  

file_lock = threading.Lock()
shutdown_flag = threading.Event()
processed_count = 0
count_lock = threading.Lock()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

AUDIT_SYSTEM_PROMPT = """
你是一名医疗数据逻辑审计员。你的任务是检查"简述理由（Rationale）"与"证据详情（Evidence JSON）"之间是否存在事实矛盾或逻辑错误。

请仔细分析后，输出一个 JSON 对象。

检查重点：
1. **时间计算矛盾**：例如 Rationale 说"病程大于30天"，但 Evidence 中显示"入院日-事件日 = 5天"。
2. **数据一致性**：例如 Rationale 说"Tn升高"，但 Evidence 中 Tn 为"正常"或"未升高"。
3. **诊断逻辑冲突**：例如 Rationale 说"无缺血症状"，但 Evidence 中记录了"胸痛"。

输出格式（仅输出JSON，不要有其他内容）：
{
  "has_contradiction": "present" 或 "不存在",
  "analysis": "简述发现的矛盾点，如果没有矛盾则留空或写'一致'"
}
"""

# Prompt for o1/o3 reasoning models
AUDIT_USER_PROMPT_FOR_REASONING = """
你是一名医疗数据逻辑审计员。你的任务是检查"简述理由（Rationale）"与"证据详情（Evidence JSON）"之间是否存在事实矛盾或逻辑错误。

检查重点：
1. **时间计算矛盾**：例如 Rationale 说"病程大于30天"，但 Evidence 中显示"入院日-事件日 = 5天"。
2. **数据一致性**：例如 Rationale 说"Tn升高"，但 Evidence 中 Tn 为"正常"或"未升高"。
3. **诊断逻辑冲突**：例如 Rationale 说"无缺血症状"，但 Evidence 中记录了"胸痛"。

请检查以下两条信息的一致性：

【Rationale (简述)】
{rationale}

【Evidence (证据详情)】
{evidence}

请仔细分析后，输出一个 JSON 对象（仅输出JSON，不要有其他内容）：
{{
  "has_contradiction": "present" 或 "不存在",
  "analysis": "简述发现的矛盾点，如果没有矛盾则留空或写'一致'"
}}
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

def call_audit_model_standard(rationale: str, evidence_json: str, limiter: ThreadSafeRateLimiter):
    """Call standard ChatGPT model for audit."""

    user_content = f"""
请检查以下两条信息的一致性：

【Rationale (简述)】
{rationale}

【Evidence (证据详情)】
{evidence_json}
"""

    messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
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
                messages=messages
            )

            full_answer = completion.choices[0].message.content or ""
            json_data = parse_json_from_content(full_answer)

            if json_data is None:
                return {
                    "has_contradiction": "parse_error",
                    "analysis": f"JSON parse failed, raw: {full_answer[:200]}",
                    "_thinking_trace": ""
                }

            json_data["_thinking_trace"] = ""  
            return json_data

        except InterruptedError:
            raise
        except Exception as e:
            error_msg = str(e)
            print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {error_msg}")

            if "rate_limit" in error_msg.lower():
                print(f"     Rate limited, waiting {wait_seconds * 2}s...")
                time.sleep(wait_seconds * 2)
            elif "context_length" in error_msg.lower():
                return {
                    "has_contradiction": "context_overflow",
                    "analysis": "Input exceeds context length limit",
                    "_thinking_trace": ""
                }
            else:
                time.sleep(wait_seconds)

            wait_seconds *= 2
            continue

    return {
        "has_contradiction": "timeout",
        "analysis": "Max retries exceeded",
        "_thinking_trace": ""
    }

def call_audit_model_reasoning(rationale: str, evidence_json: str, limiter: ThreadSafeRateLimiter):
    """Call o1/o3 reasoning model."""

    user_content = AUDIT_USER_PROMPT_FOR_REASONING.format(
        rationale=rationale,
        evidence=evidence_json
    )

    messages = [
        {"role": "user", "content": user_content},
    ]

    MAX_RETRIES = 5
    wait_seconds = 10

    for attempt in range(MAX_RETRIES):
        if shutdown_flag.is_set():
            raise InterruptedError("Shutdown signal received")

        limiter.wait()
        try:
            
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                
                max_completion_tokens=4096,
            )

            full_answer = completion.choices[0].message.content or ""

            reasoning_content = ""
            if hasattr(completion.choices[0].message, 'reasoning_content'):
                reasoning_content = completion.choices[0].message.reasoning_content or ""

            json_data = parse_json_from_content(full_answer)

            if json_data is None:
                return {
                    "has_contradiction": "parse_error",
                    "analysis": f"JSON parse failed, raw: {full_answer[:200]}",
                    "_thinking_trace": reasoning_content
                }

            json_data["_thinking_trace"] = reasoning_content
            return json_data

        except InterruptedError:
            raise
        except Exception as e:
            error_msg = str(e)
            print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {error_msg}")

            if "rate_limit" in error_msg.lower():
                print(f"     Rate limited, waiting {wait_seconds * 2}s...")
                time.sleep(wait_seconds * 2)
            else:
                time.sleep(wait_seconds)

            wait_seconds *= 2
            continue

    return {
        "has_contradiction": "timeout",
        "analysis": "Max retries exceeded",
        "_thinking_trace": ""
    }

def call_audit_model(rationale: str, evidence_json: str, limiter: ThreadSafeRateLimiter):
    """Unified API call dispatcher."""
    if USE_REASONING_MODEL:
        return call_audit_model_reasoning(rationale, evidence_json, limiter)
    else:
        return call_audit_model_standard(rationale, evidence_json, limiter)

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
    """Process a single audit task."""
    global processed_count

    if shutdown_flag.is_set():
        return False

    case_id = str(row.get("case_id", "")).strip()
    rationale = str(row.get("rationale_short", ""))
    evidence = str(row.get("evidence_json", ""))
    final_label = row.get("final_label", "")

    if len(rationale) < 5 or len(evidence) < 5:
        print(f"  SKIP {case_id}: rationale/evidence too short")
        return True

    try:
        print(f"Checking: {case_id} ...")

        result = call_audit_model(rationale, evidence, limiter)

        out_row = {
            "case_id": case_id,
            "final_label": final_label,
            "rationale_short": rationale,
            "audit_result": result.get("has_contradiction", "unknown"),
            "audit_analysis": result.get("analysis", ""),
            "audit_thinking": result.get("_thinking_trace", "")[:10000]  
        }

        safe_append_row(OUTPUT_CHECK_FILE, fieldnames, out_row)

        with count_lock:
            processed_count += 1

        if out_row['audit_thinking']:
            think_preview = out_row['audit_thinking'][:60].replace('\n', ' ')
            print(f"   Thinking: {think_preview}...")
        print(f"   Result: {out_row['audit_result']} - {out_row['audit_analysis']}")

        return True

    except InterruptedError:
        print(f"  INTERRUPTED: {case_id}")
        return False
    except Exception as e:
        print(f"  FAILED {case_id}: {e}")

        out_row = {
            "case_id": case_id,
            "final_label": final_label,
            "rationale_short": rationale,
            "audit_result": "ERROR",
            "audit_analysis": f"ERROR: {type(e).__name__} - {str(e)}",
            "audit_thinking": ""
        }

        safe_append_row(OUTPUT_CHECK_FILE, fieldnames, out_row)
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
    print("CoT Audit - GPT API")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"API: {BASE_URL}")
    print(f"Input: {INPUT_CSV_FILE}")
    print(f"Output: {OUTPUT_CHECK_FILE}")
    print(f"Reasoning: {'o1/o3' if USE_REASONING_MODEL else 'standard'}")
    print("=" * 70)
    print("Press Ctrl+C to interrupt safely")
    print("=" * 70)

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"ERROR: Input file not found: {INPUT_CSV_FILE}")
        return

    df = None
    encodings_to_try = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'iso-8859-1', 'latin-1']
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(INPUT_CSV_FILE, encoding=enc)
            print(f"Encoding: {enc}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if df is None:
        print(f"ERROR: Cannot read input: {INPUT_CSV_FILE}")
        return

    print(f"Loaded {len(df)} records for audit...")
    print(f"Columns: {df.columns.tolist()}")

    fieldnames = ["case_id", "final_label", "rationale_short", "audit_result", "audit_analysis", "audit_thinking"]

    # Checkpoint resume
    processed_ids = set()
    if os.path.exists(OUTPUT_CHECK_FILE) and os.path.getsize(OUTPUT_CHECK_FILE) > 0:
        df_done = None
        for enc in encodings_to_try:
            try:
                df_done = pd.read_csv(OUTPUT_CHECK_FILE, usecols=["case_id"], encoding=enc)
                print(f"[Resume] Encoding: {enc}")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                print(f"WARN: Checkpoint read error: {e}")
                break

        if df_done is not None:
            processed_ids = set(df_done["case_id"].astype(str).str.strip())
            print(f"[Resume] Already done: {len(processed_ids)}")
        else:
            print(f"WARN: Cannot read checkpoint, starting fresh")

    tasks = []
    skip_invalid_id = 0
    skip_already_done = 0

    for idx, row in df.iterrows():
        case_id = str(row.get("case_id", "")).strip()

        if not case_id or case_id == "nan":
            skip_invalid_id += 1
            continue
        if case_id in processed_ids:
            skip_already_done += 1
            continue

        tasks.append(row)

    print(f"\nData filtering summary:")
    print(f"   Total records:      {len(df)}")
    print(f"   Invalid ID:         {skip_invalid_id}")
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
        print("Audit complete!")
    print(f"   Total:     {len(tasks)}")
    print(f"   Success:   {success_count}")
    print(f"   Failed:    {fail_count}")
    print(f"   Elapsed:   {elapsed_total / 60:.1f} min")
    print(f"   Output:    {OUTPUT_CHECK_FILE}")
    print("=" * 70)

    if os.path.exists(OUTPUT_CHECK_FILE):
        try:
            df_result = None
            for enc in encodings_to_try:
                try:
                    df_result = pd.read_csv(OUTPUT_CHECK_FILE, encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue

            if df_result is not None:
                print("\nAudit result distribution:")
                print("-" * 40)
                print(df_result['audit_result'].value_counts().to_string())
        except Exception as e:
            print(f"Stats error: {e}")

if __name__ == "__main__":
    main()