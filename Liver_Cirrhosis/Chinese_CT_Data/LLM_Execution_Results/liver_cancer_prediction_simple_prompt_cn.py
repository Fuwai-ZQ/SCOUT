"""
Liver cancer risk prediction from CT findings — baseline prompt (no CoT).
Uses a minimal prompt for 4-class classification.
This serves as the auxiliary model (Maux) with baseline prompt in the SCOUT framework.
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

# ==================== Configuration ====================

API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-XXX")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v3.1"

INPUT_CSV_FILE = "ct_reports_final.csv"
OUTPUT_CSV_FILE = "liver_cancer_prediction_simple.csv"

ID_COLUMN = "patient_id"
TEXT_COLUMN = "ct_findings"

MAX_WORKERS = 40
CALLS_PER_MINUTE = 200
WINDOW_SECONDS = 65

file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==================== Prompt ====================
# NOTE: The prompt is in Chinese because the input CT reports are in Chinese.

SYSTEM_PROMPT = """任务：根据肝硬化患者的CT平扫报告，对肝脏病变进行四分类。

提示：MT是恶性肿瘤的缩写。

分类标准：
1. confirmed_liver_cancer：明确肝癌/肝恶性肿瘤，或伴门静脉癌栓
2. high_risk：可疑低密度病灶、边界不清占位、性质待定结节（非明确囊肿）
3. benign_lesion：明确良性病变（肝囊肿、血管瘤、钙化灶）
4. no_focal_lesion：仅肝硬化改变，无局灶性病变

仅输出JSON：
{"category": "分类结果"}
"""


# ==================== Utilities ====================

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
    """Extract JSON object from LLM response text."""
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


def call_model(ct_findings: str, limiter: ThreadSafeRateLimiter):
    """Call LLM to predict liver cancer risk (baseline prompt)."""
    user_content = f"""CT检查结果：
{ct_findings}

请输出JSON分类结果。"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    MAX_RETRIES = 3
    wait_seconds = 5

    for attempt in range(MAX_RETRIES):
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
                return {"parse_error": True, "raw_response": full_answer,
                        "thinking_trace": full_reasoning, "category": "PARSE_ERROR"}
            json_data["thinking_trace"] = full_reasoning
            json_data["raw_response"] = full_answer
            return json_data

        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(wait_seconds)
            wait_seconds *= 2

    return {"timeout_error": True, "category": "TIMEOUT_ERROR"}


def safe_append_row(path: str, fieldnames: list, row: dict):
    """Thread-safe CSV row append."""
    with file_lock:
        file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
        with open(path, "a" if file_exists else "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def process_single_task(row: pd.Series, limiter: ThreadSafeRateLimiter, fieldnames: list) -> bool:
    """Process one patient record."""
    case_id = str(row.get(ID_COLUMN, "")).strip()
    ct_findings = str(row.get(TEXT_COLUMN, "")).strip()

    try:
        result = call_model(ct_findings, limiter)
        out_row = {
            ID_COLUMN: case_id,
            "exam_date": row.get("exam_date", ""),
            "center": row.get("center", ""),
            "predicted_category": result.get("category", "N/A"),
            "model_thinking": (result.get("thinking_trace") or "")[:5000],
        }
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        print(f"  {case_id} | pred: {result.get('category')}")
        return True

    except Exception as e:
        print(f"  FAILED {case_id}: {e}")
        out_row = {
            ID_COLUMN: case_id,
            "exam_date": row.get("exam_date", ""),
            "center": row.get("center", ""),
            "predicted_category": "ERROR",
            "model_thinking": f"ERROR: {type(e).__name__} - {e}",
        }
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        return False


# ==================== Main ====================

def main():
    print(f"Model: {MODEL_NAME} | Input: {INPUT_CSV_FILE} | Output: {OUTPUT_CSV_FILE}")

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"File not found: {INPUT_CSV_FILE}")
        return

    try:
        df = pd.read_csv(INPUT_CSV_FILE, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(INPUT_CSV_FILE, encoding="utf-8")

    print(f"Total records: {len(df)}")

    fieldnames = [ID_COLUMN, "exam_date", "center", "predicted_category", "model_thinking"]

    # Resume from checkpoint
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV_FILE) and os.path.getsize(OUTPUT_CSV_FILE) > 0:
        try:
            df_done = pd.read_csv(OUTPUT_CSV_FILE, usecols=[ID_COLUMN], encoding="utf-8-sig")
            processed_ids = set(df_done[ID_COLUMN].astype(str).str.strip())
            print(f"[Resume] Already processed: {len(processed_ids)}")
        except Exception as e:
            print(f"Warning reading checkpoint: {e}")

    tasks = []
    for _, row in df.iterrows():
        case_id = str(row.get(ID_COLUMN, "")).strip()
        ct_findings = str(row.get(TEXT_COLUMN, "")).strip()
        if not case_id or case_id == "nan":
            continue
        if not ct_findings or ct_findings.lower() == "nan" or len(ct_findings) < 20:
            continue
        if case_id in processed_ids:
            continue
        tasks.append(row)

    print(f"Pending: {len(tasks)} | Workers: {MAX_WORKERS} | Rate limit: {CALLS_PER_MINUTE}/min")

    if not tasks:
        print("Nothing to process.")
        return

    limiter = ThreadSafeRateLimiter(CALLS_PER_MINUTE, WINDOW_SECONDS)
    success_count, fail_count = 0, 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_task, row, limiter, fieldnames) for row in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                if future.result():
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                print(f"  Exception: {e}")
            if i % 10 == 0 or i == len(futures):
                elapsed = time.time() - start_time
                rate = i / elapsed * 60 if elapsed > 0 else 0
                print(f"Progress: {i}/{len(futures)} | OK: {success_count} | "
                      f"Fail: {fail_count} | Rate: {rate:.1f}/min")

    elapsed_total = time.time() - start_time
    print(f"\nDone. Total: {len(tasks)}, OK: {success_count}, Fail: {fail_count}, "
          f"Time: {elapsed_total / 60:.1f} min")

    if os.path.exists(OUTPUT_CSV_FILE):
        try:
            df_result = pd.read_csv(OUTPUT_CSV_FILE, encoding="utf-8-sig")
            print("\nPrediction distribution:")
            print(df_result['predicted_category'].value_counts().to_string())
        except Exception as e:
            print(f"Stats error: {e}")


if __name__ == "__main__":
    main()
