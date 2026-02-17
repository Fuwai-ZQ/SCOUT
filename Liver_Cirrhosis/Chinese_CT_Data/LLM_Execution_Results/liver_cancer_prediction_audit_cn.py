"""
Reasoning critique (Strategy 3) for SCOUT framework.
Audits the chain-of-thought reasoning from the main prediction model
for internal contradictions, modality hallucinations, and logic gaps.
Uses a separate checker model (Mcheck).
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
MODEL_NAME = "qwen3-32b"

INPUT_CSV_FILE = "liver_cancer_prediction_with_reasoning.csv"
OUTPUT_CHECK_FILE = "liver_reasoning_audit_results.csv"
ID_COLUMN = "patient_id"

MAX_WORKERS = 1
CALLS_PER_MINUTE = 400
WINDOW_SECONDS = 65

file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==================== Prompt ====================
# NOTE: The prompt is in Chinese because the input reasoning chains are in Chinese.

AUDIT_SYSTEM_PROMPT = """你是一名肝脏影像学逻辑审计专家，专门审核AI生成的CT推理链的**内部逻辑一致性**。

## 背景
- 推理基于**CT平扫**（非增强CT）
- 患者均为肝硬化患者
- MT是恶性肿瘤（Malignant Tumor）的缩写

## 审计标准（检查以下错误）

1. **内部矛盾**：预测结果与CT发现是否自相矛盾？
   - 错误示例：发现描述"单纯囊肿、水样密度、边界清晰"，但预测为"高风险/恶性"
   - 错误示例：发现描述"无局灶性病变"，但预测为"确诊肝癌"

2. **检查方式幻觉**：推理中是否出现平扫CT不可能看到的特征？
   - 错误：提及"动脉期强化"、"门脉期廓清"、"富血供"、"早期强化"、"延迟廓清"（这些需要增强CT才能看到）
   - 注意：有效的平扫CT推理中**绝不应该**出现这些增强相关术语

3. **逻辑断层**：结论是否有文本支持？
   - 错误示例：预测为"confirmed_liver_cancer"但未描述任何肿块或病变
   - 错误示例：预测为"benign_lesion"但发现描述了浸润性边缘和门静脉癌栓

4. **严重程度不匹配**：风险分类是否与发现的严重程度匹配？
   - 错误示例：发现描述"巨大浸润性肿块伴门静脉癌栓"但分类为"benign_lesion"
   - 错误示例：发现描述"单纯囊肿"但分类为"high_risk"

## 输出格式
仅输出JSON，不要包含markdown代码块：
{
  "is_logically_sound": true或false,
  "flagged_issue": "无"或"逻辑错误的简要描述",
  "error_type": "无"或"内部矛盾 / 检查方式幻觉 / 结论无依据 / 严重程度不匹配",
  "audit_reasoning": "简要说明为何通过或未通过审计"
}
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


def parse_json_response(content: str):
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


def call_audit_model(reasoning_detail: str, predicted_category: str,
                     limiter: ThreadSafeRateLimiter):
    """Call checker model to audit reasoning chain."""
    user_content = f"""请审计以下AI生成的肝癌CT平扫预测推理链：

【预测分类】
{predicted_category}

【推理详情】
{reasoning_detail}

请检查是否存在内部矛盾、检查方式幻觉（平扫CT中出现增强相关术语）、逻辑断层等问题。
以JSON格式输出审计结果。
"""
    messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
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
                temperature=0.1,
                max_tokens=1024,
                extra_body={"enable_thinking": False},
            )
            content = completion.choices[0].message.content
            result = parse_json_response(content)
            if result:
                return result
            return {
                "is_logically_sound": "PARSE_ERROR",
                "flagged_issue": "JSON parse failed",
                "error_type": "parse_error",
                "audit_reasoning": content[:500],
            }
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(wait_seconds)
            wait_seconds *= 2

    return {
        "is_logically_sound": "TIMEOUT",
        "flagged_issue": "API call failed after retries",
        "error_type": "timeout",
        "audit_reasoning": "Retries exhausted",
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


def process_single_task(row: pd.Series, limiter: ThreadSafeRateLimiter,
                        fieldnames: list) -> dict:
    """Audit one prediction record."""
    case_id = str(row.get(ID_COLUMN, "")).strip()
    exam_date = str(row.get("exam_date", "")).strip()
    reasoning_detail = str(row.get("reasoning_detail", "")).strip()
    predicted_category = str(row.get("predicted_category", "")).strip()

    try:
        result = call_audit_model(reasoning_detail, predicted_category, limiter)
        is_sound = result.get("is_logically_sound", "Unknown")

        out_row = {
            ID_COLUMN: case_id,
            "exam_date": exam_date,
            "predicted_category": predicted_category,
            "reasoning_detail_preview": (reasoning_detail[:200] + "..."
                                         if len(reasoning_detail) > 200
                                         else reasoning_detail),
            "is_logically_sound": is_sound,
            "flagged_issue": result.get("flagged_issue", ""),
            "error_type": result.get("error_type", ""),
            "audit_reasoning": result.get("audit_reasoning", ""),
        }
        safe_append_row(OUTPUT_CHECK_FILE, fieldnames, out_row)

        if str(is_sound).lower() in ("true", "1", "yes"):
            status = "success"
            print(f"  {case_id} | {predicted_category} | PASS")
        elif str(is_sound).lower() in ("false", "0", "no"):
            status = "flagged"
            print(f"  {case_id} | {predicted_category} | FLAGGED: {result.get('error_type')}")
        else:
            status = "error"
            print(f"  {case_id} | {predicted_category} | ERROR: {is_sound}")

        return {"status": status, "case_id": case_id}

    except Exception as e:
        print(f"  FAILED {case_id}: {e}")
        out_row = {
            ID_COLUMN: case_id,
            "exam_date": exam_date,
            "predicted_category": predicted_category,
            "reasoning_detail_preview": (reasoning_detail[:200] + "..."
                                         if len(reasoning_detail) > 200
                                         else reasoning_detail),
            "is_logically_sound": "ERROR",
            "flagged_issue": f"Exception: {e}",
            "error_type": "processing_error",
            "audit_reasoning": f"ERROR: {type(e).__name__} - {e}",
        }
        safe_append_row(OUTPUT_CHECK_FILE, fieldnames, out_row)
        return {"status": "error", "case_id": case_id}


# ==================== Main ====================

def main():
    print(f"Model: {MODEL_NAME} | Input: {INPUT_CSV_FILE} | Output: {OUTPUT_CHECK_FILE}")
    print(f"Workers: {MAX_WORKERS} | Rate limit: {CALLS_PER_MINUTE}/min")

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"File not found: {INPUT_CSV_FILE}")
        return

    try:
        df = pd.read_csv(INPUT_CSV_FILE, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(INPUT_CSV_FILE, encoding="gbk")
        except Exception:
            df = pd.read_csv(INPUT_CSV_FILE, encoding="gb18030")

    print(f"Total records: {len(df)}")

    # Auto-detect ID column
    id_column = ID_COLUMN
    if id_column not in df.columns:
        for col in ["patient_id", "technology_id", "hadm_id", "case_id", "id"]:
            if col in df.columns:
                id_column = col
                break
        else:
            id_column = df.columns[0]
            print(f"Warning: using first column as ID: {id_column}")

    fieldnames = [
        id_column, "exam_date", "predicted_category", "reasoning_detail_preview",
        "is_logically_sound", "flagged_issue", "error_type", "audit_reasoning",
    ]

    # Resume from checkpoint
    processed_ids = set()
    if os.path.exists(OUTPUT_CHECK_FILE) and os.path.getsize(OUTPUT_CHECK_FILE) > 0:
        try:
            df_done = pd.read_csv(OUTPUT_CHECK_FILE, usecols=[id_column], encoding="utf-8-sig")
            processed_ids = set(df_done[id_column].astype(str).str.strip())
            print(f"[Resume] Already processed: {len(processed_ids)}")
        except Exception as e:
            print(f"Warning reading checkpoint: {e}")

    tasks = []
    for _, row in df.iterrows():
        case_id = str(row.get(id_column, "")).strip()
        if case_id in processed_ids:
            continue
        reasoning = str(row.get("reasoning_detail", "")).strip()
        pred_cat = str(row.get("predicted_category", "")).strip()
        if not reasoning or reasoning.lower() == "nan" or len(reasoning) < 20:
            continue
        if pred_cat in ["ERROR", "TIMEOUT_ERROR", "PARSE_ERROR"]:
            continue
        tasks.append(row)

    print(f"Pending: {len(tasks)}")

    if not tasks:
        print("Nothing to process.")
        return

    limiter = ThreadSafeRateLimiter(CALLS_PER_MINUTE, WINDOW_SECONDS)
    success_count, flagged_count, error_count = 0, 0, 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_task, row, limiter, fieldnames)
                   for row in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                if result["status"] == "success":
                    success_count += 1
                elif result["status"] == "flagged":
                    flagged_count += 1
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
                print(f"  Exception: {e}")
            if i % 10 == 0 or i == len(futures):
                elapsed = time.time() - start_time
                rate = i / elapsed * 60 if elapsed > 0 else 0
                print(f"Progress: {i}/{len(futures)} | Pass: {success_count} | "
                      f"Flagged: {flagged_count} | Error: {error_count} | Rate: {rate:.1f}/min")

    total = success_count + flagged_count + error_count
    elapsed_total = time.time() - start_time
    print(f"\nDone. Total: {total}, Pass: {success_count} "
          f"({success_count / max(total, 1) * 100:.1f}%), "
          f"Flagged: {flagged_count} ({flagged_count / max(total, 1) * 100:.1f}%), "
          f"Error: {error_count}, Time: {elapsed_total / 60:.1f} min")

    if os.path.exists(OUTPUT_CHECK_FILE):
        try:
            df_result = pd.read_csv(OUTPUT_CHECK_FILE, encoding="utf-8-sig")
            print("\nAudit result distribution:")
            print(df_result['is_logically_sound'].value_counts().to_string())
            flagged_df = df_result[df_result['is_logically_sound'] == False]
            if len(flagged_df) > 0:
                print("\nFlagged error types:")
                print(flagged_df['error_type'].value_counts().to_string())
        except Exception as e:
            print(f"Stats error: {e}")


if __name__ == "__main__":
    main()
