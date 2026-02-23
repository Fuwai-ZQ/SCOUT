"""
Step 2: Classify liver lesions from CT impression via LLM.
Outputs liver_lesion_category (4-class) and prior_cancer_history for each patient.
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
MODEL_NAME = "deepseek-v3.2"

INPUT_FILE = "portal_cohort_deduped.xlsx"
OUTPUT_CSV_FILE = "liver_cancer_classification_result.csv"

ID_COLUMN = "patient_id"
TEXT_COLUMN = "ct_impression"

MAX_WORKERS = 15
CALLS_PER_MINUTE = 150
WINDOW_SECONDS = 65
ENABLE_THINKING = False

file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==================== Prompt ====================
# NOTE: The prompt is in Chinese because the input CT reports are in Chinese.

SYSTEM_PROMPT = """你是一名经验丰富的肝脏病理学专家和影像科医生。你的任务是分析CT报告的检查结论，判断患者的肝脏病变情况。

## 重要提示
- **MT是恶性肿瘤（Malignant Tumor）的缩写**，出现MT应理解为恶性肿瘤相关描述。

## 分类任务

### 任务1：肝脏病变四分类（liver_lesion_category）

请根据检查结论内容，将肝脏病变分为以下四类之一：

1. **confirmed_liver_cancer**（确诊肝癌）
   - 明确诊断为肝细胞癌（HCC）、肝癌、胆管癌、肝脏恶性肿瘤
   - 明确提到肝脏转移瘤、肝转移
   - 使用了"肝癌"、"肝恶性肿瘤"、"肝脏MT"、"肝MT"、"胆管细胞癌"等明确恶性诊断词汇
   - 明确提及肝脏占位且倾向恶性/考虑恶性

2. **high_risk**（肝癌高风险）
   - 存在可疑肝脏占位性病变、可疑肝脏结节
   - 存在需要进一步检查的不确定性肝脏结节/肿块
   - 描述中使用"不除外恶性"、"恶性待排"、"性质待定"、"建议进一步检查"等措辞描述肝脏病变
   - 存在增强特征异常的肝脏结节（动脉期强化、门脉期廓清等）
   - 肝脏占位性病变性质不明

3. **benign_lesion**（良性病变）
   - 明确诊断为肝囊肿
   - 明确诊断为肝血管瘤
   - 明确诊断为局灶性结节增生（FNH）
   - 明确诊断为肝脂肪瘤或其他良性病变
   - 稳定的、特征明确的肝脏良性病变
   - 肝内钙化灶

4. **no_focal_lesion**（无局灶性病变）
   - 无肝脏占位性病变的描述
   - 仅描述弥漫性病变如肝硬化、脂肪肝、肝淤血等
   - 明确说明"肝脏未见明显占位"或类似表述
   - 报告中完全未提及肝脏相关内容

### 任务2：既往癌症病史判断（prior_cancer_history）

判断报告中是否可以推断出患者既往有癌症病史（任何部位）：

- **yes**：有明确证据表明既往癌症病史
  - 提到"XX癌术后"、"XX癌病史"
  - 提到"恶性肿瘤治疗后"
  - 提到其他部位的转移瘤来源
  - 提到"MT术后"、"MT治疗后"

- **no**：无法从报告中判断出既往癌症病史

- **uncertain**：有一些线索但不确定
  - 提到可能与肿瘤相关的术后改变但未明确说明
  - 有一些可疑但未确认的表述

## 输出要求

**必须严格输出JSON格式，不要包含任何推理过程、Markdown标记或额外文字。**

输出格式：
{"liver_lesion_category": "分类结果", "prior_cancer_history": "yes/no/uncertain", "confidence": "high/medium/low", "key_findings": "简要说明关键发现（中文，50字以内）"}

## 分类示例

示例1 - 确诊肝癌：
检查结论: "肝右叶占位，考虑肝细胞癌可能大。肝硬化，门脉高压，脾大。"
输出: {"liver_lesion_category": "confirmed_liver_cancer", "prior_cancer_history": "no", "confidence": "high", "key_findings": "肝右叶占位考虑肝细胞癌"}

示例2 - 高风险：
检查结论: "肝右叶结节，性质待定，建议MRI进一步检查。肝硬化。"
输出: {"liver_lesion_category": "high_risk", "prior_cancer_history": "no", "confidence": "high", "key_findings": "肝右叶结节性质待定需进一步检查"}

示例3 - 良性病变：
检查结论: "肝囊肿。肝硬化伴腹水。"
输出: {"liver_lesion_category": "benign_lesion", "prior_cancer_history": "no", "confidence": "high", "key_findings": "肝囊肿"}

示例4 - 无局灶性病变：
检查结论: "肝硬化，门脉高压，脾大，腹水。双肺少许炎症。"
输出: {"liver_lesion_category": "no_focal_lesion", "prior_cancer_history": "no", "confidence": "high", "key_findings": "肝硬化无局灶性病变"}

示例5 - 有既往癌症病史：
检查结论: "肝MT术后改变，未见明确复发征象。肝硬化。"
输出: {"liver_lesion_category": "no_focal_lesion", "prior_cancer_history": "yes", "confidence": "high", "key_findings": "肝MT术后无复发"}

示例6 - 肝转移：
检查结论: "肺癌术后。肝内多发转移瘤。"
输出: {"liver_lesion_category": "confirmed_liver_cancer", "prior_cancer_history": "yes", "confidence": "high", "key_findings": "肝内多发转移瘤"}
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


def call_model(case_text: str, limiter: ThreadSafeRateLimiter):
    """Call LLM API to classify a single CT impression."""
    user_content = f"请分析以下CT报告的检查结论，判断肝脏病变情况：\n\n{case_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    MAX_RETRIES = 3
    wait_seconds = 5

    for attempt in range(MAX_RETRIES):
        limiter.wait()
        try:
            extra_body = {"enable_thinking": True} if ENABLE_THINKING else {}
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=ENABLE_THINKING,
                temperature=0.3,
                top_p=0.7,
                **({
                    "extra_body": extra_body,
                    "stream_options": {"include_usage": True}
                } if ENABLE_THINKING else {})
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
                    "raw_response": full_answer,
                    "_parse_error": True,
                }
            json_data["_thinking_trace"] = full_reasoning
            return json_data

        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(wait_seconds)
            wait_seconds *= 2

    return {
        "liver_lesion_category": "TIMEOUT_ERROR",
        "prior_cancer_history": "TIMEOUT_ERROR",
        "_timeout_error": True,
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


def process_single_task(row: pd.Series, limiter: ThreadSafeRateLimiter, fieldnames: list) -> bool:
    """Process one patient record."""
    case_id = str(row.get(ID_COLUMN, "")).strip()
    raw_text = str(row.get(TEXT_COLUMN, "")).strip()

    try:
        result = call_model(raw_text, limiter)
        out_row = {
            ID_COLUMN: case_id,
            "exam_date": row.get("exam_date", ""),
            "center": row.get("center", ""),
            TEXT_COLUMN: raw_text,
            "liver_lesion_category": result.get("liver_lesion_category", ""),
            "prior_cancer_history": result.get("prior_cancer_history", ""),
            "confidence": result.get("confidence", ""),
            "key_findings": result.get("key_findings", ""),
            "reasoning_content": result.get("_thinking_trace", ""),
            "raw_response": result.get("raw_response", ""),
        }
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        print(f"  {case_id} | category: {result.get('liver_lesion_category')} | "
              f"prior_cancer: {result.get('prior_cancer_history')}")
        return True

    except Exception as e:
        print(f"  FAILED {case_id}: {e}")
        out_row = {
            ID_COLUMN: case_id,
            "exam_date": row.get("exam_date", ""),
            "center": row.get("center", ""),
            TEXT_COLUMN: raw_text,
            "liver_lesion_category": "ERROR",
            "prior_cancer_history": "ERROR",
            "confidence": "", "key_findings": f"ERROR: {type(e).__name__}",
            "reasoning_content": "", "raw_response": str(e),
        }
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        return False


# ==================== Main ====================

def main():
    print(f"Model: {MODEL_NAME} | Input: {INPUT_FILE} | Output: {OUTPUT_CSV_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    if INPUT_FILE.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(INPUT_FILE)
    else:
        try:
            df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(INPUT_FILE, encoding="utf-8")

    print(f"Total records: {len(df)}")

    fieldnames = [
        ID_COLUMN, "exam_date", "center", TEXT_COLUMN,
        "liver_lesion_category", "prior_cancer_history",
        "confidence", "key_findings",
        "reasoning_content", "raw_response",
    ]

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
        raw_text = str(row.get(TEXT_COLUMN, "")).strip()
        if not case_id or case_id == "nan":
            continue
        if not raw_text or raw_text.lower() == "nan" or len(raw_text) < 5:
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
            if i % 20 == 0 or i == len(futures):
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
            print("\nClassification distribution:")
            print(df_result['liver_lesion_category'].value_counts().to_string())
            print("\nPrior cancer history distribution:")
            print(df_result['prior_cancer_history'].value_counts().to_string())
        except Exception as e:
            print(f"Stats error: {e}")


if __name__ == "__main__":
    main()
