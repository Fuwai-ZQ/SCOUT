"""
Step 4: Analyze discordance between gold_category and pred_main.
Uses an LLM to diagnose error types and subtypes for each misclassified case.
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

INPUT_CSV_FILE = "prediction_discordant_cases.csv"
OUTPUT_CSV_FILE = "output/inconsistency_analysis_result.csv"
ID_COLUMN = "patient_id"

MAX_WORKERS = 80
CALLS_PER_MINUTE = 200
WINDOW_SECONDS = 65

file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==================== Prompt ====================
# NOTE: The prompt is in Chinese because the input CT reports are in Chinese.

SYSTEM_PROMPT = """你是一位资深的放射科医师和医学AI专家，擅长分析CT影像诊断中的分类不一致问题。

## 背景
在肝硬化患者的肝细胞癌（HCC）风险评估任务中，我们使用CT平扫报告进行四分类预测：
- confirmed_liver_cancer：明确肝癌/肝恶性肿瘤，或伴门静脉癌栓
- high_risk：可疑低密度病灶、边界不清占位、性质待定结节（非明确囊肿）
- benign_lesion：明确良性病变（肝囊肿、血管瘤、钙化灶）
- no_focal_lesion：仅肝硬化改变，无局灶性病变

## 你的任务
分析"检查结果"（CT影像的客观描述）与"检查结论"（放射科医生的诊断结论）之间的信息差异，解释为什么基于"检查结果"的AI预测（pred_main）与基于"检查结论"的标准分类（gold_category）不一致。

## 错误类型定义

### 1. 漏检型错误（missed_detection）
- 检查结果中存在病灶描述，但pred_main未识别或低估其风险
- 子类型：
  - missed_focal_lesion：遗漏了明确描述的病灶
  - underestimated_risk：识别了病灶但低估其恶性风险
  - missed_indirect_signs：遗漏了间接征象（门脉癌栓、淋巴结转移等）

### 2. 过度警惕型错误（over_alerting）
- 将正常变化或良性病灶误判为高风险
- 子类型：
  - benign_as_suspicious：将明确良性病灶判为可疑
  - normal_as_lesion：将正常结构或伪影判为病灶
  - cirrhosis_only_as_risk：仅有肝硬化改变却判为高风险

### 3. 信息缺失型错误（information_gap）
- 检查结果中缺少检查结论所依据的关键信息
- 子类型：
  - key_finding_not_described：关键发现未在检查结果中描述
  - insufficient_detail：描述过于简略无法准确分类
  - external_info_required：需要结合其他检查（如增强CT、病理）才能确诊

### 4. 推理偏差型错误（reasoning_bias）
- 推理过程存在逻辑问题
- 子类型：
  - over_interpretation：对模糊征象过度解读
  - under_interpretation：对可疑征象解读不足
  - context_ignored：忽略了肝硬化背景对风险评估的影响
  - feature_misattribution：特征归因错误（如将其他器官病变误归于肝脏）

### 5. 分类边界错误（boundary_error）
- 病例处于分类边界，难以明确归类
- 子类型：
  - ambiguous_lesion_nature：病灶性质本身难以确定
  - multiple_lesions_mixed：多发病灶性质不一致时的分类困难

## 输出格式
请仅输出JSON，格式如下：
{
    "error_type": "主错误类型（英文）",
    "error_subtype": "子类型（英文）",
    "error_analysis": "详细的中文错误原因分析（200-400字），需要：1)指出具体哪些信息导致了不一致；2)说明为什么会产生这种错误；3)给出合理的改进建议"
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


def call_model_for_analysis(row_data: dict, limiter: ThreadSafeRateLimiter):
    """Call LLM to analyze the discordance for a single case."""
    user_content = f"""## 病例信息

### 检查结果（CT影像客观描述）
{row_data['ct_findings']}

### 检查结论（放射科医生诊断）
{row_data['ct_impression']}

### 分类结果对比
- gold_category（基于检查结论的标准分类）: {row_data['gold_category']}
- pred_main（基于检查结果的AI预测）: {row_data['pred_main']}

### AI推理过程
{row_data.get('reasoning_detail', 'N/A')}

## 请分析
请分析上述病例中gold_category与pred_main不一致的原因，输出JSON格式的分析结果。"""

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
                return {
                    "parse_error": True,
                    "raw_response": full_answer,
                    "thinking_trace": full_reasoning,
                    "error_type": "PARSE_ERROR",
                    "error_subtype": "PARSE_ERROR",
                    "error_analysis": f"Parse failed. Raw: {full_answer[:500]}",
                }

            json_data["thinking_trace"] = full_reasoning
            json_data["raw_response"] = full_answer
            return json_data

        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(wait_seconds)
            wait_seconds *= 2

    return {
        "timeout_error": True,
        "error_type": "TIMEOUT_ERROR",
        "error_subtype": "TIMEOUT_ERROR",
        "error_analysis": "API call timed out",
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
    """Process a single discordant case."""
    case_id = str(row.get(ID_COLUMN, "")).strip()

    try:
        row_data = {
            'ct_findings': str(row.get('ct_findings', '')).strip(),
            'ct_impression': str(row.get('ct_impression', '')).strip(),
            'gold_category': str(row.get('gold_category', '')).strip(),
            'pred_main': str(row.get('pred_main', '')).strip(),
            'reasoning_detail': str(row.get('reasoning_detail', '')).strip(),
        }

        result = call_model_for_analysis(row_data, limiter)

        out_row = {
            ID_COLUMN: case_id,
            "exam_date": row.get("exam_date", ""),
            "center": row.get("center", ""),
            "gold_category": row_data['gold_category'],
            "pred_main": row_data['pred_main'],
            "error_type": result.get("error_type", "N/A"),
            "error_subtype": result.get("error_subtype", "N/A"),
            "error_analysis": (result.get("error_analysis") or "")[:3000],
            "model_thinking": (result.get("thinking_trace") or "")[:3000],
        }
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        print(f"  {case_id} | {row_data['gold_category']} vs {row_data['pred_main']} "
              f"| error: {result.get('error_type', 'N/A')}")
        return True

    except Exception as e:
        print(f"  FAILED {case_id}: {e}")
        out_row = {
            ID_COLUMN: case_id,
            "exam_date": row.get("exam_date", ""),
            "center": row.get("center", ""),
            "gold_category": str(row.get('gold_category', '')).strip(),
            "pred_main": str(row.get('pred_main', '')).strip(),
            "error_type": "PROCESSING_ERROR",
            "error_subtype": "PROCESSING_ERROR",
            "error_analysis": f"{type(e).__name__}: {e}",
            "model_thinking": "",
        }
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        return False


# ==================== Main ====================

def main():
    print(f"Model: {MODEL_NAME} | Input: {INPUT_CSV_FILE} | Output: {OUTPUT_CSV_FILE}")

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"File not found: {INPUT_CSV_FILE}")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)

    try:
        df = pd.read_csv(INPUT_CSV_FILE, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(INPUT_CSV_FILE, encoding="utf-8")

    print(f"Total records: {len(df)}")

    fieldnames = [
        ID_COLUMN, "exam_date", "center",
        "gold_category", "pred_main",
        "error_type", "error_subtype", "error_analysis",
        "model_thinking",
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
        if not case_id or case_id == "nan":
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
            print("\nError type distribution:")
            print(df_result['error_type'].value_counts().to_string())
            print("\nError subtype distribution:")
            print(df_result['error_subtype'].value_counts().to_string())
        except Exception as e:
            print(f"Stats error: {e}")


if __name__ == "__main__":
    main()
