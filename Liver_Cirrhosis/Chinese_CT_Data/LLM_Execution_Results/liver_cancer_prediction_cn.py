"""
Liver cancer risk prediction from CT findings with chain-of-thought reasoning.
Uses LLM to classify CT findings into 4 risk categories with detailed diagnostic reasoning.
This is the main model (Mmain) in the SCOUT framework.
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
OUTPUT_CSV_FILE = "liver_cancer_prediction_with_reasoning.csv"

ID_COLUMN = "patient_id"
TEXT_COLUMN = "ct_findings"

MAX_WORKERS = 150
CALLS_PER_MINUTE = 200
WINDOW_SECONDS = 65

file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==================== Prompt ====================
# NOTE: The prompt is in Chinese because the input CT reports are in Chinese.

SYSTEM_PROMPT = """你是一名资深的肝脏影像学专家，专门通过CT平扫影像诊断肝硬化患者的并发症。你的任务是在保证高特异性（不乱报假阳性）的前提下，敏锐地识别出伪装性强的恶性病变。

## 核心逻辑升级
1. **区分"描述"与"结论"**：影像描述中的"囊样"不等于结论中的"囊肿"。若结论未下定论，视为高风险。
2. **警惕"混合征象"**：良性特征（如钙化、囊变）若出现在实性结节内部，反而是恶性征象。
3. **重视"放射科医生的犹豫"**：任何包含"性质待定"、"建议增强"、"？"的描述，在肝硬化背景下均升级为高风险。

## 判读标准（精细化版）

### 一、绝对高风险 (Confirmed/High Risk)
1. **明确恶性关键词**：肝癌、HCC、MT、占位、恶性肿瘤、新发病灶。
2. **典型恶性形态**：
   - 肿块伴门静脉癌栓（充盈缺损）。
   - 密度不均的低密度灶（提示坏死/出血）。
   - 边界模糊、呈分叶状或浸润性生长。
3. **【关键改进】伪装成良性的恶性征象**：
   - **不纯的钙化**：描述为"结节样混杂密度影，内见钙化"（非单纯钙化灶）。
   - **不纯的囊性灶**：描述为"囊实性"、"囊壁增厚"、"囊内分隔"或"囊样低密度，性质待定"。
   - **危险的弥漫病变**：描述为"弥漫性结节" 且同时伴有 "门静脉癌栓/充盈缺损" 或 "新发大量腹水"。

### 二、典型良性 (Benign Lesion) - 需严格满足以下条件
**只有完全符合以下情况，才能归为良性（自信排除）：**

1. **确诊的囊肿**：
   - 报告结论部分明确写有"肝囊肿"。
   - 或描述为"无壁"、"水样密度"且"边界锐利/清晰"。
   - **注意**：仅描述为"囊样"但结论写"性质待定"者，**不**属于此类，应归为高风险。

2. **单纯钙化灶**：
   - 描述为"高密度钙化灶"、"肝内钙化点"。
   - **排除**：主要为低密度结节，仅内部有点状钙化者（归为高风险）。

3. **血管瘤**：
   - 报告明确提及"血管瘤"或"血管瘤可能"。

4. **陈旧性病变/术后改变**：
   - 明确提及"术后改变"、"碘油沉积"（高密度）、"治疗后无变化"。

### 三、无局灶病变 (No Focal Lesion)
- 仅有肝硬化背景改变（肝表面不平、比例失调、再生结节）。
- **注意**：如果报告提及"弥漫性低密度结节"且未明确排除恶性，若同时伴有门静脉主干模糊或充盈缺损，应升级为高风险；否则可视为背景。

---

## 决策思维链（CoT）要求
在 `diagnostic_reasoning` 中，请执行以下检查：
1. **语义检查**：这是"囊肿"（诊断）还是"囊样"（描述）？如果是后者且无定论，倾向高风险。
2. **纯度检查**：这是"纯钙化"还是"结节含钙化"？后者高风险。
3. **时效检查**：这是"较前相仿"（良性倾向）还是"新发/增大"（恶性倾向）？
4. **结论验证**：放射科医生的最终结论是否包含"性质待定"、"建议进一步检查"？如果有，必须High Risk。

## 输出格式 (JSON)
{
  "cirrhosis_signs": "...",
  "plain_ct_findings": {
    "focal_lesion_detected": "Yes/No",
    "lesion_description": "...",
    "suspicious_indicators": ["列出具体的警示词，如：性质待定、边界欠清、结节伴钙化"]
  },
  "diagnostic_reasoning": "...",
  "risk_assessment": {
    "category": "confirmed_liver_cancer / high_risk / benign_lesion / no_focal_lesion",
    "confidence_score": 0-100,
    "key_evidence": "..."
  }
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


def call_model(ct_findings: str, limiter: ThreadSafeRateLimiter):
    """Call LLM to predict liver cancer risk from CT findings."""
    user_content = f"""请根据以下CT报告的检查结果，分析并预测患者是否存在肝脏恶性肿瘤：

## CT检查结果
{ct_findings}

请按照要求的JSON格式输出你的分析结果。"""

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
                    "prediction": {"category": "PARSE_ERROR", "confidence": "", "key_evidence": ""},
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
        "prediction": {"category": "TIMEOUT_ERROR", "confidence": "", "key_evidence": ""},
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
    ct_findings = str(row.get(TEXT_COLUMN, "")).strip()

    try:
        result = call_model(ct_findings, limiter)
        plain_ct = result.get("plain_ct_findings", {})
        risk = result.get("risk_assessment", {})

        # Assemble structured reasoning detail
        parts = []
        if result.get("cirrhosis_signs"):
            parts.append(f"[Cirrhosis] {result['cirrhosis_signs']}")
        ct_parts = []
        for key in ("focal_lesion_detected", "lesion_description", "mass_effect",
                     "portal_vein_thrombus", "other_anomalies"):
            if plain_ct.get(key):
                ct_parts.append(f"{key}: {plain_ct[key]}")
        if ct_parts:
            parts.append(f"[CT findings] {'; '.join(ct_parts)}")
        if result.get("diagnostic_reasoning"):
            parts.append(f"[Reasoning] {result['diagnostic_reasoning']}")
        risk_parts = []
        for key in ("confidence_score", "key_warning_signs", "clinical_advice"):
            if risk.get(key):
                risk_parts.append(f"{key}: {risk[key]}")
        if risk_parts:
            parts.append(f"[Risk] {'; '.join(risk_parts)}")

        out_row = {
            ID_COLUMN: case_id,
            "exam_date": row.get("exam_date", ""),
            "center": row.get("center", ""),
            "predicted_category": risk.get("category", ""),
            "reasoning_detail": "\n".join(parts),
            "model_thinking": (result.get("thinking_trace") or "")[:5000],
        }
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        print(f"  {case_id} | pred: {risk.get('category')} | conf: {risk.get('confidence_score')}")
        return True

    except Exception as e:
        print(f"  FAILED {case_id}: {e}")
        out_row = {
            ID_COLUMN: case_id,
            "exam_date": row.get("exam_date", ""),
            "center": row.get("center", ""),
            "predicted_category": "ERROR",
            "reasoning_detail": f"ERROR: {type(e).__name__} - {e}",
            "model_thinking": "",
        }
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        return False


# ==================== Main ====================

def read_csv_auto(filepath):
    """Try multiple encodings to read a CSV file."""
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'latin-1']:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot read {filepath} with any supported encoding")


def main():
    print(f"Model: {MODEL_NAME} | Input: {INPUT_CSV_FILE} | Output: {OUTPUT_CSV_FILE}")

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"File not found: {INPUT_CSV_FILE}")
        return

    df = read_csv_auto(INPUT_CSV_FILE)
    print(f"Total records: {len(df)}")

    fieldnames = [
        ID_COLUMN, "exam_date", "center",
        "predicted_category", "reasoning_detail", "model_thinking",
    ]

    # Resume from checkpoint
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV_FILE) and os.path.getsize(OUTPUT_CSV_FILE) > 0:
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'latin-1']:
            try:
                df_done = pd.read_csv(OUTPUT_CSV_FILE, usecols=[ID_COLUMN], encoding=enc)
                processed_ids = set(df_done[ID_COLUMN].astype(str).str.strip())
                print(f"[Resume] Already processed: {len(processed_ids)}")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                print(f"Warning reading checkpoint: {e}")
                break

    tasks = []
    skip_id, skip_text, skip_done = 0, 0, 0
    for _, row in df.iterrows():
        case_id = str(row.get(ID_COLUMN, "")).strip()
        ct_text = str(row.get(TEXT_COLUMN, "")).strip()
        if not case_id or case_id == "nan":
            skip_id += 1; continue
        if not ct_text or ct_text.lower() == "nan" or len(ct_text) < 20:
            skip_text += 1; continue
        if case_id in processed_ids:
            skip_done += 1; continue
        tasks.append(row)

    print(f"Skipped: {skip_id} invalid ID, {skip_text} invalid text, {skip_done} already done")
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
            df_result = read_csv_auto(OUTPUT_CSV_FILE)
            print("\nPrediction distribution:")
            print(df_result['predicted_category'].value_counts().to_string())
        except Exception as e:
            print(f"Stats error: {e}")


if __name__ == "__main__":
    main()
