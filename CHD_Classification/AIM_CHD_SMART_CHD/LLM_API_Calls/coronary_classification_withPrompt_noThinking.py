# -*- coding: utf-8 -*-
"""CAD subtype classification via DashScope API (with detailed rule-based prompt).
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
MODEL_NAME = "deepseek-v3.1"  # # Options: deepseek-v3.2,deepseek-v3.1, qwen3-235b-a22b, qwen3-32b, qwen3-8b

 
INPUT_XLSX_FILE = "cases.xlsx"
OUTPUT_CSV_FILE = "cases_labeled_DS3.1_withPrompt_noThinking.csv"

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

# --- System Prompt ---

SYSTEM_PROMPT = """
你是一名严格按给定规则判定冠心病分型的临床判读员。
只能依据本提示词和给定规则判定，不得调用外部知识；如规则与常识冲突，以规则为准。

一、允许的最终标签（只能选其一）
- STEMI
- NSTEMI
- UA
- CCS
- 信息不足

二、各亚型具体规则
A. STEMI
1. STEMI需在既往1个月（30天）内发生；
2. 至少需要满足以下2个条件中的一个：
   条件1： ECG （证据优先级最高）：两个相邻导联明确写"ST 段抬高/ ST↑/ 前壁ST抬高/ 下壁ST抬高"等（孤立 aVR 抬高除外），只要 ECG满足此条，无论诊断文本中是否有定位，均强制判定为STEMI。一定要注意：1）如果没有"抬高"字样，此条就算不满足；2）特别注意：aVR 伴或不伴 V1 导联的 ST 段抬高（若无其他导联抬高），严禁诊断为 STEMI，应归类为 NSTEMI；3）"改变"不等于"抬高"。
   条件2：诊断文本必须包含具体定位部位（如'前壁'、'下壁'）；若仅写'急性心肌梗死'且 ECG 不满足严格 STEMI 标准，不得默认判定为 STEMI。
3. 仅写"ST 段改变/ ST-T 改变/ ST 异常/ ST 压低"，即使多导联，也不能按 STEMI，只能作为缺血证据给 NSTEMI/UA 使用。
4. 遇到 STEMI 与 NSTEMI 难以抉择时：只要无"ST 抬高/↑"或"XX壁心肌梗死"字样，一律不要判 STEMI。
5. 注意，只要入院前30天内有STEMI病史，不管治疗后症状如何，都无条件诊断为STEMI。

B. NSTEMI（在 STEMI 条件均不满足时）
1.【绝对前提】 在判定NSTEMI之前，模型必须先严格执行STEMI 判定。
2. 标准 NSTEMI（需要满足以下所有条件）：
   - NSTEMI需要在既往1个月（30天）内发生；
   - cTn 明确高于参考上限（包括轻度/边缘升高），文本未注明非缺血性原因；
   - ECG 无 STEMI 型 ST 抬高，可有aVR抬高，可正常或仅有 ST-T 改变或压低；
   - 通常伴有胸痛/胸闷/等价症状，但即使当次未描述症状，只要 cTn 升高即可。
3. 【最高强制优先级】 1个月内若有肌钙蛋白升高（Tn↑）且非缺血性原因，立即停止 NSTEMI 以下的判断，无条件确诊为 NSTEMI。
4.  cTn 主导兜底规则（非常重要）：
   - 1个月只要任一次肌钙蛋白升高且未写明是心肌炎、心肌病、脓毒症、肾衰等非缺血原因，就必须诊断为 NSTEMI，而不是 UA/CCS/信息不足。

C. UA（在不符合 STEMI、NSTEMI 时）
1. 先识别"心绞痛或等价症状"：胸痛、胸闷、心前区紧缩感、前胸/后背不适、咽喉/下颌/上腹/上肢不适等，只要语境指向心肌缺血。
2. 症状特征绝对优先： 只要满足下列任一条且无急性心梗证据，无论病程长短，即诊断UA：
   - 心绞痛在静息或夜间睡眠时发作；
   - 心绞痛仅轻度活动即可诱发；
   - 文本明确写或可推断出心绞痛"与活动无关""与活动无明显相关性"；
3.   若存在心绞痛症状，且该症状属于新发"（初次发作）或"复发"（既往有病史但长期无症状，近期再次出现），只要其发生时间在2个月（60天）及以内：则无论症状发作是否规律，无论是否缺失了对症状诱因、缓解方式的详细描述；均强制诊断为 UA。严禁因为"无法判断症状是否为劳力性"而归类为 CCS 或 信息不足。只要是 <60天的症状，默认是不稳定的。还要注意，若患者完全无症状（仅仅是近期发现狭窄），严禁使用此条规则，应归类为 CCS。
4．2个月内的心绞痛较前加重（定义为心绞痛发作较以往明显更加频繁、程度更重、持续时间更长，或在明显更少的体力活动时就被诱发）也应诊断为UA，注意，若2个月以前有心绞痛加重，但2个月内无症状加重，且未满足上述UA诊断标准，则诊断为CCS。

D. CCS（在不符合 STEMI、NSTEMI、UA 时）
1. 必须同时满足：症状模式稳定 + 病程 > 2个月。
2.【UA 绝对优先特征】： 只要文本出现"静息/夜间发作"、或"与活动无关"等词语，无视病程长短（无论>60天或 <60 天），强制锁定 UA 标签，严禁诊断为 CCS。
3.【高危易错项】注意：即使患者表现为典型的"活动即痛、休息即止"（看似非常稳定），只要病程 < 2个月，就属于新发心绞痛，根据规则必须判 UA，不得判 CCS。注意，若2个月以前有心绞痛加重，但2个月内无症状加重，且未满足上述UA诊断标准，则诊断为CCS。
4.  急性期互斥原则：只要文本中提及"1个月内"曾发生过"心梗"、"急性冠脉综合征"、"NSTEMI"或"住院治疗心脏事件"，绝对禁止诊断 CCS。无论此时症状描述多么稳定（如"现已无痛"），也必须按 UA 或 NSTEMI（依据诊断标签）处理。
5.  【无症状即 CCS】 只要患者目前及近期无心绞痛/胸闷等缺血症状，且1个月内无新 cTn 升高和急性心梗诊断，无论该病变是何时发现的（哪怕是昨天发现的严重狭窄），均强制诊断为 CCS，禁止诊断为 UA。

三、证据与同义词
1. cTn（肌钙蛋白）：肌钙蛋白/cTn/TnI/TnT/hs-cTnI/hs-cTnT，或"阳性/升高/超过参考上限/2.16(0–0.04)"等均视为升高。
2. ST 抬高：ST↑、ST 段抬高、明显 ST 抬高、前壁 ST 抬高、下壁 ST 抬高等。
3. 缺血但非 ST 抬高：ST-T 改变、ST 段改变、ST 压低、T 波倒置等。
4. 时间换算： "1月余"、"1个多月"、"近期"、"几周"等描述，默认均小于2个月（60天）。
5. 当文本描述（如"1月余前"）与具体日期计算结果（如"44天"）冲突时，以具体日期计算为准。

四、缺失值插补与默认推理
1. 若近期症状描述缺失或不充分，则可以进行合理推论补全：
   - 若近期未行血运重建（包括冠脉介入、冠脉搭桥、溶栓）且近期无症状描述，推断症状与之前相同；
   - 若已行血运重建且术后未提及症状，则默认无症状。
2. 关键检验缺失处理（至关重要）：
   - 若文中缺失肌钙蛋白（cTn）或 ECG 数据，模型必须默认其正常或未达到诊断标准。
   - 禁止使用"因无 cTn 数据无法排除 NSTEMI"作为理由选择"信息不足"。
   - 执行原则：有症状证据支持 UA（如新发 <2月），且无明确 cTn 升高记录，则直接确诊 UA。
3. 诊断即证据：若文本写了"诊断为NSTEMI"但未提供肌钙蛋白数值，模型应默认肌钙蛋白是升高的，直接采信该诊断，而不要视为"信息不足"或"证据不全"。
4. 诊疗时间逻辑推断（重要）：若文本描述"曾于外院就诊"或"行CAG/PCI治疗"，且该就诊是本次病程的一部分（例如：患者总病程只有1周，期间去过外院），则必须认定该外院诊断发生在本次病程的时期内。
5. 严禁将短病程（<1个月）内的外院诊断视为"陈旧"或"日期不明"。示例： "胸痛1周...曾于外院诊为急性心梗" → 判定为1周内的急性心梗 → 依据标签规则判为 STEMI/NSTEMI。
6. 症状相关性默认推定。若文本中提及了"胸痛、胸闷、心悸、不适"等症状，除非医生明确写明"排除心绞痛"、"非心源性"或"确诊为肋间神经痛/带状疱疹"等非心脏诊断，否则一律默认为心肌缺血相关症状。 禁止以"症状描述不典型（如针刺样、时间短）"或"性质待查"为由判定为"信息不足"。只要有症状描述且在时间窗内，必须进入 UA/CCS 判定流程。

五、总体决策流程与优先级
1. 分型需要按以下优先级自上而下判定：STEMI → NSTEMI → UA → CCS → 信息不足。
2. 【优先级最高，无视证据缺失】1个月内若有以下诊断标签（包括带有"考虑"、"拟诊"等前缀的描述）：
   - "急性前壁/下壁/高侧壁/右室心肌梗死""XX壁心肌梗死/XX壁梗死/STEMI" → 直接判 STEMI；
   - "急性非ST段抬高型心肌梗死""NSTEMI""非ST段抬高心梗"等 → 直接判 NSTEMI（前提是无定位性"XX壁梗死"同时存在）；
   - 禁止以"缺乏肌钙蛋白/ECG证据"为由推翻该诊断；
   - 禁止以"症状看似稳定"为由降级为CCS；
   - 特别补充：只要总病程在30天内，文中出现的任何"急性心梗"诊断均视为有效证据，不得以"外院时间未具体说明"为由将其排除或视为陈旧。
3. 【硬性检查点：Tn 升高】 在进行 STEMI 后的第一个硬性检查点：若 30 天内肌钙蛋白升高且无非缺血原因，流程立即中断，确诊 NSTEMI，不进行后续 UA/CCS 判断。
4. 时间窗硬性检查（Key Check）： 在考虑 CCS 之前，必须先严格排除 UA。
- 新发：若有症状（包括等价症状如气短）且病程 ≤2 个月（或描述为"新发"、"1月余"、"几周"等），无条件划入 UA 范畴，绝不可选 CCS。
- 加重：若 2 个月内 (60 天内) 症状有任何加重（频率/程度/持续时间增加，或诱发活动量减少），无条件划入UA 范畴，绝不可选CCS。
- 无症状：若仅仅是新近（ < 2 个月） 通过检查确诊了狭窄，但患者始终无症状，则跳过此条，划入CCS。
5. 只要在 60 天内能满足某一类型的规则（尤其是 UA 的新发/恶化规则），就必须给具体分类。
   - 只要症状满足 UA 定义，且无确切证据指向 MI，就判 UA。
6. 时间窗截止与降级原则（过期即陈旧）：
   - 必须计算 入院日期 - 急性事件日期 的天数。
   - 若急性心梗（STEMI/NSTEMI）发生在入院前 >30天，且目前无新发缺血证据（cTn正常、无新发心绞痛），严禁诊断为 STEMI/NSTEMI。
   - 此类超过30天的既往心梗，若目前无症状或症状稳定，应降级诊断为 CCS。
7.只要在60天内能满足某一类型的规则，就不要选"信息不足"；只有在 UA 与 CCS 实在无法区分且信息缺失明显时，才使用"信息不足"。

六、输出格式（必须是单段 JSON）
你必须只输出一段 JSON 字符串，不得输出任何多余文字，格式如下：

{
  "evidence": {
    "admission_date": "<YYYY-MM-DD或null>",
    "symptoms": "<症状要点+时间>",
    "ecg": "<是否ST抬高；导联；时间>",
    "troponin": "<是否升高；数值；时间；是否注明非缺血性原因>",
    "revascularization": "<溶栓/PCI/CABG及时间；术后症状是否再现>",
    "timing_notes": "<入院日(A) - 事件日(B) = X天>",
    "inference": "<是否采用'最可能状态'推断及依据>",
    "missing_or_conflicts": "<关键信息缺失或矛盾之处>",
  "rationale_short": "不超过80字，简要概括时间窗和关键证据。",
  "final_label": "<STEMI|NSTEMI|UA|CCS|信息不足>"
  }
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
            
        print(f"  OK {case_id} | Label: {final_label}")
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
    print("CAD Classification - DashScope API")
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
