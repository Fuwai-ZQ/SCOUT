# -*- coding: utf-8 -*-
"""
Diseased Vessel Counting — Main Model Inference (Optimized Prompt)
Corresponds to Mmain in the SCOUT framework (Strategy 1 & 2).
Uses DeepSeek-V3.1 via Aliyun DashScope API with chain-of-thought reasoning.
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
OUTPUT_CSV_FILE = "disease_vessels_DeepSeek_Aliyun_MultiThread.csv"

ID_COLUMN = "hadm_id"
TEXT_COLUMN = "angiography_report_raw"

MAX_WORKERS = 50
CALLS_PER_MINUTE = 160
WINDOW_SECONDS = 65

file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ========================== Prompt Definition ==========================

TASK_DEFINITION = r'''
{
  "task_description": "Extract coronary artery trunk and branch information from text/structured reports, and determine diseased vessel count (disease_vessel_count), involved trunk set (involved_vessels), and related flags according to unified rules.",

  "task_requirements": [

    "【Vessel Names and Affiliations】
    Trunks: LAD=left anterior descending, LCX=left circumflex, RCA=right coronary artery, LM=left main.
    Branches: D1/D2=diagonal branches; OM/OM1/OM2/RM=obtuse marginal/intermediate branches (all affiliated with LCX); PDA=posterior descending artery; PLA/PLB=posterolateral branches.
    Branch affiliation: D1/D2 → LAD; OM/OM1/OM2/RM → LCX; PDA/PLA/PLB: right-dominant or balanced → RCA, left-dominant → LCX, if dominance unspecified → default to RCA.
    If any trunk or its branch meets the lesion definition, that trunk must be counted as involved, even if the trunk itself is not directly mentioned.",

    "【Bypass Grafts】
    SVG=saphenous vein graft; LIMA/RIMA/BIMA=internal mammary artery; RGEA=right gastroepiploic artery; EIA=inferior epigastric artery.
    Post-CABG: the native trunk is the counting target. When only graft lesions are described, count by the native trunk they supply, and set post_CABG = true.",

    "【Text Contradiction Check】
    '\\n' denotes line breaks. For lesions described as 'patent stent' or 'intimal hyperplasia', there should be no stenosis percentage — likely an OCR error. In such cases, assume stenosis <50%.",

    "【Lesion Definition and Vessel Count (Core Rules)】
    1) Only count whether LAD, LCX, RCA are diseased; LM is not independently counted in disease_vessel_count.
    2) For LAD/LCX/RCA and their branches, any of the following qualifies as diseased:
       - Stenosis ≥70%; or
       - Positive functional test (e.g., FFR/QFR ≤0.80, or equivalent expressions such as 'ischemia-positive'); or
       - Confirmed ISR, re-occlusion, or CTO (see flag rules below).
    3) Stenosis <70% without positive functional evidence or ISR/re-occlusion/CTO must NOT be counted as diseased, even if the original text mentions 'multi-vessel disease'.
    4) LM stenosis ≥50% or functional positivity → set has_LM = true, and both LAD and LCX must be considered diseased (add both to involved_vessels even if not individually described).
    5) Multiple lesions in the same trunk count as only 1 diseased vessel.
    6) Final disease_vessel_count must equal the deduplicated count of trunks (LAD/LCX/RCA) in involved_vessels. No subjective estimation allowed.",

    "【Reason Field Requirements】
    In the reason field, briefly explain per-vessel whether LAD, LCX, RCA is diseased and why, e.g.:
    'PLA 90% affiliated with RCA, thus RCA diseased; LCX distal 99% ≥70%, thus LCX diseased; RCA distal 50% without functional positivity, thus not counted.'",

    "【CTO Flag】
    Descriptions of 'chronic total occlusion/CTO/total occlusion' → mark the corresponding trunk as diseased and set has_CTO = true.",

    "【ISR Flag】
    'In-stent restenosis/ISR' with stenosis ≥50% or re-occlusion → mark as diseased and set has_ISR = true.
    'Patent stent' or mere mention of prior stent without significant stenosis → do NOT count as diseased.",

    "【Functional Criteria Flag】
    FFR/QFR ≤0.80 or textual evidence of ischemia → mark as diseased and set used_functional_criteria = true.",

    "【Independent Judgment】
    Determine vessel count strictly by the above rules. Do NOT reference phrases like 'single/double/triple vessel disease' in the original text, and do NOT adopt the vessel count from the original report."

  ],

  "output_example": {
    "reason": "Example: PLA 90% affiliated with RCA, RCA diseased; LAD proximal 80% ≥70%, LAD diseased; LCX only 40% without functional positivity, not counted. Final: LAD and RCA involved.",
    "disease_vessel_count": 2,
    "involved_vessels": ["LAD", "RCA"],
    "has_LM": false,
    "has_CTO": true,
    "has_ISR": false,
    "used_functional_criteria": true,
    "post_CABG": false
  }
}
'''

OUTPUT_FIELDS = [
    "disease_vessel_count",
    "involved_vessels",
    "has_LM",
    "has_CTO",
    "has_ISR",
    "used_functional_criteria",
    "post_CABG",
    "reason"
]


def build_system_prompt() -> str:
    return f"""You are a Chinese medical information extraction assistant specializing in cardiovascular disease.
Below is the detailed task definition (JSON). Strictly follow the 'task_requirements' and 'output_example'.

Task Definition:
{TASK_DEFINITION}

Requirements:
1) Output exactly one JSON object whose field names and types match the 'output_example'.
2) Do NOT output any explanation, reasoning, or extra text.
3) Do NOT wrap the output in ```json code blocks.
"""


SYSTEM_PROMPT = build_system_prompt()


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
    """Call DashScope API for vessel counting analysis."""
    user_content = f"[Raw Report]\n{case_text}\n\nExtract results strictly per task definition and output JSON."
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
            "full_json_response": json.dumps(result, ensure_ascii=False)
        }
        for key in OUTPUT_FIELDS:
            val = result.get(key, "")
            if isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=False)
            out_row[key] = val

        fieldnames = [ID_COLUMN, TEXT_COLUMN, "reasoning_content"] + OUTPUT_FIELDS + ["full_json_response"]
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        print(f"  Done: {case_id} | vessel_count={result.get('disease_vessel_count', 'N/A')}")
        return True

    except Exception as e:
        print(f"  Failed: {case_id}: {e}")
        out_row = {
            ID_COLUMN: case_id, TEXT_COLUMN: raw_text,
            "reasoning_content": f"ERROR: {type(e).__name__}",
            "full_json_response": json.dumps({"error": str(e)}, ensure_ascii=False)
        }
        for key in OUTPUT_FIELDS:
            out_row[key] = "ERROR"
        fieldnames = [ID_COLUMN, TEXT_COLUMN, "reasoning_content"] + OUTPUT_FIELDS + ["full_json_response"]
        safe_append_row(OUTPUT_CSV_FILE, fieldnames, out_row)
        return False


# ========================== Main ==========================

def main():
    print("=" * 60)
    print("Diseased Vessel Counting — Main Model (Optimized Prompt)")
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
