# -*- coding: utf-8 -*-
"""Chain-of-thought audit via DashScope API (thinking disabled).
Checks rationale–evidence consistency with checkpoint resume."""
import os
import json
import time
import csv
import re
import pandas as pd
from collections import deque
from openai import OpenAI

# --- Configuration ---

# DashScope API config
API_KEY = os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-32b"  

# Input: CSV from classification step
INPUT_CSV_FILE = "cases_labeled_Qwen235b_withPrompt_withThinking.csv"
# Output: audit results
OUTPUT_CHECK_FILE = "cases_checked_results_Qwen235b_withPrompt_withThinking_using_Qwen32b_noThinking_aliyun.csv"

# Rate limiting
SAVE_EVERY_N = 5
CALLS_PER_MINUTE = 60  
WINDOW_SECONDS = 65

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- Audit Prompt ---

audit_system_prompt = """
你是一名医疗数据逻辑审计员。你的任务是检查"简述理由（Rationale）"与"证据详情（Evidence JSON）"之间是否存在事实矛盾或逻辑错误。

检查重点：
1. **时间计算矛盾**：例如 Rationale 说"病程大于30天"，但 Evidence 中显示"入院日-事件日 = 5天"。
2. **数据一致性**：例如 Rationale 说"Tn升高"，但 Evidence 中 Tn 为"正常"或"未升高"。
3. **诊断逻辑冲突**：例如 Rationale 说"无缺血症状"，但 Evidence 中记录了"胸痛"。

请输出 JSON 格式，不要包含 Markdown 标记：
{
  "has_contradiction": "present" 或 "不存在",
  "analysis": "简述发现的矛盾点，如果没有矛盾则留空或写'一致'"
}
"""

# --- Utilities ---

class RateLimiter:
    """Sliding-window rate limiter."""

    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.ts = deque()

    def wait(self):
        now = time.time()
        while self.ts and (now - self.ts[0]) >= self.window_seconds:
            self.ts.popleft()
        if len(self.ts) >= self.max_calls:
            sleep_s = self.window_seconds - (now - self.ts[0]) + 0.05
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

def call_audit_model(rationale, evidence_json, limiter):
    """Call DashScope API for rationale–evidence audit."""
    
    user_content = f"""
请检查以下两条信息的一致性：

【Rationale (简述)】
{rationale}

【Evidence (证据详情)】
{evidence_json}
"""
    
    messages = [
        {"role": "system", "content": audit_system_prompt},
        {"role": "user", "content": user_content},
    ]

    MAX_RETRIES = 5
    wait_seconds = 5

    for attempt in range(MAX_RETRIES):
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
                max_tokens=512,
            )

            full_answer = ""

            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if hasattr(delta, 'content') and delta.content:
                    full_answer += delta.content

            json_data = parse_json_from_content(full_answer)

            if json_data is None:
                return {
                    "has_contradiction": "parse_error",
                    "analysis": f"JSON parse failed, raw: {full_answer[:200]}"
                }

            return json_data

        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait_seconds)
                wait_seconds *= 2
                continue
            else:
                return {"has_contradiction": "error", "analysis": str(e)}

    return {"has_contradiction": "timeout", "analysis": "Max retries exceeded"}

def append_rows_to_csv(path, fieldnames, rows):
    file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
    mode = "a" if file_exists else "w"
    with open(path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

# --- Main ---

def main():
    print("=" * 70)
    print("CoT Audit - DashScope API (No Thinking)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"API: DashScope")
    print(f"Input: {INPUT_CSV_FILE}")
    print(f"Output: {OUTPUT_CHECK_FILE}")
    print("=" * 70)

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"ERROR: Input file not found: {INPUT_CSV_FILE}")
        return

    df = pd.read_csv(INPUT_CSV_FILE)
    print(f"Loaded {len(df)} labeled records for audit...")

    # Checkpoint resume
    processed_ids = set()
    if os.path.exists(OUTPUT_CHECK_FILE):
        try:
            df_done = pd.read_csv(OUTPUT_CHECK_FILE)
            processed_ids = set(df_done["case_id"].astype(str).str.strip())
            print(f"[Resume] Skipping {len(processed_ids)} completed")
        except:
            pass

    fieldnames = ["case_id", "final_label", "rationale_short", "has_contradiction", "analysis"]
    buffer = []

    limiter = RateLimiter(CALLS_PER_MINUTE, WINDOW_SECONDS)

    success_count = 0
    fail_count = 0
    start_time = time.time()

    for idx, row in df.iterrows():
        case_id = str(row.get("case_id", "")).strip()

        if case_id in processed_ids:
            continue

        rationale = row.get("rationale_short", "")
        evidence = row.get("evidence_json", "")

        print(f"Checking: {case_id} ...")

        
        result = call_audit_model(rationale, evidence, limiter)

        has_contradiction = result.get("has_contradiction", "unknown")
        analysis = result.get("analysis", "")

        out_row = {
            "case_id": case_id,
            "final_label": row.get("final_label"),
            "rationale_short": rationale,
            "has_contradiction": has_contradiction,
            "analysis": analysis
        }

        buffer.append(out_row)
        success_count += 1

        status = "CONFLICT" if has_contradiction == "present" else "OK"
        print(f"  {status} | {case_id} | {analysis[:50]}...")

        if len(buffer) >= SAVE_EVERY_N:
            append_rows_to_csv(OUTPUT_CHECK_FILE, fieldnames, buffer)
            print(f"Saved {len(buffer)} rows")
            buffer = []

    if buffer:
        append_rows_to_csv(OUTPUT_CHECK_FILE, fieldnames, buffer)
        print(f"Saved {len(buffer)} rows")

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 70)
    print("Audit complete!")
    print(f"   Processed: {success_count}")
    print(f"   Elapsed:   {elapsed_total / 60:.1f} min")
    print(f"   Output:    {OUTPUT_CHECK_FILE}")
    print("=" * 70)

    if os.path.exists(OUTPUT_CHECK_FILE):
        try:
            df_result = pd.read_csv(OUTPUT_CHECK_FILE)
            print("\nAudit result distribution:")
            print("-" * 40)
            print(df_result['has_contradiction'].value_counts().to_string())
        except Exception as e:
            print(f"Stats error: {e}")

if __name__ == "__main__":
    main()
