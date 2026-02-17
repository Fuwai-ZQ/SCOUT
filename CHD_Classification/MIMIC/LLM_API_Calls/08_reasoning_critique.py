# -*- coding: utf-8 -*-
###############################################################################
# 08_reasoning_critique.py
# Strategy 3 (S3): Reasoning critique — Mcheck audits Mmain's CoT trace
# Mcheck = Qwen3-32B (external checker model)
# Reference: "deploys a checker model Mcheck to audit the chain-of-thought (CoT)
#   trace produced by Mmain. The checker performs a binary pass/fail judgment
#   targeting logical fallacies in the rationale."
###############################################################################

import os
import json
import time
import csv
import re
import pandas as pd
from collections import deque
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

# =============================================================================
# Configuration
# =============================================================================

API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-XXX")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-32b"  # Mcheck

INPUT_CSV = "primary_model_predictions.csv"   # Mmain output (contains rationale + evidence)
OUTPUT_CSV = "chain_of_thought_audit.csv"

SAVE_EVERY_N = 1
CALLS_PER_MINUTE = 10
WINDOW_SECONDS = 60
MAX_RETRIES = 3

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# =============================================================================
# Audit prompt — binary pass/fail on logical consistency
# =============================================================================

AUDIT_SYSTEM_PROMPT = """
You are a medical data logic auditor. Your task is to check for factual contradictions or logical errors between the provided "Rationale" and "Evidence JSON".

Focus on the following checks:
1. **Time Calculation Contradictions**: e.g., Rationale claims "course > 30 days", but Evidence shows "Admission Date - Event Date = 5 days".
2. **Data Consistency**: e.g., Rationale claims "Tn elevated", but Evidence shows Tn as "Normal" or "Not elevated".
3. **Diagnostic Logic Conflicts**: e.g., Rationale claims "no ischemic symptoms", but Evidence records "chest pain".

Please output a JSON object strictly. Do not include Markdown formatting.
Format:
{
  "has_contradiction": "Yes" or "No",
  "analysis": "Briefly describe the contradiction found. If no contradiction, leave empty or write 'Consistent'."
}
"""


# =============================================================================
# Utilities
# =============================================================================

class RateLimiter:
    """Sliding-window rate limiter (single-threaded)."""

    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.ts = deque()

    def wait(self):
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


def parse_json_content(content: str):
    """Extract JSON from model output, handling markdown fences."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z0-9]*\n", "", content).strip()
        if content.endswith("```"):
            content = content[:-3].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except Exception:
                pass
    return None


def call_audit_model(rationale: str, evidence: str, limiter: RateLimiter):
    """Call Mcheck to audit Mmain's reasoning for logical contradictions."""
    user_content = f"Rationale: {rationale}\n\nEvidence JSON: {evidence}"
    messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    wait_s = 2.0
    for _ in range(MAX_RETRIES):
        limiter.wait()
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return parse_json_content(completion.choices[0].message.content)
        except (RateLimitError, APIConnectionError, APIStatusError) as e:
            print(f"  [API Retry: {e}]")
            time.sleep(wait_s)
            wait_s *= 2
        except Exception as e:
            print(f"  [Error: {e}]")
            return None

    print("  Max retries reached.")
    return None


def append_rows_to_csv(path: str, fieldnames: list, rows: list):
    """Batch-append rows to CSV."""
    if not rows:
        return
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a" if exists else "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Input file not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    required_cols = ["rationale_short", "evidence_json"]
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns: {required_cols}")
        return

    # Resume from checkpoint
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            done = pd.read_csv(OUTPUT_CSV)
            if "hadm_id" in done.columns:
                processed_ids = set(done["hadm_id"].astype(str).str.strip())
            print(f"[Resume] {len(processed_ids)} already audited.")
        except Exception:
            pass

    # Output columns = original columns + audit results
    fieldnames = list(df.columns)
    for col in ["audit_has_contradiction", "audit_analysis"]:
        if col not in fieldnames:
            fieldnames.append(col)

    limiter = RateLimiter(CALLS_PER_MINUTE, WINDOW_SECONDS)
    buffer = []

    for idx, row in df.iterrows():
        hadm_id = str(row.get("hadm_id", "")).strip()
        if not hadm_id or hadm_id in processed_ids:
            continue

        rationale = str(row.get("rationale_short", ""))
        evidence = str(row.get("evidence_json", ""))
        if len(rationale) < 5 or len(evidence) < 5:
            continue

        print(f"  Auditing [{idx + 1}] {hadm_id}")
        audit_res = call_audit_model(rationale, evidence, limiter)

        out = row.to_dict()
        if audit_res:
            out["audit_has_contradiction"] = audit_res.get("has_contradiction", "Unknown")
            out["audit_analysis"] = audit_res.get("analysis", "")
        else:
            out["audit_has_contradiction"] = "Error"
            out["audit_analysis"] = "API call failed"

        buffer.append(out)
        if len(buffer) >= SAVE_EVERY_N:
            append_rows_to_csv(OUTPUT_CSV, fieldnames, buffer)
            buffer.clear()

    if buffer:
        append_rows_to_csv(OUTPUT_CSV, fieldnames, buffer)

    print(f"\nAudit complete. Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
