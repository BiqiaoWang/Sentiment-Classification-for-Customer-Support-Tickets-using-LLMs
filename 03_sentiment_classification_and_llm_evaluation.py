"""
This script:
1. Performs sentiment classification on customer support tickets using Llama 3.1 and Llama 3.3.
2. Evaluates both LLMs against a manually labeled gold-standard dataset.
3. Reports Precision, Recall, and F1-score for model comparison.
"""

import os
import time
import re
from typing import Optional

import pandas as pd
from tqdm import tqdm
from groq import Groq, APIStatusError


# Models (Groq)
LLAMA31_MODEL = "llama-3.1-8b-instant"
LLAMA33_MODEL = "llama-3.3-70b-versatile"


# Files
INPUT_CSV = "gold_set_labeled.csv"  # must include: ticket_id, ticket_description

OUT_WITH_LABELS = "gold_set_with_llm_labels.csv"

OUT_PRED_LLAMA31 = "llama31_predictions.csv"
OUT_PRED_LLAMA33 = "llama33_predictions.csv"

OUT_METRICS = "metrics_comparison.txt"

# Columns
ID_COL = "ticket_id"
TEXT_COL = "ticket_description"
GOLD_COL = "gold_label"  # gold-standard labels used for model evaluation

LLAMA31_COL = "llama31_8b_label"
LLAMA33_COL = "llama33_70b_label"


# Runtime params
MAX_API_RETRIES = 5
SAVE_EVERY = 10
SLEEP_SECONDS = 0.2
ALLOWED_LABELS = {"0", "1"}

# System prompt used for all LLM inference
SYSTEM_PROMPT = (
    "You are a sentiment classifier for customer support tickets.\n"
    "Return ONLY one character:\n"
    "1 = Negative\n"
    "0 = Non-Negative\n"
    "No other text."
)

# Helpers

def parse_wait_seconds_from_msg(msg: str, default: int = 60) -> int:
    # Handles strings like "2m10s" or "45s"
    m = re.search(r"(\d+)m(\d+)s", msg)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m2 = re.search(r"(\d+)s", msg)
    if m2:
        return int(m2.group(1))
    return default

# Build task-specific user prompt
def build_user_prompt(text: str) -> str:
    return (
        "You must output ONLY a single character.\n"
        "Output 1 if the sentiment is Negative.\n"
        "Output 0 if the sentiment is Non-Negative.\n"
        "Do NOT output words, labels, or explanations.\n\n"
        f"Ticket description:\n{text}"
    )

# Normalize LLM outputs to binary labels (0 or 1)
def normalize_binary_label(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s in ALLOWED_LABELS:
        return s
    # Sometimes model returns "Label: 1" / "1\n"
    m = re.search(r"\b([01])\b", s)
    if m:
        return m.group(1)
    return None

def is_valid_label(value) -> bool:
    if value is None or pd.isna(value):
        return False
    try:
        numeric = float(value)
        if numeric in (0.0, 1.0):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() in ALLOWED_LABELS

# Groq client + retry

def init_groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=key)

groq_client = init_groq_client()


def call_groq_with_retry(**kwargs):
    backoff = 10
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            return groq_client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            msg = str(exc)
            if "rate_limit_exceeded" in msg or "TPM" in msg or "TPD" in msg:
                wait_s = parse_wait_seconds_from_msg(msg, default=60)
                print(f"[Groq] rate limit (attempt {attempt}); waiting {wait_s}s")
                time.sleep(wait_s)
                backoff = min(backoff * 2, 600)
                continue
            print(f"[Groq] APIStatusError: {msg}")
            raise
        except Exception as exc:
            print(f"[Groq] call failed: {exc}; attempt {attempt}; waiting {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)
    raise RuntimeError("[Groq] retries exhausted.")


def classify_with_model(model_name: str, text: str) -> int:
    prompt = build_user_prompt(text)
    while True:
        resp = call_groq_with_retry(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=16,
            temperature=0,
            stop=["\n"]
        )
        raw = resp.choices[0].message.content
        label = normalize_binary_label(raw)
        if label is not None:
            return int(label)
        print(f"[Groq {model_name}] invalid label returned; retrying...")
        time.sleep(1)

# Main
def main():
    df = pd.read_csv(INPUT_CSV)

    # Validate required columns
    if ID_COL not in df.columns or TEXT_COL not in df.columns:
        raise RuntimeError(f"Input CSV must contain columns: {ID_COL}, {TEXT_COL}")

    # Ensure prediction columns exist and are nullable Int64
    for col in (LLAMA31_COL, LLAMA33_COL):
        if col not in df.columns:
            df[col] = pd.NA
        else:
            df[col] = df[col].apply(lambda v: pd.NA if not is_valid_label(v) else int(float(v)))
        df[col] = df[col].astype("Int64")

    total = len(df)
    save_counter = 0

    for idx, row in tqdm(df.iterrows(), total=total):
        text = row.get(TEXT_COL, "")
        text = "" if pd.isna(text) else str(text)

        updated = False

        # LLaMA 3.1 8B labels
        if not is_valid_label(row.get(LLAMA31_COL)):
            df.at[idx, LLAMA31_COL] = classify_with_model(LLAMA31_MODEL, text)
            updated = True

        # LLaMA 3.3 70B labels
        if not is_valid_label(row.get(LLAMA33_COL)):
            df.at[idx, LLAMA33_COL] = classify_with_model(LLAMA33_MODEL, text)
            updated = True

        if updated:
            save_counter += 1
            if save_counter % SAVE_EVERY == 0:
                df.to_csv(OUT_WITH_LABELS, index=False)
            time.sleep(SLEEP_SECONDS)

    # Final save
    df.to_csv(OUT_WITH_LABELS, index=False)
    print(f"Done. Saved labeled file: {OUT_WITH_LABELS}")

    # Save clean prediction files
    df[[ID_COL, LLAMA31_COL]].to_csv(OUT_PRED_LLAMA, index=False)
    df[[ID_COL, LLAMA33_COL]].to_csv(OUT_PRED_LLAMA33, index=False)
    print(f"Saved: {OUT_PRED_LLAMA}")
    print(f"Saved: {OUT_PRED_LLAMA33}")

    # Model evaluation using the gold-standard labels
    if GOLD_COL in df.columns:
        try:
            from sklearn.metrics import recall_score, precision_score, f1_score

            y_true = df[GOLD_COL].astype(int)

            def write_metrics(f, name: str, y_pred):
                f.write(f"{name}\n")
                f.write(f"  Recall (Negative):    {recall_score(y_true, y_pred, pos_label=1):.4f}\n")
                f.write(f"  Precision (Negative): {precision_score(y_true, y_pred, pos_label=1):.4f}\n")
                f.write(f"  F1 (Negative):        {f1_score(y_true, y_pred, pos_label=1):.4f}\n\n")


            # Evaluate metrics using Negative (label=1) as the positive class.
            # In scikit-learn, "positive class" refers to the target class for evaluation,
            # not positive sentiment.
            with open(OUT_METRICS, "w", encoding="utf-8") as f:
                f.write("Metrics (pos_label=1 => Negative)\n\n")
                write_metrics(f, "llama-3.1-8b-instant (Groq)", df[LLAMA31_COL].astype(int))
                write_metrics(f, "llama-3.3-70b-versatile (Groq)", df[LLAMA33_COL].astype(int))

            print(f"Saved: {OUT_METRICS}")
        except Exception as e:
            print(f"gold_label found but metrics failed: {e}")
    else:
        print("Note: gold_label not found; metrics file not generated.")


if __name__ == "__main__":
    main()
