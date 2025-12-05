import json
import argparse
import re
import random
from pathlib import Path
from collections import defaultdict

from agent import solve, call_llm

DEV_DATA_PATH = Path("cse476_final_project_dev_data.json")

def load_data(path: Path):
    with path.open("r") as f:
        return json.load(f)

def extract_answer(text, domain: str) -> str:
    """Extract the actual answer from text."""
    if not isinstance(text, str):
        return str(text)

    text = text.strip()

    # For math: check for #### format
    if "####" in text:
        match = re.search(r'####\s*(\d+)', text)
        if match:
            return match.group(1)

    return text


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = str(text or "").strip().lower()

    # Normalize boolean values
    if text in ["yes", "true", "1"]:
        return "true"
    if text in ["no", "false", "0"]:
        return "false"

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove common punctuation at the end
    text = text.rstrip('.,;:!?')

    return text

#Usee llm as judge to evaluate the prediction
def llm_judge(question: str, prediction: str, expected: str, domain: str) -> bool:

    system = """You are a strict grader evaluating answers to questions.
Your job is to determine if the prediction is correct given the expected answer.

STRICT GRADING RULES:
- For math: Numbers must match exactly (allow small rounding like 42.0 vs 42)
- For True/False: Must be semantically equivalent (Yes=True, No=False is fine)
- For factual answers: Must be the same entity (synonyms fine, related but different is wrong)
- For code: Must have equivalent functionality (different variable names fine, different logic is wrong)
- For lists: Order matters unless it's clearly unordered. Must contain the same items.

IMPORTANT: Be strict. If unsure, mark as incorrect.

Reply with ONLY one word: "correct" or "incorrect"
"""

    prompt = f"""Domain: {domain}

Question: {question[:300]}

Expected Answer: {expected}

Prediction: {prediction}

Is the prediction correct? Reply with ONLY "correct" or "incorrect"."""

    result_text = call_llm(prompt, system, temperature=0.0, max_tokens=512)

    if not result_text:
        return False

    response = result_text.strip().upper()

    # Only accept if response starts with "correct"
    return response.startswith("correct")   


def is_correct(pred, gold, domain: str, use_llm_judge: bool = False, question: str = "") -> bool:
    """Check if prediction matches expected answer."""
    pred = str(pred or "").strip()
    gold = str(gold or "").strip()

    # Extract answers
    gold_answer = extract_answer(gold, domain)
    pred_answer = extract_answer(pred, domain)

    # Normalize for comparison
    pred_norm = normalize_answer(pred_answer)
    gold_norm = normalize_answer(gold_answer)

    # Exact match (case insensitive, normalized)
    if pred_norm == gold_norm:
        return True

    # Numeric comparison for math and numbers
    try:
        pred_nums = re.findall(r'[-+]?\d+\.?\d*', pred_answer)
        gold_nums = re.findall(r'[-+]?\d+\.?\d*', gold_answer)

        if pred_nums and gold_nums:
            pred_num = float(pred_nums[-1])
            gold_num = float(gold_nums[-1])
            if abs(pred_num - gold_num) < 0.01:
                return True
    except:
        pass

    # For code, check if the essential code is the same
    if domain == "coding":
        pred_code = re.sub(r'\s+', '', pred_answer)
        gold_code = re.sub(r'\s+', '', gold_answer)
        if pred_code == gold_code:
            return True
        if gold_code in pred_code and len(gold_code) > 10:
            return True

    # Substring containment
    if len(gold_norm) > 5:
        if gold_norm in pred_norm:
            return True
        if pred_norm in gold_norm and len(pred_norm) > 5:
            return True

    # If all string matching fails and LLM judge is enabled, use it as fallback
    if use_llm_judge and question:
        return llm_judge(question, pred, gold, domain)

    return False

def get_mixed_sample(data, n):
    """Get balanced sample across domains."""
    by_domain = defaultdict(list)
    for item in data:
        by_domain[item.get("domain", "common_sense")].append(item)

    domains = list(by_domain.keys())
    per_domain = max(1, n // len(domains))

    random.seed(42)
    sample = []
    for domain in domains:
        items = by_domain[domain]
        sample.extend(random.sample(items, min(per_domain, len(items))))

    random.shuffle(sample)
    return sample[:n]