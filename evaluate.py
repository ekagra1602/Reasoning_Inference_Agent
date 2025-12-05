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