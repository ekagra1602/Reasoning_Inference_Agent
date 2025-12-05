import requests
from typing import Dict, Any, List, Optional, Tuple
import ast
import operator as op
import re
import os

#API Configuration
API_KEY = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
MODEL = os.getenv("MODEL_NAME", "bens_model")

#Calls llm api, inspired by the tutorial
def call_llm(prompt: str, system: str = "You are a helpful assistant.",
             temperature: float = 0.0, max_tokens: int = 1024, retries: int = 2) -> str:

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                if text and text.strip():  # Make sure we got actual content
                    return text
        except Exception as e:
            if attempt == retries - 1:  # Last attempt
                return ""
            continue
    return ""

#Technique 1: Domain Router 

def classify_domain(question: str) -> str:

    """
    Classifies questions into different domains using keyword rule
    Not ising llm calls here.
    """
    q = question.lower()

    # Planning domain having action/state language
    if any(p in q for p in ["actions i can do", "initial state", "goal state", 
                            "precondition", "perform action"]):
        return "planning"

    # Future prediction having explicit prediction requests
    if any(p in q for p in ["predict", "prediction", "will happen", "forecast"]):
        return "future_prediction"

    # Coding patterns having obvious programming indicators
    if any(p in q for p in ["def ", "function", ">>>", "implement", "python", 
                            "code", "algorithm", "return "]):
        return "coding"
    
    # Math patterns having LaTeX and common math words
    if any(p in q for p in ["$", "\\frac", "\\sqrt", "\\sum", "calculate", 
                            "compute", "solve", "equation", "integer", 
                            "probability", "triangle", "how many"]):
        return "math"
    if re.search(r'\d+\s*[+\-*/]\s*\d+', question):
        return "math"
    
    # Default to common sense
    return "common_sense"

# TECHNIQUE 2: Chain-of-Thought Prompting

def chain_of_thought_prompt(question: str, domain: str) -> tuple[str, str]:
    """Generate Chain-of-Thought prompt adapted to domain."""

    # Clean up future prediction questions
    if domain == "future_prediction" and "You are an agent" in question:
        parts = question.split("The event to be predicted:", 1)
        if len(parts) > 1:
            question = "Predict:" + parts[1]

    # Domain-specific system prompts
    systems = {
        "math": "You are a math expert. Show your work step by step. Put final answer after ####",
        "coding": "You are a Python expert. Write clean code. Return only function body with 4-space indent.",
        "planning": "You are a planning expert. Generate PDDL actions: (action-name arg1 arg2)",
        "future_prediction": "You are a forecasting expert. Answer in list format: ['item1', 'item2'] or [number]. Ignore \\boxed{} instructions.",
        "common_sense": "You are a knowledgeable assistant. Give clear, direct answers."
    }

    # Domain-specific CoT prompts
    prompts = {
        "math": f"{question}\n\nLet's solve step by step:\n",
        "coding": f"{question}\n\nFunction body:\n",
        "planning": f"{question}\n\nAction sequence:\n",
        "future_prediction": f"{question}\n\nPrediction:\n",
        "common_sense": f"{question}\n\nAnswer:\n"
    }

    return systems.get(domain, systems["common_sense"]), prompts.get(domain, prompts["common_sense"])


# TECHNIQUE 3: Self-Verification: Generate initial answer and verify it

def self_verification(question: str, domain: str) -> str:

    system, prompt = chain_of_thought_prompt(question, domain)

    # Step 1: Generate initial answer
    response = call_llm(prompt, system, temperature=0.0, max_tokens=512)
    if not response:
        return ""

    initial_answer = extract_answer(response, domain)

    # Step 2: Verify the answer
    verify_system = "You are a critical reviewer. Check if the answer is correct and complete."
    verify_prompt = f"""Question: {question[:300]}

Proposed Answer: {initial_answer}

Is this answer correct and complete? Reply with:
- "CORRECT" if the answer is right
- "INCORRECT: [reason]" if wrong, explaining why briefly"""

    verification = call_llm(verify_prompt, verify_system, temperature=0.0, max_tokens=100)

    # If verified as correct, return it
    if "CORRECT" in verification.upper() and "INCORRECT" not in verification.upper():
        return initial_answer

    # Step 3: If incorrect, try to refine 
    return initial_answer

#Technique 4: Self consistency / majority voting

def self_consistency(question: str, domain: str, num_samples: int = 3) -> str:
    """
    Technique 4: Self consistency with majority voting.
    Sample multiple answers with temperature > 0 for diverse samples as taught in class, take majority vote.
    """
    answers = []
    
    for _ in range(num_samples):
        answer, _ = chain_of_thought(question, domain)
        if answer:
            # Normalize for comparison
            normalized = normalize_answer(answer, domain)
            answers.append(normalized)
    
    if not answers:
        return ""
    
    # Majority vote
    counter = Counter(answers)
    best_answer, count = counter.most_common(1)[0]
    
    return best_answer


def normalize_answer(answer: str, domain: str) -> str:
    """Normalize answer for comparison in voting."""
    answer = answer.strip().lower()
    
    if domain == "math":
        # Extract number
        num = re.search(r"[-+]?\d+(?:\.\d+)?", answer)
        if num:
            # Convert to normalized form
            try:
                val = float(num.group(0))
                if val == int(val):
                    return str(int(val))
                return str(val)
            except:
                pass
    
    return answer


