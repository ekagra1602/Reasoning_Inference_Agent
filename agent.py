import requests
from typing import Dict, Any, List, Optional, Tuple
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
        "math": "You are a math expert. Show your reasoning, think step by step for everything, then on a NEW LINE write ONLY the final number. No words on that last line, just the number.",
        "coding": "You are a Python expert. Write the function body with 4-space indent. Put ONLY the code, no explanations.",
        "planning": "You are a planning expert. Think through the problem, then on NEW LINES put ONLY the action sequence in format: (action arg1 arg2)",
        "future_prediction": "You are a forecasting expert. Think it through, then on a NEW LINE put ONLY the prediction in format ['item'] or [number]. No extra words on that line.",
        "common_sense": "You are a knowledgeable assistant. Think through it, then on a NEW LINE put ONLY the direct answer. No 'Answer:' - just the answer itself on the last line."
    }

    # Domain-specific CoT prompts
    prompts = {
        "math": f"{question}\n\nShow your work, then on a NEW LINE write only the number:\n",
        "coding": f"{question}\n\nWrite the function body (4-space indent, no 'def' line):\n",
        "planning": f"{question}\n\nThink through the plan, then on NEW LINES list only the actions:\n",
        "future_prediction": f"{question}\n\nThink it through, then on a NEW LINE give only the prediction:\n",
        "common_sense": f"{question}\n\nThink about it, then on a NEW LINE give ONLY the answer:\n"
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

#Technique 4: Format and extract answer properly

def extract_answer(response: str, domain: str) -> str:
    if not response or not response.strip():
        return ""

    response = response.strip()

    # Remove markdown formatting
    response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
    response = re.sub(r'\$\$([^$]+)\$\$', r'\1', response)
    response = re.sub(r'\$([^$]+)\$', r'\1', response)

    # Take last non-empty line
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if lines:
        return lines[-1].rstrip('.,;:')

    return response


# MAIN SOLVER

def solve(question: str, domain: Optional[str] = None, use_self_consistency: bool = False) -> str:
    # Truncate very long inputs
    if len(question) > 20000:
        question = question[:20000] + "..."

    # Technique 1: Domain Classification
    if domain is None:
        domain = classify_domain(question)

    # Technique 3: Self-Verification
    if use_self_consistency: 
        return self_verification(question, domain)

    # Techniques 2 & 3: Domain-Adaptive Prompting + Chain-of-Thought
    system, prompt = chain_of_thought_prompt(question, domain)
    response = call_llm(prompt, system, temperature=0.0,
     max_tokens=1024)

    if not response:
        return ""

    return extract_answer(response, domain)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"Domain: {classify_domain(q)}")
        print(f"Answer: {solve(q)}")
    else:
        print(f"Answer: {solve('What is 33 + 33?')}")
