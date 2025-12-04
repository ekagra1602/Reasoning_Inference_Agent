import requests
from typing import Dict, Any, List, Optional, Tuple
import ast
import operator as op
import re

#API Configuration
API_KEY = "cse476"
API_BASE = "http://10.4.58.53:41701/v1"
MODEL = "bens_model"

def call_llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 60
) -> Dict[str, Any]:

    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """

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
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

#Tool: Safe Calculator

ALLOWED_BINOPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod
}
ALLOWED_UNOPS = {ast.UAdd: op.pos, ast.USub: op.neg}

def safe_eval(expr: str):
    """
    Safe arithmetic evaluator
    Supports: numbers, + - * / ** % parentheses, round(x, ndigits)
    """
    expr = expr.replace("^", "**")
    if len(expr) > 300:
        raise ValueError("Expression too long.")
    
    node = ast.parse(expr, mode="eval")
    
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.UnaryOp) and type(n.op) in ALLOWED_UNOPS:
            return ALLOWED_UNOPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in ALLOWED_BINOPS:
            return ALLOWED_BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "round":
            args = [ev(a) for a in n.args]
            return round(*args)
        raise ValueError(f"Disallowed: {ast.dump(n)}")
    
    return ev(node)

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
    
    # Default to common sense
    return "common_sense"

#Technique 2: Chain of thought prompting

COT_SYSTEM = """You are a careful reasoning assistant. 
Think through problems step by step before giving your final answer.
After your reasoning, provide your answer on a new line starting with "FINAL ANSWER:"
"""

COT_PROMPT = """Question: {question}

Let's think step by step, then provide the final answer.
End your response with:
FINAL ANSWER: <your answer>"""


def chain_of_thought(question: str, domain: str = "default") -> Tuple[str, str]:
    """
    Technique 2: Chain of Thought prompting.
    Returns (answer, reasoning) tuple.
    """
    prompt = COT_PROMPT.format(question=question)
    
    result = call_llm(prompt=prompt, system=COT_SYSTEM, temperature=0.0)
    
    if not result["ok"]:
        return "", ""
    
    response = result["text"] or ""
    answer = extract_final_answer(response, domain)
    
    return answer, response

#Helper function to extract the final answer from a CoT response.
def extract_final_answer(response: str, domain: str) -> str:
    #Find explicit FINAL ANSWER marker
    patterns = [
        r"FINAL ANSWER:\s*(.+?)(?:\n|$)",
        r"Final Answer:\s*(.+?)(?:\n|$)",
        r"The answer is[:\s]+(.+?)(?:\n|$)",
        r"Answer:\s*(.+?)(?:\n|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            return clean_answer(answer, domain)
    
    # Fallback: extract based on domain
    if domain == "math":
        boxed = re.search(r"\\boxed\{([^}]+)\}", response)
        if boxed:
            return boxed.group(1).strip()
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", response)
        if nums:
            return nums[-1]
    
    # Return last non-empty line
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    return lines[-1] if lines else ""


def clean_answer(answer: str, domain: str) -> str:
    """Clean up extracted answer."""
    answer = answer.strip().rstrip(".")
    
    # For math, extract just the number
    if domain == "math":
        boxed = re.search(r"\\boxed\{([^}]+)\}", answer)
        if boxed:
            return boxed.group(1).strip()
        num = re.search(r"[-+]?\d+(?:\.\d+)?", answer)
        if num:
            return num.group(0)
    
    return answer

#Technique 3: Tool-using agent loop 

TOOL_SYSTEM = """You are a math tool-using agent.
You may do exactly ONE of the following in your reply:
1) CALCULATE: <arithmetic expression>
   - use only numbers, + - * / **, parentheses, and round(x, ndigits)
   - example: CALCULATE: round((3*2.49)*1.07, 2)
2) FINAL: <answer>
Return ONE line with the directive and value. No other text."""

ACTION_RE = re.compile(r"^\s*(CALCULATE|FINAL)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


def parse_action(text: str) -> Tuple[str, str]:
    """Parse CALCULATE or FINAL action from LLM response."""
    m = ACTION_RE.match(text.strip())
    if not m:
        # Fallback: treat whole thing as final answer
        return "FINAL", text.strip()
    return m.group(1).upper(), m.group(2).strip()


def tool_agent(question: str, max_tool_uses: int = 3) -> str:
    """
    Technique 3: Tool-using agent loop (from Mini Lab 5).
    Uses calculator tool for arithmetic.
    """
    # First prompt
    first_prompt = f"""Question: {question}
If you need arithmetic to get the answer, reply as:
CALCULATE: <expression>
Otherwise reply:
FINAL: <answer>"""
    
    r1 = call_llm(prompt=first_prompt, system=TOOL_SYSTEM, temperature=0.0)
    if not r1["ok"]:
        return ""
    
    action, payload = parse_action(r1["text"])
    tool_uses = 0
    
    while action == "CALCULATE" and tool_uses < max_tool_uses:
        tool_uses += 1
        
        try:
            calc_value = safe_eval(payload)
        except Exception as e:
            calc_value = f"Error: {e}"
        
        # Second prompt with result
        second_prompt = f"""The calculation result is: {calc_value}
Now provide the final answer.
Reply exactly as: FINAL: <answer>"""
        
        rN = call_llm(prompt=second_prompt, system=TOOL_SYSTEM, temperature=0.0)
        if not rN["ok"]:
            return str(calc_value)
        
        action, payload = parse_action(rN["text"])
    
    return payload



