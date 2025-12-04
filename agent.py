import requests
from typing import Dict, Any, List, Optional, Tuple
import ast
import operator as op

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

