from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")

TEST_LIMIT = None

# Import the agent
from agent import solve

def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data

def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answers = []
    for idx, question in enumerate(questions, start=1):
        # Get the question text
        question_text = question.get("input", "")
        #present in dev but not in test
        domain_hint = question.get("domain")  
        
        # Run the agent to get the answer
        try:
            real_answer = solve(question_text, domain_hint)
        except Exception as e:
            print(f"[{idx}] Error: {e}")
            real_answer = ""
        
        # String answer and check limits
        real_answer = str(real_answer or "").strip()
        if len(real_answer) >= 5000:
            real_answer = real_answer[:4999]
        
        answers.append({"output": real_answer})
        
        # Progress 
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(questions)} questions")
    return answers

def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )

def main() -> None:
    questions = load_questions(INPUT_PATH)

    if TEST_LIMIT is not None:
        questions = questions[:TEST_LIMIT]
        print(f"Testing: Processing first {len(questions)} questions only")

    print(f"Loaded {len(questions)} questions from {INPUT_PATH}")
    
    answers = build_answers(questions)

    with OUTPUT_PATH.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    with OUTPUT_PATH.open("r") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )

if __name__ == "__main__":
    main()