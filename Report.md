# Project Report - Ekagra Gupta

##Github Repo Link - https://github.com/ekagra1602/Reasoning_Inference_Agent

## Details 
I built a reasoning agent that solves problems across 5 domains (math, coding, planning, common sense, future prediction) using 4 inference-time techniques. The agent achieved ~45% accuracy on the dev set with LLM-based evaluation, using only 1 API call per question
## Techniques Implemented

### 1. Domain Classification (`agent.py:45-71`)
The agent detects what problem it's dealing with by looking for keywords or different patterns. For example, if it sees latex math symbols or arithmetic operations, it classifies it as math, and if it sees coding syntax, it classifies it as coding.

### 2. Domain-Adaptive Prompting  (`agent.py:75-102`)
One the domain is known from the first function, we can use custom prompts for each type. For example math type is told to reason step by step and only give the final number answer at the end.

### 3. Chain-of-Thought Prompting (integrated in technique 2)
Instead of asking direct answers, we tell the model to show work, then on a new line write only the number for math or write function body. This makes it reason through the problem.

### 4. Self-Verification (`agent.py:107-135`)
Two step process where model generates and answer then a second LLM call verifies it. 

## Other things I tried that didn't help or work

**Self-Consistency / Majority Voting**: I implemented this by sampling 5 answers at temperature = 0.7 and taking the most common response. It actually made things worse, dropped from 30% to 10% accuracy. I think the randomness introduced more errors than it fixed along with 5x api call.

**Calculator Tool (similar to Mini Lab 5)**: I tried integrating the safe arithmetic evaluator to verify calculations in math problems. The tool worked great for simple arithmetic like 6*(15+39), but all the dataset has reasoning and all so it didnt work and dropped accuracy.

## Results
| Domain | Accuracy (LLM Judge) |

| Math | 55% |

| Coding | 75% |

| Common Sense | 70% |

| Planning | 15% |

| Future Prediction | 15% |

| **Overall** | **~45%** |

The LLM judge helps a lot for coding and common sense because it understands semantically equivalent answers (e.g., "False" vs "No" or different variable names in code). Math stays the same since numbers are objective.

## How to Run
```bash
# Single question
python agent.py "What is the capital of Italy"

# Evaluate on dev set
python evaluate.py --sample 100 --mix --llm-judge

# Fill the project answers from test set 
#Also has a test limit defaulted to None, but if we need to fill till specific questions we can set that limit
#The test data and project answer file needs to be in this directory
python generate_answers.py

```