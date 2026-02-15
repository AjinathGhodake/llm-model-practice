"""
Phase 5, Lesson 3: Full ReAct Agent from Scratch
=================================================

CONCEPT FIRST
-------------
We now have:
  - The loop mechanics (Lesson 1)
  - Proper tool definitions and dispatch (Lesson 2)

Let's put them together into a complete, working ReAct agent.

This agent can answer "multi-hop" questions — questions that require
chaining multiple tool calls together.

Example multi-hop question:
  "What is the square root of the number of days in a year?"

Steps the model must take:
  1. Know (or look up) that a year has 365 days
  2. Compute sqrt(365)
  3. Report the result

This requires REASONING (figuring out the steps) + TOOL USE (doing the math).

THE KNOWLEDGE BASE
------------------
We add a "search" tool backed by a small local knowledge base.
In production this would be a vector database or web search.
Here it's a dict — same loop mechanics, much simpler setup.

LOOP ARCHITECTURE
-----------------

    ┌──────────────────────────────────────────┐
    │           REACT AGENT LOOP               │
    │                                          │
    │  1. Build prompt (question + tools)      │
    │  2. Generate text from model             │
    │  3. Parse output:                        │
    │     a. Contains "Final Answer:"? → DONE  │
    │     b. Contains "Action:"? → run tool   │
    │     c. Neither? → model went off rails   │
    │  4. Append Observation to context        │
    │  5. Go to step 2                         │
    │                                          │
    │  Safety: stop at max_steps               │
    └──────────────────────────────────────────┘

HOW TO RUN
----------
    uv run phase5_agents/03_react_agent.py
"""

import re
import math
import json
from datetime import datetime
from typing import Any


# ============================================================
# STEP 1: Build the knowledge base
# ============================================================
# A real agent would search Wikipedia, a vector DB, or the web.
# We use a simple dict — the tool mechanics are identical.

KNOWLEDGE_BASE = {
    "year": "A calendar year has 365 days (366 in a leap year). The Earth takes 365.25 days to orbit the sun.",
    "days in a year": "A standard year has 365 days. A leap year has 366 days.",
    "month": "A month is a unit of time. Most months have 30 or 31 days. February has 28 days (29 in a leap year).",
    "pi": "Pi (π) is approximately 3.14159265. It is the ratio of a circle's circumference to its diameter.",
    "speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 3 × 10^8 m/s).",
    "fibonacci": "The Fibonacci sequence starts 0, 1, 1, 2, 3, 5, 8, 13, 21, 34... Each number is the sum of the two before it.",
    "prime numbers": "Prime numbers are divisible only by 1 and themselves: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29...",
    "alphabet": "The English alphabet has 26 letters: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z.",
    "planets": "The solar system has 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
    "water": "Water (H₂O) boils at 100°C (212°F) at sea level. It freezes at 0°C (32°F).",
    "square numbers": "Perfect squares: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225...",
    "chess": "A standard chess board has 64 squares (8 × 8). A chess game has 32 pieces (16 per player).",
    "hours in a day": "A day has 24 hours, 1440 minutes, or 86400 seconds.",
    "weeks in a year": "A year has 52 weeks and 1 day (52.18 weeks on average).",
}


def search_knowledge_base(query: str) -> str:
    """
    Search the local knowledge base for information about a topic.
    Returns the most relevant entry, or a 'not found' message.
    """
    query_lower = query.lower().strip()

    # Exact match first
    if query_lower in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[query_lower]

    # Partial match — find entries where the key appears in the query
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower or query_lower in key:
            return value

    # Word overlap match — find entries sharing words with the query
    query_words = set(query_lower.split())
    best_match = None
    best_overlap = 0
    for key, value in KNOWLEDGE_BASE.items():
        key_words = set(key.split())
        overlap = len(query_words & key_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = value

    if best_match and best_overlap > 0:
        return best_match

    return f"No information found for '{query}' in the knowledge base."


def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports all math module functions."""
    try:
        safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_globals["__builtins__"] = {}
        result = eval(expression, safe_globals, {})
        return str(round(float(result), 6))
    except Exception as e:
        return f"Error: {e}"


def get_current_date(fmt: str = "%Y-%m-%d") -> str:
    """Return the current date."""
    return datetime.now().strftime(fmt)


# ============================================================
# STEP 2: Tool registry (from Lesson 2, simplified)
# ============================================================

TOOLS = {
    "search": search_knowledge_base,
    "calculator": calculator,
    "get_current_date": get_current_date,
}

TOOL_DESCRIPTIONS = {
    "search": "search(query) — Look up factual information. Use for: number of days in a year, constants, facts.",
    "calculator": "calculator(expression) — Evaluate math. Use sqrt(), **, +, -, *, /. Example: calculator(sqrt(365))",
    "get_current_date": "get_current_date() — Return today's date.",
}


def dispatch(tool_name: str, args: str) -> str:
    """Run a tool by name with the given argument string."""
    if tool_name not in TOOLS:
        return f"Unknown tool: {tool_name}. Available: {list(TOOLS.keys())}"
    return TOOLS[tool_name](args) if args else TOOLS[tool_name]()


# ============================================================
# STEP 3: The ReAct prompt template
# ============================================================

def build_react_prompt(question: str) -> str:
    """
    Build the initial prompt for the ReAct loop.

    We include:
    1. Available tools (so model knows what it can call)
    2. The format to follow (Thought/Action/Observation)
    3. Two few-shot examples (so model learns the pattern)
    4. The actual question
    """
    tool_list = "\n".join(f"  - {desc}" for desc in TOOL_DESCRIPTIONS.values())

    return f"""You are a reasoning assistant that uses tools to answer questions step by step.

AVAILABLE TOOLS:
{tool_list}

FORMAT:
Thought: [your reasoning about what to do next]
Action: tool_name(argument)
Observation: [result — filled in by the system]
... (repeat as needed)
Final Answer: [your complete answer]

RULES:
- Always write a Thought before an Action
- Only call one tool per step
- Use the Observation before deciding the next step
- Write "Final Answer:" when you have enough information

EXAMPLE 1:
Question: How many letters are in the English alphabet, and what is that number squared?
Thought: I need to find how many letters are in the alphabet, then square that number.
Action: search(alphabet)
Observation: The English alphabet has 26 letters: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z.
Thought: The alphabet has 26 letters. Now I'll compute 26 squared.
Action: calculator(26 ** 2)
Observation: 676.0
Thought: 26 squared is 676.
Final Answer: The English alphabet has 26 letters, and 26 squared is 676.

EXAMPLE 2:
Question: What is the cube root of the number of squares on a chess board?
Thought: I need to find how many squares a chess board has, then take the cube root.
Action: search(chess)
Observation: A standard chess board has 64 squares (8 × 8). A chess game has 32 pieces (16 per player).
Thought: A chess board has 64 squares. Now I'll compute the cube root: 64 ** (1/3).
Action: calculator(64 ** (1/3))
Observation: 4.0
Thought: The cube root of 64 is 4.
Final Answer: A chess board has 64 squares, and the cube root of 64 is 4.

NOW ANSWER THIS:
Question: {question}
"""


# ============================================================
# STEP 4: Parse model output
# ============================================================

def parse_output(text: str):
    """
    Parse model output to extract either:
    - An action: returns ("action", tool_name, args)
    - A final answer: returns ("final", answer_text)
    - Unknown: returns ("unknown", raw_text)
    """
    # Check for Final Answer first
    final_match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", text, re.DOTALL)
    if final_match:
        return ("final", final_match.group(1).strip())

    # Check for Action: tool_name(args)
    action_match = re.search(r"Action:\s*(\w+)\(([^)]*)\)", text)
    if action_match:
        tool_name = action_match.group(1).strip()
        args = action_match.group(2).strip()
        # Strip surrounding quotes
        args = args.strip('"\'')
        return ("action", tool_name, args)

    return ("unknown", text.strip())


# ============================================================
# STEP 5: The complete ReAct agent
# ============================================================

class ReActAgent:
    """
    A complete ReAct agent.

    Initialize with a language model function (or a simulated one).
    Call .run(question) to answer a question using the tool loop.
    """

    def __init__(self, model_fn, max_steps: int = 8, verbose: bool = True):
        """
        model_fn: function(prompt: str) -> str
            Takes the full conversation so far and returns the next completion.
        max_steps: safety limit on loop iterations
        verbose: print each step for learning
        """
        self.model_fn = model_fn
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, question: str) -> str:
        """
        Run the ReAct loop to answer a question.
        Returns the final answer string.
        """
        # Build the initial prompt
        prompt = build_react_prompt(question)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"QUESTION: {question}")
            print(f"{'='*60}")

        for step in range(self.max_steps):
            # Generate the next piece of text from the model
            new_text = self.model_fn(prompt)

            if self.verbose:
                print(f"\n[Step {step + 1}] Model output:")
                print(f"  {new_text.strip()}")

            # Parse what the model said
            parsed = parse_output(new_text)

            if parsed[0] == "final":
                answer = parsed[1]
                if self.verbose:
                    print(f"\n✓ Final Answer: {answer}")
                return answer

            elif parsed[0] == "action":
                _, tool_name, args = parsed
                if self.verbose:
                    print(f"\n[Tool] {tool_name}({args!r})")

                # Run the tool
                observation = dispatch(tool_name, args)
                if self.verbose:
                    print(f"[Observation] {observation}")

                # Append BOTH the model's output AND the observation to the prompt
                prompt += new_text
                if not new_text.endswith("\n"):
                    prompt += "\n"
                prompt += f"Observation: {observation}\n"

            else:
                # Model went off-script — neither action nor final answer
                if self.verbose:
                    print(f"\n⚠ Unexpected output (no action or final answer)")
                    print(f"  Adding to context and trying again...")
                prompt += new_text
                if not new_text.endswith("\n"):
                    prompt += "\n"

        if self.verbose:
            print(f"\n⚠ Reached max_steps={self.max_steps}")
        return "Could not determine answer within step limit."


# ============================================================
# STEP 6: Simulate a "perfect" model for demonstration
# ============================================================
# Since GPT-2 won't follow ReAct format, we script ideal model outputs.
# This lets us demonstrate the COMPLETE loop working correctly.

def make_scripted_model(scripted_outputs: list[str]):
    """
    Return a model function that serves pre-written outputs in order.

    step 0 → scripted_outputs[0]
    step 1 → scripted_outputs[1]
    etc.
    """
    step = [0]  # Use a list so the closure can mutate it

    def model_fn(prompt: str) -> str:
        idx = step[0]
        step[0] += 1
        if idx < len(scripted_outputs):
            return scripted_outputs[idx]
        return "Final Answer: I have run out of scripted responses."

    return model_fn


# ============================================================
# STEP 7: Attempt real GPT-2 inference
# ============================================================

def make_gpt2_model():
    """
    Try to load GPT-2 and return a model function.
    Returns None if transformers is not available.
    """
    try:
        from transformers import pipeline

        print("Loading GPT-2...")
        generator = pipeline(
            "text-generation",
            model="gpt2",
            device=-1,
            pad_token_id=50256,
        )
        print("GPT-2 loaded.")

        def model_fn(prompt: str) -> str:
            # Generate up to 100 new tokens
            result = generator(
                prompt,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
                pad_token_id=50256,
            )
            full = result[0]["generated_text"]
            return full[len(prompt):]  # Just the new part

        return model_fn
    except Exception as e:
        print(f"Could not load GPT-2: {e}")
        return None


# ============================================================
# DEMO QUESTIONS AND SCRIPTED ANSWERS
# ============================================================

DEMOS = [
    {
        "question": "What is the square root of the number of days in a year?",
        "steps": [
            # Step 1: search for the number of days
            "Thought: I need to find the number of days in a year, then take the square root.\nAction: search(days in a year)\n",
            # Step 2: compute sqrt(365)
            "Thought: A standard year has 365 days. Now I'll compute sqrt(365).\nAction: calculator(sqrt(365))\n",
            # Step 3: final answer
            "Thought: The square root of 365 is approximately 19.105. I have the answer.\nFinal Answer: The number of days in a year is 365, and its square root is approximately 19.105.\n",
        ]
    },
    {
        "question": "How many hours are in a week? Show your calculation.",
        "steps": [
            "Thought: I need to know hours in a day and days in a week.\nAction: search(hours in a day)\n",
            "Thought: A day has 24 hours. A week has 7 days. So hours in a week = 24 * 7.\nAction: calculator(24 * 7)\n",
            "Thought: 24 * 7 = 168. A week has 168 hours.\nFinal Answer: A week has 168 hours (24 hours/day × 7 days/week = 168 hours).\n",
        ]
    },
    {
        "question": "What is 15% of the number of squares on a chess board?",
        "steps": [
            "Thought: First I'll look up how many squares are on a chess board.\nAction: search(chess)\n",
            "Thought: A chess board has 64 squares. Now I'll compute 15% of 64.\nAction: calculator(64 * 0.15)\n",
            "Thought: 15% of 64 is 9.6.\nFinal Answer: A chess board has 64 squares, and 15% of that is 9.6.\n",
        ]
    },
]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(__doc__)

    # --- Try real GPT-2 first ---
    print("\n" + "#"*60)
    print("# Attempting GPT-2 on a ReAct prompt (expect format issues)")
    print("#"*60)

    gpt2_model = make_gpt2_model()
    if gpt2_model:
        agent = ReActAgent(gpt2_model, max_steps=3, verbose=True)
        agent.run("What is the square root of the number of days in a year?")

    # --- Scripted demos (show the complete loop working) ---
    print("\n" + "#"*60)
    print("# Scripted Demos — Complete ReAct Loop")
    print("#"*60)

    for i, demo in enumerate(DEMOS):
        print(f"\n{'='*60}")
        print(f"DEMO {i+1}/{len(DEMOS)}")
        scripted_model = make_scripted_model(demo["steps"])
        agent = ReActAgent(scripted_model, max_steps=8, verbose=True)
        agent.run(demo["question"])

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY — What We Built")
    print("="*60)
    print("""
Key concepts from this lesson:

1. THE COMPLETE REACT LOOP
   prompt → model → parse → (action? dispatch + append obs | final? return | else: append + retry)

2. MULTI-HOP REASONING
   Some questions require chaining multiple tool calls.
   The model plans ahead: "I need X, which requires looking up Y first."

3. STOPPING CONDITIONS
   - "Final Answer:" detected → clean exit
   - max_steps reached → safety exit (prevents infinite loops)
   - Model goes off-script → append anyway, try again

4. THE KNOWLEDGE BASE SEARCH TOOL
   Gives the model access to facts it might not have memorized.
   In production: swap this for web search or a vector database.

5. PROMPT ENGINEERING FOR AGENTS
   - Few-shot examples teach the model the format
   - Tool descriptions guide when to use each tool
   - The prompt grows with each step (it's the agent's "memory")

6. GPT-2 REALITY CHECK
   GPT-2 struggles because:
   - It generates statistically likely text, not instruction-following text
   - It wasn't trained to stop and call tools
   - Real agents use instruction-tuned models (GPT-4, Claude, Llama-3-instruct)

7. WHAT COMES NEXT (Lesson 4)
   - Structured function calling (the modern API approach)
   - How Anthropic/OpenAI handle tool use natively
   - Using the Anthropic API with claude-haiku-4-5
""")
