"""
Phase 5, Lesson 1: LLM as a Reasoner — What Is an Agent?
=========================================================

CONCEPT FIRST
-------------
So far in this course, we've used LLMs in one direction:
  you give it a prompt → it generates text → done.

An "agent" changes this into a *loop*:
  you give it a goal → it generates text → if it needs a tool, run the tool →
  feed the result back → it generates more text → repeat until done.

That's it. Agents are just:
    LLM + loop + tools

THE REACT PATTERN (Reason + Act)
---------------------------------
ReAct is the most famous agent reasoning pattern, from a 2022 paper.
The idea: force the model to interleave *thinking* and *acting*.

Each step looks like this:

    Thought: I need to calculate 144 / 12. I'll use the calculator tool.
    Action: calculator(144 / 12)
    Observation: 12.0
    Thought: The result is 12. I can now give the final answer.
    Final Answer: 144 divided by 12 is 12.

Why does this work?
- The model "thinks out loud" before acting (reduces dumb mistakes)
- Each observation gives new information (grounds the model in reality)
- The loop continues until a stopping condition is met

WHY GPT-2 IS LIMITING HERE
----------------------------
GPT-2 (117M parameters) was trained to predict the next token in web text.
It was NOT trained to follow instructions, use tools, or stay on task.

Real agents need models that were:
1. Instruction-tuned (trained to follow directions)
2. RLHF-trained (aligned to be helpful and not go off-rails)
3. Large enough to reason (GPT-4 / Claude 3+ / Llama 70B+)

We'll use GPT-2 here to show the MECHANICS of the agent loop,
then explain what a real model would do differently.

HOW TO RUN
----------
    uv run phase5_agents/01_llm_as_reasoner.py
"""

import re
import sys

# ============================================================
# STEP 1: Define a simple tool — a calculator
# ============================================================
# A "tool" is just a Python function that does something the LLM can't do
# natively (like precise arithmetic, fetching live data, reading files).

def calculator(expression: str) -> str:
    """
    Safely evaluate a math expression and return the result as a string.

    Why str output? Because we're feeding results back into the LLM's
    text context. Everything in the loop is text.
    """
    try:
        # eval() is dangerous with user input — never do this in production!
        # For this learning exercise it's fine with controlled expressions.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(round(float(result), 6))
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# STEP 2: The ReAct prompt template
# ============================================================
# We teach the model HOW to use the agent loop by putting examples
# directly in the prompt. This is called "few-shot prompting".
#
# The format:
#   Thought: [model reasons about what to do]
#   Action: tool_name(arguments)
#   Observation: [we fill this in after running the tool]
#   ... repeat ...
#   Final Answer: [model gives the final answer]

REACT_SYSTEM_PROMPT = """You are a helpful assistant that can use tools to answer questions.

You have access to the following tools:
- calculator(expression): Evaluates a math expression. Example: calculator(2 + 2)

To use a tool, write:
Thought: [your reasoning]
Action: tool_name(arguments)

After each Action, you will see:
Observation: [tool result]

When you have the final answer, write:
Final Answer: [your answer]

Let's begin.

Question: What is 25 multiplied by 4?
Thought: I need to multiply 25 by 4. I'll use the calculator.
Action: calculator(25 * 4)
Observation: 100.0
Thought: The result is 100. I can answer now.
Final Answer: 25 multiplied by 4 is 100.

Question: What is the square root of 256?
Thought: I need to compute the square root of 256. I'll use the calculator.
Action: calculator(256 ** 0.5)
Observation: 16.0
Thought: The square root of 256 is 16.
Final Answer: The square root of 256 is 16.

Question: {question}
"""


# ============================================================
# STEP 3: The tool dispatcher
# ============================================================
# Given the model's output "Action: calculator(2 + 2)", we need to:
# 1. Parse the tool name and arguments
# 2. Run the right Python function
# 3. Return the result as a string

TOOLS = {
    "calculator": calculator,
}

def parse_action(text: str):
    """
    Extract the tool name and argument from a line like:
        Action: calculator(2 + 2)

    Returns: (tool_name, argument_string) or None if no action found.
    """
    # Look for: Action: tool_name(arguments)
    match = re.search(r"Action:\s*(\w+)\((.+?)\)", text, re.IGNORECASE)
    if match:
        tool_name = match.group(1).strip()
        args = match.group(2).strip()
        return tool_name, args
    return None

def dispatch_tool(tool_name: str, args: str) -> str:
    """
    Look up the tool by name and call it with the given arguments.
    """
    if tool_name in TOOLS:
        return TOOLS[tool_name](args)
    else:
        return f"Unknown tool: {tool_name}"


# ============================================================
# STEP 4: The agent loop (hand-built, no frameworks)
# ============================================================
# This is the core of every agent system:
#   1. Build a prompt with the question
#   2. Generate text from the model
#   3. Check: does the output contain an Action?
#   4. If yes: run the tool, append the Observation, go to step 2
#   5. If no: check for "Final Answer:" — return it
#   6. If max_steps reached: stop to avoid infinite loops

def run_agent_with_gpt2(question: str, max_steps: int = 5):
    """
    Run a ReAct agent loop using GPT-2 as the language model.

    IMPORTANT LIMITATION: GPT-2 will likely NOT follow the ReAct format
    correctly — it wasn't trained to. We'll show this limitation explicitly,
    then demonstrate what the loop WOULD do with a properly trained model.
    """
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    # Build the initial prompt
    prompt = REACT_SYSTEM_PROMPT.format(question=question)

    # Try to load GPT-2 for real inference
    try:
        from transformers import pipeline
        print("\nLoading GPT-2 (this uses the cached model)...")
        generator = pipeline(
            "text-generation",
            model="gpt2",
            device=-1,  # CPU — GPT-2 is small enough to not need MPS
        )
        use_real_model = True
        print("GPT-2 loaded.")
    except ImportError:
        print("transformers not available — using simulated output to show the loop")
        use_real_model = False

    # The full conversation so far (we append to this each step)
    conversation = prompt

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")

        if use_real_model:
            # Generate a continuation of the current conversation
            output = generator(
                conversation,
                max_new_tokens=80,        # short — we just need the next action
                do_sample=False,          # greedy decoding for consistency
                temperature=1.0,
                pad_token_id=50256,       # GPT-2 doesn't have a pad token by default
            )
            # Extract just the NEW text (not the prompt we already have)
            full_output = output[0]["generated_text"]
            new_text = full_output[len(conversation):]
        else:
            # Simulate what a well-trained model WOULD produce
            # (so we can see the loop working even without the model)
            simulated_responses = [
                "\nThought: I need to calculate this. Let me use the calculator.\nAction: calculator(25 * 13)\n",
                "\nThought: The result is 325. I can now give the final answer.\nFinal Answer: 25 multiplied by 13 is 325.\n",
            ]
            new_text = simulated_responses[min(step, len(simulated_responses) - 1)]

        print(f"Model output:\n{new_text.strip()}")

        # Append the new text to our conversation
        conversation += new_text

        # Check for Final Answer — if found, we're done
        if "Final Answer:" in new_text:
            # Extract just the final answer text
            final_match = re.search(r"Final Answer:\s*(.+)", new_text, re.DOTALL)
            if final_match:
                answer = final_match.group(1).strip()
                print(f"\n✓ FINAL ANSWER: {answer}")
                return answer
            break

        # Check for an Action — if found, run the tool
        action = parse_action(new_text)
        if action:
            tool_name, args = action
            print(f"\nDispatch: {tool_name}({args})")
            result = dispatch_tool(tool_name, args)
            print(f"Tool result: {result}")

            # Append the observation to the conversation so the model can see it
            observation = f"\nObservation: {result}\n"
            conversation += observation
            print(f"Appended to context: {observation.strip()}")
        else:
            # No action, no final answer — model went off-script
            # (This is what GPT-2 will usually do)
            print("\n⚠ Model did not follow the ReAct format.")
            print("  This is expected with GPT-2 — it wasn't trained for this.")
            print("  A model like Claude or GPT-4 would stay on format.")
            break

    print(f"\n(Reached max_steps={max_steps} or loop ended early)")
    return None


# ============================================================
# STEP 5: Demonstrate the agent loop WITHOUT a model
# ============================================================
# Since GPT-2 won't follow the format, let's build a "scripted demo"
# that shows EXACTLY what the loop does — step by step — so the
# mechanics are crystal clear.

def scripted_demo(question: str, scripted_steps: list):
    """
    Run the agent loop with pre-written model outputs.
    This makes the loop mechanics visible even when GPT-2 can't follow format.

    In real usage, replace scripted_steps with actual model generations.
    """
    print(f"\n{'='*60}")
    print(f"SCRIPTED DEMO (showing the loop mechanics)")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    conversation = REACT_SYSTEM_PROMPT.format(question=question)

    for step_num, model_output in enumerate(scripted_steps):
        print(f"\n--- Step {step_num + 1} ---")
        print(f"[Model would generate]:\n{model_output.strip()}")

        conversation += model_output

        # Final answer check
        if "Final Answer:" in model_output:
            match = re.search(r"Final Answer:\s*(.+)", model_output, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                print(f"\n✓ FINAL ANSWER: {answer}")
                return answer

        # Action check
        action = parse_action(model_output)
        if action:
            tool_name, args = action
            print(f"\n[Dispatcher]: run {tool_name}({args})")
            result = dispatch_tool(tool_name, args)
            print(f"[Tool returned]: {result}")

            observation = f"\nObservation: {result}\n"
            conversation += observation
            print(f"[Appended to context]: Observation: {result}")

    return None


# ============================================================
# MAIN — run both demos
# ============================================================

if __name__ == "__main__":
    print(__doc__)

    # --- Demo 1: Show the agent loop with GPT-2 (will likely fail gracefully) ---
    print("\n" + "#"*60)
    print("# DEMO 1: GPT-2 Agent (shows real model limitations)")
    print("#"*60)
    run_agent_with_gpt2("What is 144 divided by 12?")

    # --- Demo 2: Scripted demo to show loop mechanics clearly ---
    print("\n" + "#"*60)
    print("# DEMO 2: Scripted ReAct Loop (mechanics made visible)")
    print("#"*60)

    # Multi-step problem: square root of (20 * 5)
    # Step 1: model figures out it needs to compute 20 * 5 first
    # Step 2: model takes the square root of the result
    # Step 3: model gives the final answer
    scripted_demo(
        question="What is the square root of 20 multiplied by 5?",
        scripted_steps=[
            # Step 1: model reasons and calls the calculator
            "\nThought: I need to find the square root of (20 * 5). First let me compute 20 * 5.\nAction: calculator(20 * 5)\n",
            # After observation is appended, step 2: model takes sqrt
            "\nThought: 20 * 5 = 100. Now I need the square root of 100.\nAction: calculator(100 ** 0.5)\n",
            # After observation, step 3: final answer
            "\nThought: The square root of 100 is 10.0.\nFinal Answer: The square root of 20 multiplied by 5 is 10.\n",
        ]
    )

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY — What We Built")
    print("="*60)
    print("""
Key concepts from this lesson:

1. AN AGENT IS A LOOP
   Model generates → parse for actions → run tools → feed back → repeat

2. REACT PATTERN
   Thought → Action → Observation → Thought → ... → Final Answer

3. THE TOOL DISPATCHER
   A dict mapping tool names to Python functions.
   parse_action() extracts the name + args from model text.
   dispatch_tool() looks up and calls the right function.

4. STOPPING CONDITIONS
   - "Final Answer:" detected in model output
   - max_steps reached (prevents infinite loops)

5. WHY GPT-2 STRUGGLES
   - Not instruction-tuned (doesn't follow the format)
   - Not RLHF-trained (doesn't know what "helpful" looks like)
   - Too small to reason reliably (117M vs 7B+ for real agents)

6. WHAT COMES NEXT (Lesson 2)
   - Defining tools with JSON schemas (the OpenAI/Anthropic standard)
   - Building multiple tools and a proper dispatcher
   - Making the parsing more robust
""")
