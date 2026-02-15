"""
Phase 5, Lesson 5: Build a Coding Agent
========================================

CONCEPT FIRST
-------------
We've built the pieces:
  - The ReAct loop (Lesson 1, 3)
  - Tool definition and dispatch (Lesson 2)
  - Structured function calling (Lesson 4)

Now let's build something real: a *coding agent*.

A coding agent is given a programming task and autonomously:
  1. Writes the code to a file
  2. Runs the code
  3. Reads the output
  4. Fixes errors if any
  5. Verifies the result
  6. Reports success

This is EXACTLY the pattern behind:
  - Claude Code (this tool you're using right now)
  - Cursor's "Composer" feature
  - GitHub Copilot Workspace
  - Devin (the "AI software engineer")

The only differences in production:
  - Larger, instruction-tuned models
  - More tools (web search, git, terminal, browser)
  - Better context management (RAG, file trees)
  - Safety sandboxing (Docker containers)

TOOLS WE'LL BUILD
-----------------
1. read_file(path) — read a file from disk
2. write_file(path, content) — write content to a file
3. run_python(code) — execute Python code and capture output
4. list_files(directory) — list files in a directory
5. web_search(query) — mocked web search (real one would call an API)

THE TASK
--------
We'll give the agent this task:
  "Write a Python function called sort_dicts_by_key(data, key) that sorts
   a list of dictionaries by a given key. Then write tests for it."

Watch the agent:
  → Think about the approach
  → Write the function to a file
  → Write tests to another file
  → Run the tests
  → (If tests fail) fix and re-run
  → Report success

HOW TO RUN
----------
    uv run phase5_agents/05_build_a_coding_agent.py

    (Optionally set ANTHROPIC_API_KEY to use the real Claude model)
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime


# ============================================================
# STEP 1: Workspace setup
# ============================================================
# The agent needs a directory to work in.
# We use a temp directory so we don't litter the project with agent files.

class AgentWorkspace:
    """
    A temporary workspace directory for the coding agent.
    Files created here are cleaned up automatically when done.
    """

    def __init__(self, base_name: str = "coding_agent_workspace"):
        # Create a temp directory for this session
        self.root = Path(tempfile.mkdtemp(prefix=f"{base_name}_"))
        print(f"Agent workspace: {self.root}")

    def path(self, filename: str) -> Path:
        """Get the full path for a file in the workspace."""
        return self.root / filename

    def cleanup(self):
        """Remove the workspace directory."""
        shutil.rmtree(self.root, ignore_errors=True)
        print(f"Workspace cleaned up: {self.root}")


# ============================================================
# STEP 2: Tool implementations
# ============================================================

def make_tools(workspace: AgentWorkspace):
    """
    Create tool functions bound to the given workspace.

    We use a factory function so tools know WHERE to read/write files.
    This is the "sandboxing" pattern — tools are scoped to the workspace.
    """

    def read_file(path: str) -> str:
        """Read a file from the agent's workspace."""
        full_path = workspace.path(path)
        try:
            return full_path.read_text()
        except FileNotFoundError:
            return f"Error: File '{path}' not found in workspace."
        except Exception as e:
            return f"Error reading '{path}': {e}"

    def write_file(path: str, content: str) -> str:
        """Write content to a file in the agent's workspace."""
        full_path = workspace.path(path)
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            return f"Successfully wrote {len(content)} characters to '{path}'."
        except Exception as e:
            return f"Error writing '{path}': {e}"

    def run_python(filename: str) -> str:
        """
        Execute a Python file in the workspace and return its output.

        Uses subprocess — the code runs in a fresh Python interpreter,
        not in our current process. This is the basic sandboxing approach.
        """
        full_path = workspace.path(filename)
        if not full_path.exists():
            return f"Error: '{filename}' not found."

        try:
            result = subprocess.run(
                [sys.executable, str(full_path)],
                capture_output=True,
                text=True,
                timeout=15,  # Kill after 15 seconds (prevents infinite loops)
                cwd=str(workspace.root),  # Run from workspace directory
            )
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            return output.strip() if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Code timed out after 15 seconds."
        except Exception as e:
            return f"Error running '{filename}': {e}"

    def list_files(directory: str = ".") -> str:
        """List files in the workspace (or a subdirectory)."""
        full_path = workspace.path(directory) if directory != "." else workspace.root
        try:
            files = list(full_path.iterdir())
            if not files:
                return "No files in workspace."
            return "\n".join(f.name for f in sorted(files))
        except Exception as e:
            return f"Error listing '{directory}': {e}"

    def web_search(query: str) -> str:
        """
        Mock web search — returns canned results for common queries.

        In production, this would call the Brave Search API, SerpAPI, etc.
        The mock lets us see the tool in the loop without API calls.
        """
        knowledge = {
            "sort list of dicts python": (
                "To sort a list of dicts by a key in Python, use sorted() with a lambda:\n"
                "sorted(data, key=lambda x: x['key_name'])\n"
                "Or use itemgetter: from operator import itemgetter; sorted(data, key=itemgetter('key'))"
            ),
            "python unittest": (
                "Python's unittest module: import unittest\n"
                "class TestFoo(unittest.TestCase):\n"
                "    def test_bar(self): self.assertEqual(result, expected)\n"
                "if __name__ == '__main__': unittest.main()"
            ),
            "python list comprehension": (
                "List comprehension syntax: [expression for item in iterable if condition]\n"
                "Example: [x*2 for x in range(10) if x % 2 == 0]"
            ),
        }
        query_lower = query.lower()
        for key, value in knowledge.items():
            if any(word in query_lower for word in key.split()):
                return value
        return f"No results found for '{query}'. (Mock search — add more entries for real use)"

    return {
        "read_file": read_file,
        "write_file": write_file,
        "run_python": run_python,
        "list_files": list_files,
        "web_search": web_search,
    }


# ============================================================
# STEP 3: Tool schemas (Anthropic format)
# ============================================================

TOOL_SCHEMAS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file in the workspace. Use to check existing code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace (e.g., 'solution.py')"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the workspace. Use to create or update code files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write (e.g., 'solution.py')"},
                "content": {"type": "string", "description": "Full file contents to write"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "run_python",
        "description": "Execute a Python file and return its stdout/stderr output. Use to test code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Python file to run (e.g., 'test_solution.py')"}
            },
            "required": ["filename"]
        }
    },
    {
        "name": "list_files",
        "description": "List all files in the workspace directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory to list (default: '.')"}
            },
            "required": []
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for Python documentation, examples, or best practices.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
]


# ============================================================
# STEP 4: The coding agent prompt
# ============================================================

def build_coding_agent_prompt(task: str, tool_names: list[str]) -> str:
    return f"""You are a skilled Python developer. You have access to a workspace where you can write and run code.

AVAILABLE TOOLS:
- read_file(path): Read a file from the workspace
- write_file(path, content): Write a file to the workspace
- run_python(filename): Run a Python file and see its output
- list_files(directory): List workspace files
- web_search(query): Search for Python documentation or examples

YOUR TASK:
{task}

HOW TO WORK:
1. Think through the approach
2. Write the implementation
3. Write tests
4. Run the tests
5. If tests fail, read the error, fix the code, and re-run
6. When all tests pass, report success

FORMAT YOUR RESPONSES AS:
Thought: [your reasoning]
Action: tool_name
Input: {{"param": "value"}}

When done:
Final Answer: [summary of what you built and the test results]

RULES:
- Always run tests before reporting success
- If a test fails, fix the code (don't just change the test to pass)
- Keep code clean and well-commented
- Write at least 3 test cases

Begin now.
"""


# ============================================================
# STEP 5: The agent loop for a coding agent
# ============================================================

def parse_coding_action(text: str):
    """
    Parse action from coding agent format:

        Thought: ...
        Action: tool_name
        Input: {"param": "value"}

    Returns: ("action", tool_name, input_dict) | ("final", answer) | ("unknown", text)
    """
    import re

    # Check for Final Answer
    final_match = re.search(r"Final Answer:\s*(.+?)(?=\Z|\nThought:|\nAction:)", text, re.DOTALL)
    if final_match:
        return ("final", final_match.group(1).strip())

    # Check for Action + Input pattern
    action_match = re.search(r"Action:\s*(\w+)", text)

    if action_match:
        tool_name = action_match.group(1).strip()
        input_dict = {}

        # Find the opening brace after "Input:"
        input_label = text.find("Input:")
        if input_label != -1:
            brace_start = text.find("{", input_label)
            if brace_start != -1:
                # Walk forward tracking brace depth and string context so we
                # don't stop early at a } that's inside a string value.
                depth = 0
                in_string = False
                escape_next = False
                end = brace_start
                for i in range(brace_start, len(text)):
                    ch = text[i]
                    if escape_next:
                        escape_next = False
                        continue
                    if ch == "\\" and in_string:
                        escape_next = True
                        continue
                    if ch == '"':
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                json_str = text[brace_start : end + 1]
                try:
                    input_dict = json.loads(json_str)
                except json.JSONDecodeError:
                    input_dict = {}

        return ("action", tool_name, input_dict)

    return ("unknown", text.strip())


class CodingAgent:
    """
    A coding agent that can write, run, and fix Python code.
    """

    def __init__(self, model_fn, tools: dict, max_steps: int = 15, verbose: bool = True):
        self.model_fn = model_fn
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        self.step_log = []  # Track all steps for review

    def run(self, task: str) -> str:
        """Run the coding agent on a task."""
        prompt = build_coding_agent_prompt(task, list(self.tools.keys()))

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"CODING AGENT TASK:")
            print(f"{task}")
            print(f"{'='*60}")

        for step in range(self.max_steps):
            # Generate next action from the model
            new_text = self.model_fn(prompt)

            if self.verbose:
                print(f"\n[Step {step + 1}]")
                print(new_text.strip())

            # Log the step
            self.step_log.append({"step": step + 1, "model_output": new_text.strip()})

            parsed = parse_coding_action(new_text)

            if parsed[0] == "final":
                answer = parsed[1]
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"✓ TASK COMPLETE")
                    print(f"{'='*60}")
                    print(answer)
                return answer

            elif parsed[0] == "action":
                _, tool_name, input_dict = parsed
                if tool_name not in self.tools:
                    observation = f"Unknown tool: {tool_name}"
                else:
                    if self.verbose:
                        print(f"\n  → Tool: {tool_name}({input_dict})")
                    try:
                        observation = self.tools[tool_name](**input_dict)
                    except Exception as e:
                        observation = f"Tool error: {e}"

                    if self.verbose:
                        # Show first 500 chars of output (avoid flooding)
                        obs_preview = observation[:500]
                        if len(observation) > 500:
                            obs_preview += f"\n... (truncated, {len(observation)} total chars)"
                        print(f"  → Observation: {obs_preview}")

                self.step_log[-1]["tool"] = tool_name
                self.step_log[-1]["observation"] = observation

                # Append action + observation to the prompt
                prompt += new_text
                if not new_text.endswith("\n"):
                    prompt += "\n"
                prompt += f"Observation: {observation}\n"

            else:
                # Unknown — append and try again
                prompt += new_text
                if not new_text.endswith("\n"):
                    prompt += "\n"

        return "Agent did not complete within step limit."


# ============================================================
# STEP 6: Scripted model that builds a real solution
# ============================================================
# A scripted model that perfectly solves the sorting task.
# This shows the COMPLETE LOOP working end-to-end.

def make_scripted_coding_model():
    """
    Returns a model function that serves pre-scripted steps.
    These steps correctly solve the coding task and verify it.
    """
    SOLUTION_CODE = '''"""
Sort a list of dictionaries by a specified key.
"""


def sort_dicts_by_key(data: list, key: str, reverse: bool = False) -> list:
    """
    Sort a list of dictionaries by the value of a given key.

    Args:
        data: List of dictionaries to sort
        key: The dictionary key to sort by
        reverse: If True, sort in descending order (default: False)

    Returns:
        A new sorted list (original list is not modified)

    Raises:
        KeyError: If any dict in the list doesn't have the specified key
    """
    return sorted(data, key=lambda d: d[key], reverse=reverse)
'''

    TEST_CODE = '''"""
Tests for sort_dicts_by_key function.
"""
import unittest
from solution import sort_dicts_by_key


class TestSortDictsByKey(unittest.TestCase):

    def test_sort_by_name_ascending(self):
        """Sort a list of people by name (ascending)."""
        data = [
            {"name": "Charlie", "age": 30},
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 35},
        ]
        result = sort_dicts_by_key(data, "name")
        names = [d["name"] for d in result]
        self.assertEqual(names, ["Alice", "Bob", "Charlie"])

    def test_sort_by_age_ascending(self):
        """Sort a list of people by age (ascending)."""
        data = [
            {"name": "Charlie", "age": 30},
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 35},
        ]
        result = sort_dicts_by_key(data, "age")
        ages = [d["age"] for d in result]
        self.assertEqual(ages, [25, 30, 35])

    def test_sort_descending(self):
        """Sort in descending order."""
        data = [
            {"score": 75},
            {"score": 90},
            {"score": 60},
        ]
        result = sort_dicts_by_key(data, "score", reverse=True)
        scores = [d["score"] for d in result]
        self.assertEqual(scores, [90, 75, 60])

    def test_does_not_modify_original(self):
        """Original list should not be modified."""
        data = [{"id": 3}, {"id": 1}, {"id": 2}]
        original = data.copy()
        sort_dicts_by_key(data, "id")
        self.assertEqual(data, original)

    def test_empty_list(self):
        """Empty list should return empty list."""
        self.assertEqual(sort_dicts_by_key([], "key"), [])

    def test_single_item(self):
        """Single-item list should return same item."""
        data = [{"value": 42}]
        self.assertEqual(sort_dicts_by_key(data, "value"), [{"value": 42}])

    def test_sort_by_price(self):
        """Sort products by price."""
        products = [
            {"name": "Widget", "price": 9.99},
            {"name": "Gadget", "price": 24.99},
            {"name": "Doohickey", "price": 4.99},
        ]
        result = sort_dicts_by_key(products, "price")
        prices = [p["price"] for p in result]
        self.assertEqual(prices, [4.99, 9.99, 24.99])


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''

    steps = [
        # Step 1: Think and search for approach
        "Thought: I need to write a sort_dicts_by_key function. Let me first check if there are any existing files.\nAction: list_files\nInput: {}\n",
        # Step 2: Search for best practices
        "Thought: No existing files. Let me search for the best approach to sort a list of dicts in Python.\nAction: web_search\nInput: {\"query\": \"sort list of dicts python\"}\n",
        # Step 3: Write the solution
        f"Thought: I'll use sorted() with a lambda. Let me write the implementation.\nAction: write_file\nInput: {{\"path\": \"solution.py\", \"content\": {json.dumps(SOLUTION_CODE)}}}\n",
        # Step 4: Write the tests
        f"Thought: Now I'll write comprehensive tests covering ascending, descending, empty list, and edge cases.\nAction: write_file\nInput: {{\"path\": \"test_solution.py\", \"content\": {json.dumps(TEST_CODE)}}}\n",
        # Step 5: Run the tests
        "Thought: I've written both the function and tests. Let me run the tests to verify everything works.\nAction: run_python\nInput: {\"filename\": \"test_solution.py\"}\n",
        # Step 6: Read the solution to verify
        "Thought: All tests passed! Let me read the final solution to confirm it's clean.\nAction: read_file\nInput: {\"path\": \"solution.py\"}\n",
        # Step 7: Final answer
        "Thought: Everything looks good. The function is implemented and all 7 tests pass.\nFinal Answer: Successfully built sort_dicts_by_key().\n\nFiles created:\n- solution.py: The implementation using sorted() with a lambda key\n- test_solution.py: 7 test cases covering ascending, descending, empty list, no mutation, single item, and real-world usage\n\nAll 7 tests passed. The function correctly sorts a list of dicts by any key and supports reverse ordering.\n",
    ]

    step = [0]

    def model_fn(prompt: str) -> str:
        idx = step[0]
        step[0] += 1
        if idx < len(steps):
            return steps[idx]
        return "Final Answer: Task complete.\n"

    return model_fn


# ============================================================
# STEP 7: Real agent using the `claude` CLI
# ============================================================
# We use the `claude -p` (print mode) command that's already
# installed and authenticated as part of Claude Code.
# No separate API key needed — it reuses your existing session.
#
# This is the ReAct loop from Lesson 3, but with a real model
# instead of GPT-2. The model generates Thought/Action/Input,
# we dispatch the tool, append the Observation, and repeat.

def make_claude_cli_model(system_prompt: str):
    """
    Return a model_fn that calls `claude -p` with the given prompt.

    claude -p runs a single non-interactive query and prints the result,
    making it easy to drive from a loop like this one.
    """
    def model_fn(prompt: str) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}"
        result = subprocess.run(
            ["claude", "-p", full_prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return f"Error from claude CLI: {result.stderr[:200]}"
        return result.stdout
    return model_fn


def run_real_coding_agent(task: str, workspace: AgentWorkspace):
    """
    Run the coding agent using the real Claude model via the CLI.
    Uses the same ReAct text loop as Lesson 3 — no API key needed.

    NOTE: This only works when run from a plain terminal, NOT from inside
    a Claude Code session (which blocks nested claude invocations).
    Run it yourself: uv run phase5_agents/05_build_a_coding_agent.py
    """
    # Check the claude CLI is available
    check = subprocess.run(["which", "claude"], capture_output=True, text=True)
    if check.returncode != 0:
        print("claude CLI not found — using scripted demo instead")
        return False

    # Check we're not inside a Claude Code session (nested calls are blocked)
    if os.environ.get("CLAUDECODE"):
        print("Running inside Claude Code — nested claude calls are blocked.")
        print("Run this script from a plain terminal to use the real model.")
        return False

    print("\nUsing real Claude model via `claude -p` CLI...")

    system_prompt = """You are a skilled Python developer with access to tools for writing and running code.

AVAILABLE TOOLS:
- list_files: list_files() — List files in the workspace.
- web_search: web_search(query) — Search for Python docs or examples.
- write_file: write_file(path, content) — Write a file to the workspace.
- run_python: run_python(filename) — Run a Python file and see its output.
- read_file: read_file(path) — Read a file from the workspace.

FORMAT — use this exactly:
Thought: [your reasoning]
Action: tool_name
Input: {"param": "value"}

When done:
Final Answer: [summary]

RULES:
- Always run tests before reporting success.
- If tests fail, fix the code and re-run.
- Write at least 5 test cases."""

    tools_dict = make_tools(workspace)
    model_fn = make_claude_cli_model(system_prompt)
    agent = CodingAgent(model_fn, tools_dict, max_steps=15, verbose=True)
    result = agent.run(task)
    return bool(result and "complete" in result.lower() or result)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(__doc__)

    TASK = (
        "Write a Python function called sort_dicts_by_key(data, key, reverse=False) "
        "that sorts a list of dictionaries by a given key. "
        "Save it to 'solution.py'. "
        "Then write a test file 'test_solution.py' with at least 5 test cases. "
        "Run the tests and make sure they all pass."
    )

    # Create a workspace
    workspace = AgentWorkspace()
    tools = make_tools(workspace)

    try:
        # Try real Claude CLI agent first, fall back to scripted demo
        print("\n" + "#"*60)
        print("# REAL CODING AGENT (claude CLI)")
        print("#"*60)
        used_real = run_real_coding_agent(TASK, workspace)

        if not used_real:
            # Fall back to scripted demo
            print("\n" + "#"*60)
            print("# SCRIPTED CODING AGENT DEMO")
            print("# (Shows complete loop mechanics with pre-written steps)")
            print("#"*60)

            scripted_model = make_scripted_coding_model()
            agent = CodingAgent(scripted_model, tools, max_steps=15, verbose=True)
            agent.run(TASK)

        # Show final workspace state
        print("\n" + "="*60)
        print("FINAL WORKSPACE CONTENTS")
        print("="*60)
        for path in sorted(workspace.root.iterdir()):
            print(f"\n--- {path.name} ---")
            content = path.read_text()
            # Show first 40 lines
            lines = content.split("\n")
            for line in lines[:40]:
                print(line)
            if len(lines) > 40:
                print(f"... ({len(lines) - 40} more lines)")

    finally:
        # Always clean up
        workspace.cleanup()

    print("\n" + "="*60)
    print("SUMMARY — What We Built + Phase 5 Complete")
    print("="*60)
    print("""
Key concepts from this lesson:

1. THE CODING AGENT PATTERN
   Tools: read_file, write_file, run_python, list_files, web_search
   Loop: write → run → read output → fix → run again → verify → done

2. SANDBOXING
   The agent works in an isolated temp directory.
   Code runs in a subprocess (not the main process).
   This prevents the agent from accidentally modifying your project.

3. THE SELF-REPAIR LOOP
   When tests fail:
     Observation includes the error message
     The model reads the error in context
     It reasons about the fix
     It rewrites the file
     It runs tests again
   This is the "agentic" part — the model acts until the goal is met.

4. THIS IS CLAUDE CODE
   What you're using right now (Claude Code) does exactly this:
     - read_file → Read tool
     - write_file → Write / Edit tools
     - run_python → Bash tool
     - web_search → WebSearch tool
   The loop, the tools, and the self-repair pattern are identical.

5. WHAT MAKES PRODUCTION AGENTS BETTER
   - Larger, instruction-tuned models (Claude 3, GPT-4, Llama 70B)
   - Better context management (summarization, file trees, RAG)
   - More tools (git, browser, API calls, database access)
   - Proper sandboxing (Docker containers)
   - Human-in-the-loop checkpoints for risky operations

────────────────────────────────────────────────────────────
PHASE 5 COMPLETE — FULL LEARNING PATH SUMMARY
────────────────────────────────────────────────────────────

Phase 1: Foundations
  Tensors → Autograd → Neural Networks → Language Model Basics

Phase 2: Transformers from Scratch
  Embeddings → Attention → Transformer Blocks → nanoGPT

Phase 3: Fine-tuning
  HuggingFace → Full Fine-tune → LoRA → PEFT Pipeline

Phase 4: Project Delivery
  Dataset Building → LoRA Training → Evaluation → FastAPI Server

Phase 5: Agents & Tool Use
  ReAct Loop → Tool Definitions → Full ReAct Agent
  → Function Calling → Coding Agent

You now understand the full stack:
  from "what is a tensor" to "how does Claude Code work"
────────────────────────────────────────────────────────────
""")
