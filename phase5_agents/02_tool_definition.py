"""
Phase 5, Lesson 2: Tool Definition — How Do You Give an LLM Tools?
==================================================================

CONCEPT FIRST
-------------
In Lesson 1, we parsed tool calls with a regex and ran them.
But how do you *define* a tool so the model knows it exists?

The modern standard (from OpenAI, adopted by Anthropic and others) is:
    JSON Schema — a structured description of each tool.

A tool definition looks like this:

    {
        "name": "calculator",
        "description": "Evaluates a math expression and returns the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '2 + 2'"
                }
            },
            "required": ["expression"]
        }
    }

The description is CRITICAL — it's what the model reads to understand
what the tool does and when to use it. Write it clearly.

TWO APPROACHES TO TOOL CALLING
-------------------------------
1. Text-based (GPT-2 style): Model generates text like "Action: tool(args)"
   We parse it with regex. Fragile, but works with any model.

2. Structured (modern APIs): Model outputs a special JSON token stream.
   The API guarantees valid JSON. Reliable. Only works with models trained for it.

We'll build both in this lesson.

HOW TO RUN
----------
    uv run phase5_agents/02_tool_definition.py
"""

import json
import re
import math
from datetime import datetime
from typing import Any, Callable


# ============================================================
# STEP 1: Define 4 tools as Python functions
# ============================================================

def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports +, -, *, /, **, sqrt, etc."""
    try:
        # Allow math functions like sqrt, pi, etc.
        safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_globals["__builtins__"] = {}
        result = eval(expression, safe_globals, {})
        return str(round(float(result), 8))
    except Exception as e:
        return f"Error: {e}"


def string_length(text: str) -> str:
    """Return the number of characters in a string."""
    return str(len(text))


def current_date(format: str = "%Y-%m-%d") -> str:
    """Return the current date in the specified format."""
    try:
        return datetime.now().strftime(format)
    except Exception as e:
        return f"Error: {e}"


def list_reverse(items: str) -> str:
    """
    Reverse a comma-separated list of items.
    Input: "apple, banana, cherry"
    Output: "cherry, banana, apple"
    """
    parts = [item.strip() for item in items.split(",")]
    return ", ".join(reversed(parts))


# ============================================================
# STEP 2: JSON Schema tool definitions (the industry standard)
# ============================================================
# This is exactly the format OpenAI and Anthropic use.
# The model reads these schemas to know what tools it has.

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": (
            "Evaluates a mathematical expression and returns the numeric result. "
            "Supports +, -, *, /, ** (power), sqrt(), sin(), cos(), pi, etc. "
            "Use this for any arithmetic or math computation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate. Example: '2 + 2', 'sqrt(144)', '3 ** 4'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "string_length",
        "description": "Returns the number of characters in a given string. Use this when asked about string length.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The string to measure."
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "current_date",
        "description": "Returns the current date. Use this when the user asks what date or day it is.",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Date format string. Default: '%Y-%m-%d'. Use '%B %d, %Y' for 'January 15, 2025'.",
                    "default": "%Y-%m-%d"
                }
            },
            "required": []
        }
    },
    {
        "name": "list_reverse",
        "description": "Reverses a comma-separated list of items.",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "string",
                    "description": "Comma-separated items to reverse. Example: 'a, b, c'"
                }
            },
            "required": ["items"]
        }
    },
]


# ============================================================
# STEP 3: Build a tool registry
# ============================================================
# A registry maps tool names → (function, schema).
# This is the single source of truth for our tool system.

class ToolRegistry:
    """
    Holds all available tools and their schemas.

    Think of this as a plug-in system: you register tools here,
    and the agent can discover and use any registered tool.
    """

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}

    def register(self, fn: Callable, schema: dict):
        """Add a tool to the registry."""
        name = schema["name"]
        self._tools[name] = fn
        self._schemas[name] = schema
        print(f"  Registered tool: {name}")

    def get_function(self, name: str) -> Callable | None:
        """Look up the Python function for a tool by name."""
        return self._tools.get(name)

    def get_schema(self, name: str) -> dict | None:
        """Look up the JSON schema for a tool by name."""
        return self._schemas.get(name)

    def all_schemas(self) -> list[dict]:
        """Return all schemas (what we pass to the model)."""
        return list(self._schemas.values())

    def schema_text(self) -> str:
        """Format all schemas as readable text for prompting."""
        lines = ["Available tools:"]
        for schema in self._schemas.values():
            lines.append(f"\n  Tool: {schema['name']}")
            lines.append(f"  Description: {schema['description']}")
            props = schema["parameters"].get("properties", {})
            if props:
                lines.append("  Parameters:")
                for param_name, param_info in props.items():
                    required = param_name in schema["parameters"].get("required", [])
                    req_label = "(required)" if required else "(optional)"
                    lines.append(f"    - {param_name} {req_label}: {param_info['description']}")
        return "\n".join(lines)


# Build and populate the registry
registry = ToolRegistry()
print("Building tool registry...")
registry.register(calculator, TOOL_SCHEMAS[0])
registry.register(string_length, TOOL_SCHEMAS[1])
registry.register(current_date, TOOL_SCHEMAS[2])
registry.register(list_reverse, TOOL_SCHEMAS[3])


# ============================================================
# STEP 4: Text-based parsing (for GPT-2 style models)
# ============================================================
# When a model outputs: "Action: calculator(2 + 2)"
# We need to parse that into: tool_name="calculator", args={"expression": "2 + 2"}

def parse_text_action(text: str):
    """
    Parse a ReAct-style action from model text.

    Handles formats like:
      Action: calculator(2 + 2)
      Action: string_length("hello world")
      Action: current_date()

    Returns: (tool_name, raw_args_string) or None
    """
    # Match: Action: tool_name(anything)
    match = re.search(r"Action:\s*(\w+)\(([^)]*)\)", text, re.IGNORECASE)
    if not match:
        return None

    tool_name = match.group(1).strip()
    raw_args = match.group(2).strip()

    # Strip surrounding quotes if present (e.g., "hello world" → hello world)
    if raw_args.startswith('"') and raw_args.endswith('"'):
        raw_args = raw_args[1:-1]
    elif raw_args.startswith("'") and raw_args.endswith("'"):
        raw_args = raw_args[1:-1]

    return tool_name, raw_args


def dispatch_text_action(tool_name: str, raw_args: str) -> str:
    """
    Given a tool name and raw argument string, call the tool.

    For single-argument tools, we pass the whole string as the first arg.
    For multi-argument tools, you'd need more sophisticated parsing.
    """
    fn = registry.get_function(tool_name)
    if fn is None:
        return f"Error: Unknown tool '{tool_name}'"

    # Get the schema to know the parameter name
    schema = registry.get_schema(tool_name)
    required = schema["parameters"].get("required", [])

    if not required and not raw_args:
        # No arguments needed (like current_date with default format)
        return fn()
    elif required:
        # Pass the raw string as the first required parameter
        first_param = required[0]
        return fn(**{first_param: raw_args})
    else:
        return fn()


# ============================================================
# STEP 5: JSON-based parsing (for modern instruction-tuned models)
# ============================================================
# Modern APIs return tool calls as structured JSON, not text.
# The model output looks like:
#
#   {
#     "type": "tool_use",
#     "name": "calculator",
#     "input": {"expression": "2 + 2"}
#   }
#
# This is much more reliable than regex parsing.

def dispatch_json_action(tool_call: dict) -> str:
    """
    Execute a tool call that came in as structured JSON.

    Expected format:
      {"name": "tool_name", "input": {"param": "value"}}

    This mirrors the Anthropic API tool_use format.
    """
    tool_name = tool_call.get("name")
    input_args = tool_call.get("input", {})

    fn = registry.get_function(tool_name)
    if fn is None:
        return f"Error: Unknown tool '{tool_name}'"

    # Call the function with keyword arguments from the JSON
    try:
        return fn(**input_args)
    except Exception as e:
        return f"Error calling {tool_name}: {e}"


# ============================================================
# STEP 6: Demonstrations
# ============================================================

def demo_text_parsing():
    """Show how we parse text-format tool calls."""
    print("\n" + "="*60)
    print("DEMO 1: Text-Based Tool Call Parsing")
    print("="*60)

    test_cases = [
        "Thought: I'll compute this.\nAction: calculator(sqrt(144) + 10)\nObservation:",
        "Action: string_length(Hello, World!)",
        "Action: current_date()",
        "Action: list_reverse(alpha, beta, gamma, delta)",
        "Thought: No action needed here.",  # Should return None
    ]

    for text in test_cases:
        print(f"\nInput text:\n  {text.split(chr(10))[0]}")
        result = parse_text_action(text)
        if result:
            tool_name, raw_args = result
            output = dispatch_text_action(tool_name, raw_args)
            print(f"  → Tool: {tool_name}({raw_args!r})")
            print(f"  → Result: {output}")
        else:
            print(f"  → No action found")


def demo_json_dispatch():
    """Show how we dispatch JSON-format tool calls."""
    print("\n" + "="*60)
    print("DEMO 2: JSON-Based Tool Call Dispatch")
    print("="*60)

    # These simulate what a modern API (Anthropic/OpenAI) would return
    json_calls = [
        {"name": "calculator", "input": {"expression": "365 ** 0.5"}},
        {"name": "string_length", "input": {"text": "supercalifragilistic"}},
        {"name": "current_date", "input": {"format": "%B %d, %Y"}},
        {"name": "list_reverse", "input": {"items": "one, two, three, four, five"}},
        {"name": "unknown_tool", "input": {}},  # Error case
    ]

    for call in json_calls:
        print(f"\nTool call: {json.dumps(call)}")
        result = dispatch_json_action(call)
        print(f"Result:    {result}")


def demo_schema_inspection():
    """Show what the model sees when given tool schemas."""
    print("\n" + "="*60)
    print("DEMO 3: What the Model Sees (Tool Schemas)")
    print("="*60)

    print("\n--- As readable text (for prompting GPT-2 style) ---")
    print(registry.schema_text())

    print("\n--- As JSON (for modern API tool definitions) ---")
    # Show just the first schema as an example
    print(json.dumps(registry.all_schemas()[0], indent=2))


def demo_tool_directly():
    """Show each tool working by itself."""
    print("\n" + "="*60)
    print("DEMO 4: Tools Working Directly")
    print("="*60)

    print(f"\ncalculator('sin(pi / 2)')  → {calculator('sin(pi / 2)')}")
    print(f"calculator('2 ** 10')      → {calculator('2 ** 10')}")
    print(f"string_length('LLM agent') → {string_length('LLM agent')}")
    print(f"current_date()             → {current_date()}")
    print(f"list_reverse('a, b, c')   → {list_reverse('a, b, c')}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(__doc__)

    demo_schema_inspection()
    demo_tool_directly()
    demo_text_parsing()
    demo_json_dispatch()

    print("\n" + "="*60)
    print("SUMMARY — What We Built")
    print("="*60)
    print("""
Key concepts from this lesson:

1. JSON SCHEMA IS THE STANDARD
   Every major AI API (Anthropic, OpenAI, Gemini) uses JSON schema
   to describe tools. The model reads descriptions to know WHEN to use each tool.
   Good descriptions are essential — they're the model's instruction manual.

2. THE TOOL REGISTRY
   A central store: name → (function, schema)
   Makes it easy to add/remove tools without changing the agent loop.

3. TEXT PARSING VS JSON DISPATCH
   - Text parsing: regex to find "Action: tool(args)" — works with any model
   - JSON dispatch: structured call like {"name": "tool", "input": {...}}
     More reliable, only possible with models trained for function calling

4. THE PARAMETER PATTERN
   Schema defines: name, description, parameters (type, properties, required)
   At runtime: extract parameters from model output → call function(**kwargs)

5. WHAT COMES NEXT (Lesson 3)
   Full ReAct agent loop with:
   - Multiple tools
   - A local knowledge base (Wikipedia-style search)
   - Multi-hop questions that require several tool calls to answer
""")
