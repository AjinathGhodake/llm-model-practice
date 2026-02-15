"""
Phase 5, Lesson 4: Function Calling — The Modern API Approach
=============================================================

CONCEPT FIRST
-------------
So far we've done "hand-rolled" tool use:
  - We write the tool descriptions in the system prompt as text
  - The model outputs text like "Action: calculator(2 + 2)"
  - We parse that text with regex

The modern approach is different:
  - We pass tools as structured JSON schemas to the API
  - The model outputs a STRUCTURED tool_use block (guaranteed valid JSON)
  - We dispatch based on the structured call
  - We send the result back as a structured tool_result block

This is what Anthropic calls "tool use" and OpenAI calls "function calling".

WHY IS STRUCTURED CALLING BETTER?
----------------------------------
Old (text parsing):
  ✗ Regex is fragile — "Action: calc(2+2)" vs "Action: calculator(2+2)"
  ✗ Args can be ambiguous — does calculator("2 + 2") mean the string "2 + 2"?
  ✗ Model might forget the format mid-conversation
  ✓ Works with any model (even GPT-2)

New (structured calling):
  ✓ Guaranteed valid JSON (API validates it)
  ✓ Named parameters — no ambiguity about which arg is which
  ✓ Model was explicitly trained for this format
  ✓ Easy to parse — just json.loads()
  ✗ Requires a model trained for function calling
  ✗ Requires API access (not just local weights)

THE ANTHROPIC TOOL USE FORMAT
-------------------------------
When you define tools and the model wants to use one, it returns:

    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01abc...",
          "name": "calculator",
          "input": {"expression": "sqrt(365)"}
        }
      ]
    }

You then send back:

    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_01abc...",
          "content": "19.104973"
        }
      ]
    }

The model sees the result and continues.

PYDANTIC SCHEMAS
----------------
Pydantic gives us Python-native type definitions that can auto-generate
JSON schemas. This is a clean way to define tools without writing JSON by hand.

HOW TO RUN
----------
    uv run phase5_agents/04_function_calling_pattern.py

    (Requires: uv add anthropic pydantic)
    (API key optional — script runs without it, showing mock behavior)
"""

import json
import math
import os
from datetime import datetime
from typing import Any


# ============================================================
# STEP 1: Our tools (same as before, but typed more explicitly)
# ============================================================

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_globals["__builtins__"] = {}
        result = eval(expression, safe_globals, {})
        return str(round(float(result), 8))
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str, unit: str = "celsius") -> str:
    """
    Mock weather function.
    In production, this would call a real weather API.
    """
    weather_data = {
        "london": {"temp_c": 12, "condition": "cloudy"},
        "tokyo": {"temp_c": 22, "condition": "sunny"},
        "new york": {"temp_c": 8, "condition": "windy"},
        "sydney": {"temp_c": 28, "condition": "sunny"},
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        temp = data["temp_c"]
        if unit.lower() in ("fahrenheit", "f"):
            temp = temp * 9/5 + 32
            unit_label = "°F"
        else:
            unit_label = "°C"
        return f"{city}: {temp}{unit_label}, {data['condition']}"
    return f"Weather data not available for {city}"


def count_words(text: str) -> str:
    """Count the number of words in a text string."""
    words = text.split()
    return f"{len(words)} words"


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units of measurement."""
    conversions = {
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,
        ("kg", "lbs"): lambda x: x * 2.20462,
        ("lbs", "kg"): lambda x: x * 0.453592,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("meters", "feet"): lambda x: x * 3.28084,
        ("feet", "meters"): lambda x: x * 0.3048,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](float(value))
        return f"{value} {from_unit} = {round(result, 4)} {to_unit}"
    return f"Don't know how to convert {from_unit} to {to_unit}"


# ============================================================
# STEP 2: Define tools using Pydantic schemas
# ============================================================
# Pydantic models auto-generate JSON schema from Python type hints.
# This is cleaner than writing JSON by hand.

try:
    from pydantic import BaseModel, Field

    # Define input schemas for each tool
    class CalculatorInput(BaseModel):
        expression: str = Field(
            description="The math expression to evaluate. "
                        "Examples: 'sqrt(144)', '2 ** 10', 'sin(pi/2)'"
        )

    class WeatherInput(BaseModel):
        city: str = Field(description="The name of the city")
        unit: str = Field(
            default="celsius",
            description="Temperature unit: 'celsius' or 'fahrenheit'"
        )

    class WordCountInput(BaseModel):
        text: str = Field(description="The text to count words in")

    class UnitConvertInput(BaseModel):
        value: float = Field(description="The numeric value to convert")
        from_unit: str = Field(description="Source unit (e.g., 'km', 'kg', 'celsius')")
        to_unit: str = Field(description="Target unit (e.g., 'miles', 'lbs', 'fahrenheit')")

    # Generate schemas from Pydantic models
    def pydantic_to_tool_schema(name: str, description: str, model_class) -> dict:
        """Convert a Pydantic model to an Anthropic-compatible tool schema."""
        schema = model_class.model_json_schema()
        # Remove Pydantic-specific fields we don't need
        schema.pop("title", None)
        return {
            "name": name,
            "description": description,
            "input_schema": schema,  # Anthropic uses 'input_schema', not 'parameters'
        }

    PYDANTIC_AVAILABLE = True
    print("✓ Pydantic available — will show schema generation from Python types")

except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠ Pydantic not installed. Run: uv add pydantic")
    print("  Falling back to hand-written schemas.")


# ============================================================
# STEP 3: Hand-written Anthropic-format tool schemas
# ============================================================
# These are equivalent to what Pydantic generates,
# but written by hand so you can see the exact format.

TOOLS_ANTHROPIC_FORMAT = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression. "
            "Supports: +, -, *, /, ** (power), sqrt(), sin(), cos(), pi, log(), abs(). "
            "Use for any arithmetic computation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression. Example: 'sqrt(144)', '2 ** 10'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'London', 'Tokyo'"
                },
                "unit": {
                    "type": "string",
                    "description": "Temperature unit: 'celsius' (default) or 'fahrenheit'",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "count_words",
        "description": "Count the number of words in a text string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to count words in"
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "convert_units",
        "description": "Convert a value between units of measurement (distance, weight, temperature).",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The numeric value to convert"
                },
                "from_unit": {
                    "type": "string",
                    "description": "Source unit (km, miles, kg, lbs, celsius, fahrenheit, meters, feet)"
                },
                "to_unit": {
                    "type": "string",
                    "description": "Target unit"
                }
            },
            "required": ["value", "from_unit", "to_unit"]
        }
    },
]

# Map tool names to Python functions
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "count_words": count_words,
    "convert_units": lambda **kwargs: convert_units(**kwargs),
}


# ============================================================
# STEP 4: Local function calling dispatcher
# ============================================================
# Simulates what happens after the API returns a tool_use block.

def execute_tool_call(tool_use_block: dict) -> str:
    """
    Execute a tool based on an Anthropic-style tool_use block.

    Input:
        {"type": "tool_use", "id": "...", "name": "calculator", "input": {"expression": "2+2"}}

    Output:
        The string result of running the tool.
    """
    name = tool_use_block["name"]
    input_args = tool_use_block.get("input", {})

    if name not in TOOL_FUNCTIONS:
        return f"Unknown tool: {name}"

    fn = TOOL_FUNCTIONS[name]
    try:
        return fn(**input_args)
    except Exception as e:
        return f"Tool error: {e}"


# ============================================================
# STEP 5: Simulate the complete Anthropic tool use conversation
# ============================================================
# This simulates what a real API call looks like — without needing the API.

def simulate_anthropic_tool_use():
    """
    Simulate the full Anthropic tool use conversation format.

    In a real API call:
    1. You send messages + tools to the API
    2. API returns a tool_use block
    3. You execute the tool
    4. You send back a tool_result
    5. API returns the final text answer

    We simulate step 2 here (the model decision), execute step 3,
    and show what the full conversation looks like.
    """
    print("\n" + "="*60)
    print("SIMULATED ANTHROPIC TOOL USE CONVERSATION")
    print("="*60)

    question = "What is the weather in Tokyo, and what is that temperature in Fahrenheit?"

    # --- Turn 1: User asks the question ---
    messages = [
        {"role": "user", "content": question}
    ]

    print(f"\nUser: {question}")
    print(f"\nTools available: {[t['name'] for t in TOOLS_ANTHROPIC_FORMAT]}")

    # --- Turn 2: Model decides to call get_weather ---
    # (In reality, the API would return this; we're scripting it)
    model_response_1 = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "I'll check the weather in Tokyo for you."
            },
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "get_weather",
                "input": {"city": "Tokyo", "unit": "celsius"}
            }
        ]
    }

    print(f"\nAssistant (turn 1): [{model_response_1['content'][0]['text']}]")
    print(f"Tool call: {model_response_1['content'][1]['name']}({model_response_1['content'][1]['input']})")

    # --- Execute the tool ---
    tool_result_1 = execute_tool_call(model_response_1["content"][1])
    print(f"Tool result: {tool_result_1}")

    # --- Turn 3: Send tool result back ---
    messages.append(model_response_1)
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01",
                "content": tool_result_1
            }
        ]
    })

    # --- Turn 4: Model now calls convert_units ---
    model_response_2 = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Now I'll convert 22°C to Fahrenheit."
            },
            {
                "type": "tool_use",
                "id": "toolu_02",
                "name": "convert_units",
                "input": {"value": 22, "from_unit": "celsius", "to_unit": "fahrenheit"}
            }
        ]
    }

    print(f"\nAssistant (turn 2): [{model_response_2['content'][0]['text']}]")
    print(f"Tool call: {model_response_2['content'][1]['name']}({model_response_2['content'][1]['input']})")

    tool_result_2 = execute_tool_call(model_response_2["content"][1])
    print(f"Tool result: {tool_result_2}")

    # --- Turn 5: Send result and get final answer ---
    messages.append(model_response_2)
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_02",
                "content": tool_result_2
            }
        ]
    })

    # --- Turn 6: Final answer ---
    final_answer = (
        f"The current weather in Tokyo is 22°C and cloudy. "
        f"In Fahrenheit, that is 71.6°F."
    )
    print(f"\nAssistant (final): {final_answer}")

    return final_answer


# ============================================================
# STEP 6: REAL Anthropic API call (if key available)
# ============================================================

def run_real_anthropic_agent(question: str):
    """
    Run a real agent using the Anthropic API with claude-haiku-4-5.

    This shows the ACTUAL API in action.
    Requires: ANTHROPIC_API_KEY environment variable
    """
    print("\n" + "="*60)
    print("REAL ANTHROPIC API AGENT")
    print("="*60)

    try:
        import anthropic
    except ImportError:
        print("anthropic package not installed. Run: uv add anthropic")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("No ANTHROPIC_API_KEY found. Set it to run this demo.")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        print("\nShowing simulated output instead:")
        simulate_anthropic_tool_use()
        return

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": question}]

    print(f"\nQuestion: {question}")
    print(f"Model: claude-haiku-4-5")
    print(f"Tools: {[t['name'] for t in TOOLS_ANTHROPIC_FORMAT]}\n")

    # Agentic loop — same structure as our hand-built one, but using the real API
    for step in range(10):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=TOOLS_ANTHROPIC_FORMAT,
            messages=messages,
        )

        print(f"[Step {step + 1}] stop_reason: {response.stop_reason}")

        # Append assistant's response to the conversation
        messages.append({"role": "assistant", "content": response.content})

        # Process content blocks
        tool_results = []
        for block in response.content:
            if block.type == "text":
                print(f"  Text: {block.text}")
            elif block.type == "tool_use":
                print(f"  Tool call: {block.name}({block.input})")
                result = execute_tool_call({"name": block.name, "input": block.input})
                print(f"  Tool result: {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        # If the model stopped for tool use, send results back
        if response.stop_reason == "tool_use":
            messages.append({"role": "user", "content": tool_results})
        else:
            # stop_reason is "end_turn" — we're done
            print("\n✓ Agent finished.")
            break


# ============================================================
# STEP 7: Compare old vs. new approach
# ============================================================

def compare_approaches():
    """Side-by-side comparison of text parsing vs. structured calling."""
    print("\n" + "="*60)
    print("COMPARISON: Text Parsing vs. Structured Function Calling")
    print("="*60)

    print("""
┌─────────────────────────┬──────────────────────────┬──────────────────────────┐
│ Aspect                  │ Text Parsing (Lessons 1-3)│ Structured Calling       │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Tool definition         │ Described in prompt text  │ JSON schema passed to API│
│ Tool call format        │ Action: tool(args)        │ {"type":"tool_use",...}  │
│ Parsing                 │ Regex                     │ json.loads() guaranteed  │
│ Args passing            │ Single string             │ Named dict (kwargs)      │
│ Reliability             │ Fragile — model can drift │ Model trained for format │
│ Multi-arg support       │ Manual parsing            │ Natural (JSON dict)      │
│ Works with GPT-2?       │ Yes (sort of)             │ No (needs training)      │
│ Error handling          │ Parse errors common       │ API validates schema     │
│ What's the same?        │ ← Same loop: call → result → next step →            │
└─────────────────────────┴──────────────────────────┴──────────────────────────┘

THE CORE INSIGHT: The agentic LOOP is identical in both approaches.
The only difference is how the model communicates its tool calls.

Both approaches:
  1. Build a context with the question and tools
  2. Model generates text / structured output
  3. We detect a tool call
  4. We run the tool
  5. We add the result to context
  6. Repeat until "Final Answer" / end_turn
""")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(__doc__)

    # --- Show Pydantic schema generation ---
    if PYDANTIC_AVAILABLE:
        print("\n" + "="*60)
        print("PYDANTIC SCHEMA GENERATION")
        print("="*60)
        schema = pydantic_to_tool_schema(
            "calculator",
            "Evaluate a math expression",
            CalculatorInput
        )
        print("Schema from Pydantic model:")
        print(json.dumps(schema, indent=2))

    # --- Show hand-written schemas ---
    print("\n" + "="*60)
    print("HAND-WRITTEN TOOL SCHEMA (Anthropic format)")
    print("="*60)
    print(json.dumps(TOOLS_ANTHROPIC_FORMAT[0], indent=2))

    # --- Show tool dispatch working ---
    print("\n" + "="*60)
    print("TOOL DISPATCH")
    print("="*60)
    test_calls = [
        {"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": "365 ** 0.5"}},
        {"type": "tool_use", "id": "t2", "name": "get_weather", "input": {"city": "London"}},
        {"type": "tool_use", "id": "t3", "name": "count_words", "input": {"text": "The quick brown fox jumps over the lazy dog"}},
        {"type": "tool_use", "id": "t4", "name": "convert_units", "input": {"value": 100, "from_unit": "km", "to_unit": "miles"}},
    ]
    for call in test_calls:
        result = execute_tool_call(call)
        print(f"  {call['name']}({call['input']}) → {result}")

    # --- Simulated multi-turn conversation ---
    simulate_anthropic_tool_use()

    # --- Real API (optional) ---
    run_real_anthropic_agent(
        "What is the weather in Tokyo, and how many words are in the phrase 'The quick brown fox'?"
    )

    # --- Comparison ---
    compare_approaches()

    print("\n" + "="*60)
    print("SUMMARY — What We Built")
    print("="*60)
    print("""
Key concepts from this lesson:

1. JSON SCHEMA IS THE STANDARD
   Anthropic and OpenAI both use JSON schema for tool definitions.
   The key fields: name, description, input_schema (properties, required).

2. THE STRUCTURED TOOL_USE BLOCK
   Instead of "Action: tool(args)", the model outputs:
   {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
   This is guaranteed parseable JSON — no regex needed.

3. THE TOOL_RESULT BLOCK
   After executing a tool, we send back:
   {"type": "tool_result", "tool_use_id": "...", "content": "result string"}
   The model reads this and decides what to do next.

4. PYDANTIC FOR SCHEMA GENERATION
   Instead of writing JSON schemas by hand, use Pydantic models.
   Python type hints → auto-generated JSON schema → pass to API.

5. SAME LOOP, BETTER PLUMBING
   Structured calling is just a cleaner version of what we built in
   Lessons 1-3. Same loop, same concepts, more reliable format.

6. WHAT COMES NEXT (Lesson 5)
   Build a coding agent with tools: read_file, write_file, run_python.
   Give it a task and watch it: write code → run → fix errors → done.
   This is the pattern behind Claude Code, Cursor, and GitHub Copilot.
""")
