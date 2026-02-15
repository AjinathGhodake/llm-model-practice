"""
Phase 4, Lesson 1: Project Planning for LLM Applications
=========================================================

THE BIG PICTURE
---------------
You've learned:
  - How transformers work (Phase 2)
  - How to fine-tune them (Phase 3)

Now: how do you turn that into something useful?

This lesson covers the THINKING behind any LLM project:
  1. Problem framing — is fine-tuning even the right tool?
  2. Data strategy — what does "good training data" look like?
  3. Evaluation strategy — how do you know if it's working?
  4. Deployment planning — how does it get used?

OUR PROJECT: Python Code Documentation Assistant
-------------------------------------------------
Goal: Given a Python function (without a docstring), generate a docstring.

Example:
  Input:  def add(a, b): return a + b
  Output: "Add two numbers together. Returns their sum."

Why this is a good learning project:
  - Input/output are clear and structured
  - We can generate synthetic training data programmatically
  - We can objectively evaluate quality (does the docstring make sense?)
  - Small enough to train on a laptop
  - Useful enough to feel real
"""

# ============================================================
# PART 1: THE DECISION TREE — Fine-tune vs Prompt vs RAG
# ============================================================

print("=" * 60)
print("PART 1: Should You Fine-tune At All?")
print("=" * 60)

print("""
Before writing a single line of training code, ask:

OPTION A: Prompt Engineering
  - Just give GPT/Claude a good system prompt + examples
  - Works when: the task is general, data is small, speed matters
  - Fails when: very specific style, privacy requirements, offline use

OPTION B: Retrieval Augmented Generation (RAG)
  - Give the model access to a knowledge base at inference time
  - Works when: the task requires up-to-date or company-specific knowledge
  - Fails when: you need style/behavior change, not knowledge addition

OPTION C: Fine-tuning (what we're doing)
  - Train a model to behave differently
  - Works when: specific style/format, privacy needed, offline/cheap inference
  - Fails when: you don't have enough data, or the task is too general

OUR PROJECT: Fine-tuning is right because:
  ✓ We want a specific output FORMAT (docstring style)
  ✓ We want it to run OFFLINE on your laptop
  ✓ We can generate LOTS of training data programmatically
  ✓ Good learning exercise!
""")

# ============================================================
# PART 2: DATA STRATEGY
# ============================================================

print("=" * 60)
print("PART 2: Data Strategy")
print("=" * 60)

print("""
THE GOLDEN RULE: Your model will learn exactly what you show it.
If your data is bad → your model will be bad. Garbage in, garbage out.

For our project, we have THREE data options:

OPTION A: Real code from GitHub (best quality, hardest to get)
  - Python repos with docstrings already written
  - Filter for functions that have both code + docstring
  - Pros: Real-world style, diverse
  - Cons: License issues, data cleaning needed, slow to download

OPTION B: Synthetic data using a large model (what we'll do)
  - Use GPT/Claude to generate function+docstring pairs
  - Pros: Fast, controllable style, no licensing issues
  - Cons: Model's own style, not as diverse
  - This is how many fine-tuned models are actually built!

OPTION C: Build it manually (best for tiny datasets)
  - Write 50-200 examples by hand
  - Pros: Highest quality, exactly the style you want
  - Cons: Very slow, limits scale

WE'LL USE: A combination of B and C
  - ~50 hand-crafted examples for quality control
  - ~50 programmatically generated examples for variety
  - 100 examples total → enough to demonstrate the concept

TRAINING DATA FORMAT (what we decided in Phase 3):
  [INST] Generate a Python docstring for this function:
  def add(a, b):
      return a + b
  [/INST]
  Add two numbers and return the result.

  Args:
      a: First number.
      b: Second number.

  Returns:
      Sum of a and b.
""")

# ============================================================
# PART 3: EVALUATION STRATEGY
# ============================================================

print("=" * 60)
print("PART 3: How Will We Know If It Works?")
print("=" * 60)

print("""
EVALUATION IS HARD for generative models.
Unlike classification (right/wrong), generated text exists on a spectrum.

METRIC 1: Perplexity (automatic, imperfect)
  - Lower perplexity = model is less "surprised" by validation data
  - We track this during training (it's just exp(cross_entropy_loss))
  - Problem: A model that memorizes training data gets low perplexity

METRIC 2: BLEU Score (automatic, for reference-based evaluation)
  - Measures n-gram overlap between generated and reference text
  - Score 0-1, higher is better
  - Problem: Many valid docstrings exist for the same function

METRIC 3: Human Evaluation (best, most expensive)
  - Does the docstring accurately describe what the function does?
  - Is it in the right format (Args/Returns sections)?
  - Is it helpful?

WE'LL USE: A simple custom metric
  - Does the output contain "Args:" section? (if function has args)
  - Does the output contain "Returns:" section? (if function has return)
  - Is the first line a complete sentence?
  - Is output length reasonable (20-200 words)?

This is called a "rubric-based" evaluation. It's practical and fast.
""")

# ============================================================
# PART 4: DEPLOYMENT PLANNING
# ============================================================

print("=" * 60)
print("PART 4: Deployment Architecture")
print("=" * 60)

print("""
HOW WILL THIS BE USED?

The simplest useful deployment: a REST API
  - User sends a POST request with Python code
  - Server returns generated docstring
  - Works from any language, any tool, any editor

OUR STACK:
  FastAPI (Python web framework, very fast to build)
    ↓
  Pydantic (data validation — ensures input is valid)
    ↓
  Model (loaded once at startup, cached in memory)
    ↓
  JSON response

API DESIGN:
  POST /generate
  Request:  { "code": "def add(a, b):\\n    return a + b" }
  Response: { "docstring": "...", "model": "gpt2-lora-python", "time_ms": 234 }

  GET /health
  Response: { "status": "ok", "model_loaded": true }

PRODUCTION CONSIDERATIONS (we won't build these, but know they exist):
  - Rate limiting (don't let one user crush the server)
  - Authentication (who can use this?)
  - Batching (process multiple requests together for efficiency)
  - GPU inference server (TorchServe, TGI, vLLM for production scale)
  - Model versioning (how do you deploy a new version safely?)

FOR OUR PURPOSES: localhost FastAPI server is perfect.
""")

# ============================================================
# PART 5: THE COMPLETE PROJECT SPEC
# ============================================================

print("=" * 60)
print("PART 5: Project Specification Summary")
print("=" * 60)

PROJECT_SPEC = {
    "name": "Python Code Documentation Assistant",
    "base_model": "gpt2",
    "fine_tuning_method": "LoRA (r=8, alpha=16)",
    "target_modules": ["c_attn", "c_proj"],
    "dataset": {
        "total_examples": 100,
        "train_split": 80,
        "val_split": 20,
        "format": "[INST] ... [/INST] ...",
        "source": "hand-crafted + programmatic generation",
    },
    "training": {
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "hardware": "Apple M2 MPS",
        "estimated_time": "< 2 minutes",
    },
    "evaluation": {
        "primary": "validation loss",
        "secondary": "rubric-based checks (Args/Returns/sentence)",
        "test_set": "10 unseen functions",
    },
    "deployment": {
        "framework": "FastAPI",
        "interface": "REST API (POST /generate)",
        "serving": "localhost:8000",
    },
}

print("\nFinal project spec:")
for section, details in PROJECT_SPEC.items():
    print(f"\n  [{section.upper()}]")
    if isinstance(details, dict):
        for k, v in details.items():
            print(f"    {k}: {v}")
    else:
        print(f"    {details}")

# ============================================================
# PART 6: A TASTE OF WHAT THE DATA WILL LOOK LIKE
# ============================================================

print("\n" + "=" * 60)
print("PART 6: Sample Training Examples")
print("=" * 60)

# These are hand-crafted examples showing the format we'll use
# In Lesson 2, we'll build 100 of these

SAMPLE_EXAMPLES = [
    {
        "instruction": "Generate a Python docstring for this function:\ndef add(a, b):\n    return a + b",
        "output": """Add two numbers and return their sum.

Args:
    a: First number.
    b: Second number.

Returns:
    The sum of a and b.""",
    },
    {
        "instruction": "Generate a Python docstring for this function:\ndef is_even(n):\n    return n % 2 == 0",
        "output": """Check if a number is even.

Args:
    n: Integer to check.

Returns:
    True if n is even, False otherwise.""",
    },
    {
        "instruction": "Generate a Python docstring for this function:\ndef greet(name):\n    return f'Hello, {name}!'",
        "output": """Generate a greeting message for the given name.

Args:
    name: Name of the person to greet.

Returns:
    A greeting string in the format 'Hello, <name>!'""",
    },
]

print(f"\nWe'll have {len(SAMPLE_EXAMPLES)} sample examples (showing format):\n")
for i, ex in enumerate(SAMPLE_EXAMPLES, 1):
    print(f"Example {i}:")
    print(f"  INPUT:  {ex['instruction'][:60]}...")
    first_line = ex['output'].split('\n')[0]
    print(f"  OUTPUT: {first_line}")
    print()

# Show the full formatted prompt that goes to the model
print("Full formatted prompt (what the model actually sees):")
print("-" * 40)
ex = SAMPLE_EXAMPLES[0]
full_prompt = f"[INST] {ex['instruction']} [/INST]\n{ex['output']}"
print(full_prompt)
print("-" * 40)

# ============================================================
# PART 7: LESSON ROADMAP
# ============================================================

print("\n" + "=" * 60)
print("PART 7: Phase 4 Roadmap")
print("=" * 60)

print("""
01_project_planning.py   ← YOU ARE HERE
  What you learned: How to think about any LLM project
  Key concepts: fine-tune vs RAG vs prompt, data strategy, evaluation

02_build_dataset.py      ← NEXT
  What you'll do: Build 100 training examples for our docstring task
  Key concepts: data quality, train/val split, dataset serialization

03_train_and_evaluate.py ← AFTER THAT
  What you'll do: LoRA fine-tune + systematic evaluation
  Key concepts: training loop review, rubric evaluation, comparing outputs

04_serve_and_deploy.py   ← FINAL
  What you'll do: Build and run a real API that serves the model
  Key concepts: FastAPI, model loading, REST endpoints, real-world serving

After Phase 4, you'll have:
  ✓ A trained model (adapter weights in gpt2-lora-python/)
  ✓ A running API server (http://localhost:8000)
  ✓ The mental model for building ANY specialized LLM product
""")

print("Ready for Lesson 2: Building the dataset!")
print("Run: uv run phase4_projects/02_build_dataset.py")
