"""
Phase 4, Lesson 4: Serving the Model as a REST API
====================================================

WHAT WE'RE BUILDING
--------------------
A real, running HTTP API that:
  - Loads our fine-tuned model once at startup
  - Accepts Python code via POST request
  - Returns a generated docstring
  - Has a health check endpoint

This is the FINAL form of an LLM project: something you can
call from any tool, script, or editor plugin.

HOW TO USE THIS FILE
--------------------
This file does two things:
  1. Defines and STARTS the FastAPI server (when run directly)
  2. Demonstrates how to TEST it with HTTP requests (using httpx)

Run the server:
  uv run phase4_projects/04_serve_and_deploy.py

Then in ANOTHER terminal, test it:
  curl -X POST http://localhost:8000/generate \\
    -H "Content-Type: application/json" \\
    -d '{"code": "def square(n):\\n    return n * n"}'
"""

# ============================================================
# PART 1: THE ARCHITECTURE EXPLAINED
# ============================================================

print("=" * 60)
print("PART 1: FastAPI Architecture")
print("=" * 60)

print("""
WHY FASTAPI?
  - Automatic API documentation at /docs
  - Request/response validation with Pydantic (built-in)
  - Async support for handling multiple requests
  - Very clean code, widely used in ML serving

REQUEST LIFECYCLE:
  1. Client sends POST /generate with JSON body
  2. FastAPI validates the JSON against our GenerateRequest schema
  3. Our handler calls the model to generate text
  4. We return a GenerateResponse JSON object

THE MODEL LOADING PATTERN:
  Models are expensive to load (seconds + memory).
  We load ONCE at startup, then reuse for all requests.
  This is called "model caching" or "singleton pattern."

  startup → load model → handle request → handle request → ...
                                  ↑ model is already in memory
""")

# ============================================================
# PART 2: IMPORTS AND SETUP
# ============================================================

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

print("=" * 60)
print("PART 2: Setting Up the Server")
print("=" * 60)

# Check that the fine-tuned adapter exists
ADAPTER_PATH = "phase4_projects/gpt2-docstring-lora"
if not Path(ADAPTER_PATH).exists():
    print(f"ERROR: Adapter not found at {ADAPTER_PATH}")
    print("Please run Lesson 3 first: uv run phase4_projects/03_train_and_evaluate.py")
    sys.exit(1)

print(f"Adapter found at: {ADAPTER_PATH}")

# ============================================================
# PART 3: THE FASTAPI APPLICATION
# ============================================================

print("\n" + "=" * 60)
print("PART 3: Building the FastAPI App")
print("=" * 60)

print("""
We're about to define the FastAPI app.
Key components:

  GenerateRequest  — Pydantic model for incoming JSON
  GenerateResponse — Pydantic model for outgoing JSON
  ModelState       — Global object holding the loaded model
  /health          — GET endpoint, confirms server is alive
  /generate        — POST endpoint, runs the model
  /docs            — Auto-generated interactive API docs (free!)
""")

# Write the server to a separate file so it can be run by uvicorn
SERVER_CODE = '''"""
FastAPI server for the Python Docstring Generation model.
Run with: uvicorn phase4_projects.server:app --reload
Or via: uv run phase4_projects/04_serve_and_deploy.py
"""
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────
BASE_MODEL = "gpt2"
ADAPTER_PATH = "phase4_projects/gpt2-docstring-lora"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ── Request / Response schemas ─────────────────────────────────
class GenerateRequest(BaseModel):
    """
    Input schema for the /generate endpoint.
    The 'code' field should contain a Python function definition.
    """
    code: str = Field(
        ...,
        description="Python function source code to generate a docstring for",
        min_length=10,
        max_length=2000,
        examples=["def square(n):\\n    return n * n"],
    )
    max_tokens: Optional[int] = Field(
        default=150,
        ge=10,
        le=500,
        description="Maximum number of tokens to generate",
    )


class GenerateResponse(BaseModel):
    """Output schema returned by /generate."""
    docstring: str = Field(description="Generated docstring text")
    model: str = Field(description="Model identifier used for generation")
    generation_time_ms: float = Field(description="Time taken to generate in milliseconds")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ── Model state (loaded once at startup) ──────────────────────
class ModelState:
    """
    Holds the loaded model and tokenizer.
    Using a class as a namespace avoids global variables.
    """
    model = None
    tokenizer = None
    loaded = False


state = ModelState()


# ── Lifespan: load model at startup ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Code BEFORE yield runs at startup.
    Code AFTER yield runs at shutdown.
    """
    print(f"Loading base model ({BASE_MODEL})...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    state.model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    state.model.eval()
    state.model = state.model.to(DEVICE)

    state.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    state.tokenizer.pad_token = state.tokenizer.eos_token

    state.loaded = True
    print(f"Model loaded on {DEVICE}. Ready to serve requests.")

    yield  # Server is now running

    # Cleanup on shutdown (free GPU memory)
    print("Shutting down. Freeing model memory...")
    del state.model
    del state.tokenizer


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="Python Docstring Generator",
    description="Fine-tuned GPT-2 with LoRA for generating Python docstrings",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check that the server is running and the model is loaded."""
    return HealthResponse(
        status="ok" if state.loaded else "loading",
        model_loaded=state.loaded,
        device=DEVICE,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_docstring(request: GenerateRequest):
    """
    Generate a Python docstring for the provided function code.

    Send a POST request with JSON body:
      {"code": "def add(a, b):\\n    return a + b"}

    Returns the generated docstring text.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    # Build the prompt in the same [INST] format used during training
    prompt = (
        f"[INST] Generate a Python docstring for this function:\\n"
        f"{request.code} [/INST]\\n"
    )

    # Tokenize
    inputs = state.tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Generate
    t0 = time.time()
    with torch.no_grad():
        outputs = state.model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=state.tokenizer.eos_token_id,
            eos_token_id=state.tokenizer.eos_token_id,
        )
    elapsed_ms = (time.time() - t0) * 1000

    # Decode only the new tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    docstring = state.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return GenerateResponse(
        docstring=docstring,
        model=f"{BASE_MODEL}+lora",
        generation_time_ms=round(elapsed_ms, 1),
    )
'''

server_path = Path("phase4_projects/server.py")
server_path.parent.mkdir(exist_ok=True)
server_path.write_text(SERVER_CODE)

# Also ensure there's an __init__.py so Python treats it as a package
init_path = Path("phase4_projects/__init__.py")
if not init_path.exists():
    init_path.write_text("")

print(f"Server code written to: {server_path}")
print(f"  Endpoints: GET /health, POST /generate, GET /docs")

# ============================================================
# PART 4: START THE SERVER AND RUN TESTS
# ============================================================

print("\n" + "=" * 60)
print("PART 4: Starting Server and Running Tests")
print("=" * 60)

print("""
We'll:
  1. Start the uvicorn server as a background process
  2. Wait for it to become ready (poll /health)
  3. Send test requests and show the responses
  4. Print instructions for using it yourself
""")

import httpx
import uvicorn

# Add project root to sys.path so the server module can be imported
project_root = str(Path(".").resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from phase4_projects.server import app  # noqa: E402


def run_server():
    """Run uvicorn in a thread using the already-imported app object."""
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="warning",  # suppress request logs for cleaner output
    )

# Start server in background thread
print("Starting server on http://127.0.0.1:8000 ...")
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Poll until ready
BASE_URL = "http://127.0.0.1:8000"
max_wait = 60
for i in range(max_wait):
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=2)
        if r.status_code == 200 and r.json().get("model_loaded"):
            print(f"Server ready after {i+1}s")
            break
    except Exception:
        pass
    time.sleep(1)
    if i == 0:
        print("  (waiting for model to load...)", end="", flush=True)
    else:
        print(".", end="", flush=True)
else:
    print("\nERROR: Server did not start in time.")
    sys.exit(1)
print()

# ── Health check ─────────────────────────────────────────────
print("\n--- Test 1: Health Check ---")
r = httpx.get(f"{BASE_URL}/health")
print(f"GET /health → {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test generation ───────────────────────────────────────────
TEST_CASES = [
    {
        "name": "Simple math function",
        "code": "def square(n):\n    return n * n",
    },
    {
        "name": "String utility",
        "code": "def is_palindrome(s):\n    return s == s[::-1]",
    },
    {
        "name": "List operation",
        "code": "def flatten(nested):\n    return [item for sublist in nested for item in sublist]",
    },
    {
        "name": "Never-seen function (real test of generalization)",
        "code": "def celsius_to_kelvin(c):\n    return c + 273.15",
    },
]

print("\n--- Test 2: Generation Requests ---\n")
for tc in TEST_CASES:
    print(f"  [{tc['name']}]")
    print(f"  Code: {tc['code'].split(chr(10))[0]}")

    r = httpx.post(
        f"{BASE_URL}/generate",
        json={"code": tc["code"]},
        timeout=60,
    )

    if r.status_code == 200:
        data = r.json()
        docstring_first_line = data["docstring"].split("\n")[0]
        print(f"  Docstring: {docstring_first_line}")
        print(f"  Time: {data['generation_time_ms']}ms")
    else:
        print(f"  ERROR: {r.status_code} — {r.text}")
    print()

# ── Test validation ───────────────────────────────────────────
print("--- Test 3: Input Validation ---")

r = httpx.post(f"{BASE_URL}/generate", json={"code": "hi"}, timeout=10)
print(f"  Short input (should fail): {r.status_code} — {r.json().get('detail', 'ok')[:80]}")

r = httpx.post(f"{BASE_URL}/generate", json={}, timeout=10)
print(f"  Missing 'code' field (should fail): {r.status_code}")

# ── Show docs URL ─────────────────────────────────────────────
print("\n--- API Documentation ---")
print(f"  Interactive docs: http://127.0.0.1:8000/docs")
print(f"  OpenAPI schema:   http://127.0.0.1:8000/openapi.json")

# ============================================================
# PART 5: HOW TO USE IT FROM THE COMMAND LINE
# ============================================================

print("\n" + "=" * 60)
print("PART 5: How to Use This API")
print("=" * 60)

print("""
While the server is running (or next time you start it), you can use:

FROM THE COMMAND LINE (curl):
  curl -X POST http://localhost:8000/generate \\
    -H "Content-Type: application/json" \\
    -d '{"code": "def add(a, b):\\n    return a + b"}'

FROM PYTHON (httpx or requests):
  import httpx
  r = httpx.post("http://localhost:8000/generate", json={
      "code": "def add(a, b):\\n    return a + b"
  })
  print(r.json()["docstring"])

FROM BROWSER:
  Open http://localhost:8000/docs for interactive Swagger UI

TO START THE SERVER STANDALONE:
  uv run -m uvicorn phase4_projects.server:app --reload

TO DEPLOY TO PRODUCTION (future learning):
  - Docker: wrap in a Dockerfile, deploy to any cloud
  - Railway / Render: push Git repo, automatic deployment
  - Modal / Replicate: specialized ML deployment platforms
""")

# ============================================================
# PART 6: PHASE 4 COMPLETE — WHAT YOU BUILT
# ============================================================

print("=" * 60)
print("PART 6: Phase 4 Complete!")
print("=" * 60)

print("""
YOU BUILT THE COMPLETE PIPELINE:

  Data                Training             Serving
  ─────────────────── ──────────────────── ──────────────────────
  84 examples         GPT-2 base model     FastAPI server
  JSONL format        LoRA (0.65% params)  POST /generate
  80/20 train/val     37 seconds           JSON response
  Rubric validation   Saved adapter        Auto-generated /docs

THE FOUR PHASES COMPLETE:

  Phase 1: Foundations
    ✓ Tensors, autograd, neural networks, language modeling

  Phase 2: Transformers from Scratch
    ✓ Embeddings, attention, transformer blocks, nanoGPT

  Phase 3: Fine-tuning
    ✓ HuggingFace, full fine-tune, LoRA, PEFT library

  Phase 4: Project Delivery
    ✓ Project planning, dataset building, training, serving API

WHAT YOU CAN DO NOW:
  1. Replace GPT-2 with Llama 3.2 3B for much better results
     (just change base_model in the config and target_modules)

  2. Build your own dataset for any task:
     - Code reviewer: input=code, output=review comments
     - SQL generator: input=question, output=SQL query
     - Email formatter: input=bullet points, output=email

  3. Add features to the API:
     - Authentication (API keys)
     - Rate limiting
     - Request logging / monitoring
     - Streaming responses (yield tokens as they generate)

  4. Deploy it:
     - Modal.com for serverless GPU inference
     - HuggingFace Spaces for free demos
     - Any cloud VM with Docker

You're a model-aware engineer now.
""")

print("Lesson complete! The lesson server has been shut down.")
print()
print("To run the server standalone (and keep it running):")
print("  uv run -m uvicorn phase4_projects.server:app --host 127.0.0.1 --port 8000")
print()
print("Then test it:")
print('  curl -X POST http://localhost:8000/generate \\')
print('    -H "Content-Type: application/json" \\')
print("    -d '{\"code\": \"def square(n):\\n    return n * n\"}'")
# The daemon thread will die when the main thread exits — that's correct for the lesson.
