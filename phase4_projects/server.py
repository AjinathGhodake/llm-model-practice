"""
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
        examples=["def square(n):\n    return n * n"],
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
      {"code": "def add(a, b):\n    return a + b"}

    Returns the generated docstring text.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    # Build the prompt in the same [INST] format used during training
    prompt = (
        f"[INST] Generate a Python docstring for this function:\n"
        f"{request.code} [/INST]\n"
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
