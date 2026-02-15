# LLM Model Practice - Project Instructions

## Purpose
Learning LLM model creation from scratch, step by step, with deep understanding of every concept.
Goal: Build small, specialized models and eventually deliver project-specific models.

## Hardware
- MacBook M2, 24GB unified RAM
- Use Apple MPS (Metal Performance Shaders) backend for PyTorch — NOT CUDA
- Always prefer MPS-optimized code when writing training scripts

## Learning Philosophy
- Explain EVERY step before writing code
- No black boxes — if something is abstracted, explain what's inside it
- Build from scratch first, then show how libraries do the same thing
- Each concept must be understood before moving to the next

## Learning Path (Phases)

### Phase 1 — Foundations
- Linear algebra basics (vectors, matrices, dot products)
- PyTorch fundamentals (tensors, autograd, training loop)
- Neural network basics (forward pass, backprop, loss)
- Status: COMPLETE ✅
  - 01_tensors.py           → tensors, shapes, devices, MPS
  - 02_autograd.py          → gradients, backward, gradient descent
  - 03_neural_network.py    → layers, activations, nn.Module, Adam
  - 04_language_model_intro → tokenization, next-token prediction, bigram model
  - 05_putting_it_together  → full pipeline, train/val split, save/load

### Phase 2 — Transformers from Scratch
- Tokenization (character-level → BPE)
- Embeddings
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Feed-forward layers
- Build nanoGPT from scratch (~300 lines)
- Status: COMPLETE ✅
  - 01_embeddings.py        → token + positional embeddings, (B,T)→(B,T,C)
  - 02_attention.py         → Q,K,V, causal mask, scaled dot-product, multi-head
  - 03_transformer_block.py → residual, LayerNorm, FFN, full stackable block
  - 04_nanogpt.py           → full GPT trained on Shakespeare (5000 steps)
  - 04_nanogpt_quick.py     → quick 1000-step demo version

### Phase 3 — Fine-tuning
- Hugging Face Transformers
- LoRA / QLoRA
- Training on custom datasets
- Status: COMPLETE ✅
  - 01_huggingface_intro.py    → HF ecosystem, BPE tokenizer, GPT-2 inference
  - 02_finetuning_basics.py    → full fine-tune, dataset prep, weight diff
  - 03_lora_explained.py       → LoRA math, scratch implementation, PEFT
  - 04_finetune_with_lora.py   → complete LoRA pipeline, adapter save/load/merge

### Phase 4 — Project Delivery
- Domain-specific fine-tuned models
- Wrapping models in APIs
- Status: COMPLETE ✅
  - 01_project_planning.py    → fine-tune vs RAG vs prompt, data strategy, evaluation
  - 02_build_dataset.py       → 84 examples, validation, JSONL format, train/val split
  - 03_train_and_evaluate.py  → LoRA fine-tune (37s), rubric evaluation, before/after comparison
  - 04_serve_and_deploy.py    → FastAPI REST API, /generate + /health endpoints
  - server.py                 → standalone uvicorn server (run independently)

### Phase 5 — Agents & Tool Use
- ReAct (Reason + Act) pattern
- Tool definition with JSON schemas
- Tool dispatch and the agentic loop
- Structured function calling (Anthropic/OpenAI API format)
- Building a coding agent
- Status: COMPLETE ✅
  - 01_llm_as_reasoner.py        → agent loop, ReAct pattern, GPT-2 limitations
  - 02_tool_definition.py        → JSON schemas, tool registry, text vs structured parsing
  - 03_react_agent.py            → full ReAct loop, multi-hop questions, knowledge base search
  - 04_function_calling_pattern.py → Anthropic tool_use format, Pydantic schemas, real API
  - 05_build_a_coding_agent.py   → read/write/run tools, self-repair loop, the Claude Code pattern

## Code Style Preferences
- Python only
- Use `uv` for package management (not pip or conda)
- Keep files small and focused — one concept per file
- Heavy inline comments explaining what each line does
- Prefer explicit code over clever one-liners

## Project Structure Convention
```
llm_model_practice/
├── CLAUDE.md
├── phase1_foundations/
│   ├── 01_tensors.py
│   ├── 02_autograd.py
│   └── ...
├── phase2_transformers/
│   ├── 01_tokenization.py
│   ├── 02_embeddings.py
│   ├── 03_attention.py
│   └── ...
├── phase3_finetuning/
├── phase4_projects/
└── phase5_agents/
```

## Session Rules
- Always start by explaining the concept in plain English before any code
- After each file, summarize what was learned and what comes next
- If I ask "why", stop and explain deeply before continuing
- Never skip steps — if something depends on a previous concept, make sure it was covered
- Use analogies to explain complex math concepts

## Key Resources (for reference)
- Andrej Karpathy - "Neural Networks: Zero to Hero" (YouTube)
- github.com/karpathy/nanoGPT
- fast.ai Practical Deep Learning
- Hugging Face documentation

## Current Focus
ALL FIVE PHASES COMPLETE. The full pipeline: tensors → transformers → fine-tuning → serving → agents.
