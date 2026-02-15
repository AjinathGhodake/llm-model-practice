# LLM Model Practice — From Tensors to Agents

A complete, from-scratch learning path for understanding large language models.
Every concept is explained before it's coded. No black boxes.

## What This Covers

### Phase 1 — Foundations
PyTorch fundamentals on Apple MPS (M-series Mac).

| File | Topic |
|------|-------|
| `01_tensors.py` | Tensors, shapes, devices, MPS acceleration |
| `02_autograd.py` | Gradients, backward pass, gradient descent |
| `03_neural_network.py` | Layers, activations, `nn.Module`, Adam optimizer |
| `04_language_model_intro/` | Tokenization, next-token prediction, bigram model |
| `05_putting_it_together/` | Full pipeline: train/val split, save/load |

### Phase 2 — Transformers from Scratch
Build GPT from the ground up, one component at a time.

| File | Topic |
|------|-------|
| `01_embeddings.py` | Token + positional embeddings, `(B,T)→(B,T,C)` |
| `02_attention.py` | Q, K, V matrices, causal mask, scaled dot-product, multi-head |
| `03_transformer_block.py` | Residual connections, LayerNorm, FFN, stackable block |
| `04_nanogpt.py` | Full GPT trained on Shakespeare (5000 steps) |

### Phase 3 — Fine-tuning
Adapt pre-trained models to new tasks efficiently.

| File | Topic |
|------|-------|
| `01_huggingface_intro.py` | HF ecosystem, BPE tokenizer, GPT-2 inference |
| `02_finetuning_basics.py` | Full fine-tune, dataset prep, weight diff analysis |
| `03_lora_explained.py` | LoRA math from scratch, then with PEFT |
| `04_finetune_with_lora.py` | Complete LoRA pipeline, adapter save/load/merge |

### Phase 4 — Project Delivery
Build and serve a domain-specific fine-tuned model.

| File | Topic |
|------|-------|
| `01_project_planning.py` | Fine-tune vs RAG vs prompting, data strategy |
| `02_build_dataset.py` | 84 examples, validation, JSONL format, train/val split |
| `03_train_and_evaluate.py` | LoRA fine-tune, rubric evaluation, before/after comparison |
| `04_serve_and_deploy.py` | FastAPI REST API with `/generate` + `/health` endpoints |
| `server.py` | Standalone uvicorn server |

### Phase 5 — Agents & Tool Use
How LLMs call tools, reason in loops, and act autonomously.

| File | Topic |
|------|-------|
| `01_llm_as_reasoner.py` | Agent loop mechanics, ReAct pattern, GPT-2 limitations |
| `02_tool_definition.py` | JSON schemas, tool registry, text vs structured dispatch |
| `03_react_agent.py` | Full ReAct loop, multi-hop questions, knowledge base search |
| `04_function_calling_pattern.py` | Anthropic `tool_use` format, Pydantic schemas, real API |
| `05_build_a_coding_agent.py` | Read/write/run tools, self-repair loop — the Claude Code pattern |

## Key Concepts Learned

- How attention and transformers work mathematically
- Why LoRA is efficient (only trains low-rank weight updates)
- The ReAct reasoning pattern (Thought → Action → Observation)
- How function calling works (JSON schema → structured dispatch)
- Why GPT-2 can't follow instructions (not instruction-tuned)
- What Claude Code, Cursor, and Devin are doing under the hood

## Hardware

Built and tested on **MacBook M2, 24GB unified RAM**.
All training uses Apple MPS (Metal Performance Shaders) — no CUDA needed.

## How to Run

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run any lesson
uv run phase1_foundations/01_tensors.py
uv run phase2_transformers/04_nanogpt.py
uv run phase5_agents/03_react_agent.py
```

## Learning Path

```
Tensors → Autograd → Neural Nets → Language Models
  → Embeddings → Attention → nanoGPT
    → HuggingFace → LoRA → Fine-tuning
      → Dataset → Training → FastAPI
        → ReAct → Tool Use → Coding Agent
```

Inspired by [Andrej Karpathy's Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0) series.
