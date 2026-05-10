# =============================================================================
# Checkpoint Inspector — opens and explains every .pt file in the project
# =============================================================================

import torch
import os

def separator(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def format_size(path):
    return f"{os.path.getsize(path) / 1024:.1f} KB"

def print_tensor_table(state_dict):
    total_params = 0
    print(f"\n  {'Layer':<45} {'Shape':<25} {'Params':>10}")
    print(f"  {'-'*83}")
    for name, tensor in state_dict.items():
        params = tensor.numel()
        total_params += params
        print(f"  {name:<45} {str(list(tensor.shape)):<25} {params:>10,}")
    print(f"  {'-'*83}")
    print(f"  {'TOTAL PARAMETERS':<45} {'':25} {total_params:>10,}")
    return total_params


# =============================================================================
# 1. PHASE 1 — Bigram Model
# =============================================================================
separator("PHASE 1 — Bigram Language Model")

path = "phase1_foundations/bigram_model.pt"
print(f"\nFile: {path}  ({format_size(path)})")

ckpt = torch.load(path, map_location="cpu", weights_only=False)

print(f"\nKeys in checkpoint: {list(ckpt.keys())}")
print(f"\nVocab size:   {ckpt['vocab_size']}")
print(f"Block size:   {ckpt['block_size']}")
print(f"Train loss:   {ckpt['train_loss']:.4f}")
print(f"Val loss:     {ckpt['val_loss']:.4f}")

print(f"\nVocabulary ({len(ckpt['char_to_int'])} chars): "
      f"{repr(''.join(sorted(ckpt['char_to_int'].keys())))}")

print(f"\nWeight tensors:")
print_tensor_table(ckpt["model_state_dict"])

print(f"""
What this model is:
  A simple lookup table — for each character, look up the scores
  for what character likely comes next. That's the whole model.
  One matrix of shape ({ckpt['vocab_size']} × {ckpt['vocab_size']}).
""")


# =============================================================================
# 2. PHASE 2 — nanoGPT
# =============================================================================
separator("PHASE 2 — nanoGPT (Transformer)")

path = "phase2_transformers/nanogpt_quick.pt"
print(f"\nFile: {path}  ({format_size(path)})")

ckpt = torch.load(path, map_location="cpu", weights_only=False)

print(f"\nKeys in checkpoint: {list(ckpt.keys())}")
print(f"\nModel config:")
for k, v in ckpt["config"].items():
    print(f"  {k:<20} {v}")

# Handle both checkpoint formats (quick vs full nanogpt)
train_loss = ckpt.get("train_loss", "not saved")
val_loss   = ckpt.get("val_loss",   "not saved")
print(f"\nTrain loss:  {train_loss}")
print(f"Val loss:    {val_loss}")

cfg = ckpt["config"]
print(f"\nArchitecture summary:")
print(f"  embed_size  = {cfg['embed_size']}  → vector size for each token")
print(f"  num_heads   = {cfg['num_heads']}   → parallel attention heads")
print(f"  num_layers  = {cfg['num_layers']}   → transformer blocks stacked")
print(f"  block_size  = {cfg['block_size']}  → context window (tokens)")
print(f"  vocab_size  = {cfg['vocab_size']}  → unique characters")

# Handle both 'model_state_dict' and 'state' key names
state_dict = ckpt.get("model_state_dict") or ckpt.get("state")
print(f"\nWeight tensors:")
total = print_tensor_table(state_dict)

print(f"""
What this model is:
  A full GPT transformer with {cfg['num_layers']} stacked blocks.
  Each block = multi-head attention + feed-forward network.
  Trained on Tiny Shakespeare (~1M chars) for 1000 steps.
  Same architecture as GPT-2 — just {total:,}x smaller.
""")


# =============================================================================
# 3. PHASE 4 — LoRA Training Checkpoints
# =============================================================================
separator("PHASE 4 — LoRA Fine-tuning Checkpoints")

checkpoint_dir = "phase4_projects/gpt2-docstring-lora"
checkpoints = sorted([
    d for d in os.listdir(checkpoint_dir)
    if d.startswith("checkpoint-")
], key=lambda x: int(x.split("-")[1]))

print(f"\nFound {len(checkpoints)} training checkpoints in: {checkpoint_dir}/")
print(f"Checkpoints: {checkpoints}")

# Inspect the latest checkpoint
latest = checkpoints[-1]
latest_dir = os.path.join(checkpoint_dir, latest)

print(f"\n--- Inspecting latest: {latest} ---")

for fname in ["optimizer.pt", "scheduler.pt"]:
    fpath = os.path.join(latest_dir, fname)
    if not os.path.exists(fpath):
        continue

    size = format_size(fpath)
    data = torch.load(fpath, map_location="cpu", weights_only=False)

    print(f"\n{fname}  ({size})")
    print(f"  Type: {type(data)}")

    if fname == "optimizer.pt" and isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
        if "state" in data:
            n_params = len(data["state"])
            print(f"  Tracked parameter groups: {n_params}")
        if "param_groups" in data:
            pg = data["param_groups"][0]
            print(f"  Learning rate:  {pg.get('lr', 'N/A')}")
            print(f"  Weight decay:   {pg.get('weight_decay', 'N/A')}")
            print(f"  Betas:          {pg.get('betas', 'N/A')}")

    if fname == "scheduler.pt" and isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
        print(f"  Last epoch:      {data.get('last_epoch', 'N/A')}")
        print(f"  Last LR:         {data.get('_last_lr', 'N/A')}")

print(f"""
What these checkpoints are:
  HuggingFace Trainer saves optimizer + scheduler state every N steps.
  This lets you RESUME training from any checkpoint if it gets interrupted.
  optimizer.pt  = AdamW momentum buffers (learning history per weight)
  scheduler.pt  = LR schedule state (knows where in the warmup/decay curve)

  The actual model weights are stored separately as adapter_model.safetensors
  (LoRA adapters — the small matrices that were fine-tuned on top of GPT-2).
""")


# =============================================================================
# 4. QUICK COMPARISON
# =============================================================================
separator("COMPARISON — All Models at a Glance")

print(f"""
  {'Model':<30} {'File size':>10}  {'Description'}
  {'-'*70}""")

files = [
    ("phase1_foundations/bigram_model.pt",         "Phase 1 Bigram"),
    ("phase2_transformers/nanogpt_quick.pt",        "Phase 2 nanoGPT"),
    ("phase4_projects/gpt2-docstring-lora/checkpoint-85/optimizer.pt",
                                                    "Phase 4 Optimizer state"),
    ("phase4_projects/gpt2-docstring-lora/checkpoint-85/scheduler.pt",
                                                    "Phase 4 LR Scheduler"),
]

for fpath, label in files:
    if os.path.exists(fpath):
        size = format_size(fpath)
        print(f"  {label:<30} {size:>10}")

print(f"""
Key insight:
  Phase 1 bigram: tiny — it's just a 25×25 character probability table
  Phase 2 nanoGPT: small — full transformer, 211K learned parameters
  Phase 4 optimizer: large — AdamW stores 2 momentum values per weight
                     so optimizer state is often 2-3× the model size
""")
