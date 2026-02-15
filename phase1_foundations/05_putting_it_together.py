# =============================================================================
# LESSON 5: Putting It All Together — Clean Training Pipeline
# =============================================================================
# This file is the capstone of Phase 1.
# No new concepts — everything comes from lessons 1-4.
# Goal: organize it the way a real project is structured.
#
# What's added vs lesson 4:
#   - Train / validation split          (are we learning or memorizing?)
#   - Validation loss evaluation        (measured on data the model never trained on)
#   - Hyperparameters in one place      (easy to tune)
#   - Model checkpoint save / load      (persist trained weights)
#   - Clean inference function          (use the model after training)
#   - Detailed comments on each section (this becomes your Phase 2 template)
#
# Structure every language model training script will follow:
#   1. Config          — all hyperparameters in one place
#   2. Data            — load, tokenize, train/val split
#   3. Model           — nn.Module definition
#   4. Training loop   — forward, loss, backward, step, evaluate
#   5. Inference       — generate text from the trained model
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# =============================================================================
# 1. CONFIG — All hyperparameters in one place
# =============================================================================
# Keeping all settings here means you only change numbers in ONE place.
# When you read research papers, they list hyperparameters like this.

class Config:
    # Data
    train_split     = 0.9       # 90% train, 10% validation

    # Model architecture
    vocab_size      = None      # set after building vocabulary
    block_size      = 32        # context window: how many tokens the model sees at once

    # Training
    batch_size      = 32        # sequences processed per step
    max_steps       = 5000      # total training steps
    learning_rate   = 1e-2      # how big each gradient step is
    eval_interval   = 500       # evaluate on val set every N steps
    eval_steps      = 50        # how many batches to average for eval loss

    # System
    device          = 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint_path = 'phase1_foundations/bigram_model.pt'   # where to save weights
    seed            = 42

cfg = Config()
torch.manual_seed(cfg.seed)
print(f"Device: {cfg.device}")


# =============================================================================
# 2. DATA — Load, tokenize, split
# =============================================================================

TEXT = """to be or not to be that is the question
whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune
or to take arms against a sea of troubles
and by opposing end them to die to sleep
no more and by a sleep to say we end
the heartache and the thousand natural shocks
that flesh is heir to tis a consummation
devoutly to be wished to die to sleep
to sleep perchance to dream ay there is the rub
for in that sleep of death what dreams may come
when we have shuffled off this mortal coil
must give us pause there is the respect
that makes calamity of so long life"""

print("\n" + "=" * 60)
print("DATA")
print("=" * 60)

# --- Build vocabulary ---
chars         = sorted(set(TEXT))
cfg.vocab_size = len(chars)
char_to_int   = {ch: i for i, ch in enumerate(chars)}
int_to_char   = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    """Convert string to list of integers."""
    return [char_to_int[ch] for ch in text]

def decode(indices):
    """Convert list of integers back to string."""
    return ''.join([int_to_char[i] for i in indices])

# --- Encode full dataset ---
data = torch.tensor(encode(TEXT), dtype=torch.long)

# --- Train / validation split ---
# Never train and evaluate on the same data.
# Validation loss tells you if the model generalizes or is just memorizing.
split_idx  = int(cfg.train_split * len(data))
train_data = data[:split_idx]
val_data   = data[split_idx:]

print(f"Text:       {len(TEXT)} characters")
print(f"Vocab size: {cfg.vocab_size} unique characters")
print(f"Tokens:     {len(data)} total")
print(f"Train:      {len(train_data)} tokens ({cfg.train_split:.0%})")
print(f"Validation: {len(val_data)} tokens ({1-cfg.train_split:.0%})")


# =============================================================================
# 3. BATCH SAMPLER
# =============================================================================

def get_batch(split):
    """
    Sample a random batch of (input, target) pairs.
    split: 'train' or 'val'
    Returns tensors already moved to the correct device.
    """
    source = train_data if split == 'train' else val_data
    # Random starting positions — one per batch item
    ix = torch.randint(len(source) - cfg.block_size, (cfg.batch_size,))
    x  = torch.stack([source[i   : i+cfg.block_size  ] for i in ix])
    y  = torch.stack([source[i+1 : i+cfg.block_size+1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


# =============================================================================
# 4. MODEL
# =============================================================================
# Same BigramLanguageModel from lesson 4 — now with device awareness.
# In Phase 2 we'll swap this class out for a Transformer. Everything else stays.

class BigramLanguageModel(nn.Module):
    """
    Simplest possible language model.
    Each token independently predicts the next token via a lookup table.
    No memory of tokens before the current one.
    """
    def __init__(self, vocab_size):
        super().__init__()
        # Embedding table: row i = logits for what follows token i
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx:     (B, T) — token indices
        # targets: (B, T) — next token at each position (optional)
        logits = self.token_embedding(idx)          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B*T, C),    # (B*T, vocab_size)
                targets.view(B*T)       # (B*T,)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Autoregressively generate new tokens.

        idx:           (B, T) — starting context
        max_new_tokens: how many tokens to generate
        temperature:   >1 = more random, <1 = more focused (greedy at 0)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to block_size (model can't see further back)
            idx_cond = idx[:, -cfg.block_size:]

            logits, _ = self(idx_cond)

            # Last time step = next token prediction
            logits_last = logits[:, -1, :]           # (B, vocab_size)

            # Temperature scales the logits before softmax:
            #   temperature=1.0  → normal sampling
            #   temperature=0.5  → more confident, less variety
            #   temperature=2.0  → more random, more creative
            logits_last = logits_last / temperature

            probs      = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx        = torch.cat([idx, next_token], dim=1)

        return idx


# =============================================================================
# 5. EVALUATION
# =============================================================================

@torch.no_grad()
def estimate_loss(model):
    """
    Evaluate average loss on train and val splits.
    Uses multiple batches for a stable estimate.
    Switches model to eval mode, then back to train mode.
    """
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        split_losses = torch.zeros(cfg.eval_steps)
        for k in range(cfg.eval_steps):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean().item()
    model.train()
    return losses


# =============================================================================
# 6. TRAINING LOOP
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

model     = BigramLanguageModel(cfg.vocab_size).to(cfg.device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel:      BigramLanguageModel")
print(f"Parameters: {total_params:,}")
print(f"Device:     {cfg.device}")
print(f"Steps:      {cfg.max_steps:,}")
print(f"Batch size: {cfg.batch_size}")
print()

# Track losses for final summary
history = {'step': [], 'train_loss': [], 'val_loss': []}

for step in range(cfg.max_steps):

    # --- Evaluate periodically ---
    # Do this BEFORE the training step so we see the initial (untrained) loss too
    if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
        losses = estimate_loss(model)
        history['step'].append(step)
        history['train_loss'].append(losses['train'])
        history['val_loss'].append(losses['val'])
        print(f"  Step {step:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

    # --- Training step ---
    xb, yb = get_batch('train')             # sample batch

    logits, loss = model(xb, yb)            # forward pass

    optimizer.zero_grad()                   # clear previous gradients
    loss.backward()                         # compute gradients
    optimizer.step()                        # update weights


# =============================================================================
# 7. SAVE MODEL CHECKPOINT
# =============================================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

# torch.save saves the model's weights (state_dict) to disk.
# We also save config and vocab so the model can be fully restored later.
checkpoint = {
    'model_state_dict': model.state_dict(),     # the learned weights
    'char_to_int':      char_to_int,            # needed to encode new input
    'int_to_char':      int_to_char,            # needed to decode output
    'vocab_size':       cfg.vocab_size,
    'block_size':       cfg.block_size,
    'train_loss':       history['train_loss'][-1],
    'val_loss':         history['val_loss'][-1],
}
torch.save(checkpoint, cfg.checkpoint_path)
size_kb = os.path.getsize(cfg.checkpoint_path) / 1024
print(f"\nCheckpoint saved: {cfg.checkpoint_path}")
print(f"File size: {size_kb:.1f} KB")


# =============================================================================
# 8. LOAD MODEL & INFERENCE
# =============================================================================

print("\n" + "=" * 60)
print("LOADING MODEL & GENERATING TEXT")
print("=" * 60)

# Load the checkpoint
ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device, weights_only=False)

# Reconstruct the model from saved config
loaded_model = BigramLanguageModel(ckpt['vocab_size']).to(cfg.device)
loaded_model.load_state_dict(ckpt['model_state_dict'])

# Restore vocab
loaded_char_to_int = ckpt['char_to_int']
loaded_int_to_char = ckpt['int_to_char']

print(f"\nModel loaded from checkpoint")
print(f"  Vocab size:  {ckpt['vocab_size']}")
print(f"  Train loss:  {ckpt['train_loss']:.4f}")
print(f"  Val loss:    {ckpt['val_loss']:.4f}")

def generate_text(model, start_text, max_new_tokens=200, temperature=1.0):
    """Generate text starting from a prompt string."""
    model.eval()
    # Encode the starting prompt
    start_tokens = [loaded_char_to_int[ch] for ch in start_text if ch in loaded_char_to_int]
    idx = torch.tensor([start_tokens], dtype=torch.long).to(cfg.device)  # (1, len)

    # Generate
    output_idx = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)

    # Decode and return
    return ''.join([loaded_int_to_char[i] for i in output_idx[0].tolist()])

# Generate with different temperatures to see the effect
for temp in [0.5, 1.0, 1.5]:
    print(f"\n--- Temperature {temp} ---")
    text = generate_text(loaded_model, start_text="to ", max_new_tokens=120, temperature=temp)
    print(text)


# =============================================================================
# 9. LOSS CURVE SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

print(f"\nLoss over training:")
print(f"  {'Step':>6}  {'Train':>8}  {'Val':>8}  {'Gap':>8}")
print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
for s, tl, vl in zip(history['step'], history['train_loss'], history['val_loss']):
    gap = vl - tl
    # Gap = val_loss - train_loss
    # Small gap → model generalizes well
    # Large gap → model is overfitting (memorizing training data)
    flag = "  ← overfitting" if gap > 0.3 else ""
    print(f"  {s:6d}  {tl:8.4f}  {vl:8.4f}  {gap:+8.4f}{flag}")

print(f"""
Train loss: {history['train_loss'][-1]:.4f}
Val loss:   {history['val_loss'][-1]:.4f}
Gap:        {history['val_loss'][-1] - history['train_loss'][-1]:+.4f}

A small gap means the model generalizes.
A growing gap means it's memorizing — needs more data or regularization.
""")


print("=" * 60)
print("PHASE 1 COMPLETE")
print("=" * 60)
print("""
What you built across Phase 1:

  01_tensors.py            → tensors, shapes, devices, operations
  02_autograd.py           → gradients, backward pass, gradient descent
  03_neural_network.py     → layers, activations, nn.Module, optimizers
  04_language_model_intro  → tokenization, next-token prediction, bigram model
  05_putting_it_together   → clean pipeline, train/val split, save/load

You now understand the FULL training pipeline that powers every LLM.

The bigram model's limitation:
  It only sees 1 token of context.
  "to b" → the model doesn't know the 'b' follows 'to '.
  It only knows "given 'b', what comes next?"

Phase 2 fixes this with the TRANSFORMER:
  Self-attention lets every token look at ALL previous tokens.
  That context is why GPT can complete sentences coherently.

Phase 2 roadmap:
  01_tokenization_deep.py   → BPE tokenization (how GPT really tokenizes)
  02_embeddings.py          → token embeddings + positional embeddings
  03_attention.py           → self-attention mechanism from scratch
  04_transformer_block.py   → multi-head attention + feed-forward
  05_nanogpt.py             → full GPT assembled and trained
""")
