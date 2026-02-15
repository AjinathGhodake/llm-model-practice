# =============================================================================
# PHASE 2 — LESSON 4: nanoGPT — Full GPT From Scratch
# =============================================================================
# Every concept from Phase 1 and Phase 2 assembled into one working model.
#
# Architecture (identical to GPT-2, just smaller):
#   Token + Positional Embeddings
#   → N × TransformerBlock [LN → MHA → residual → LN → FFN → residual]
#   → Final LayerNorm
#   → Linear LM Head (C → vocab_size)
#
# New additions vs previous lessons:
#   Dropout          → randomly zero activations during training (reduces overfitting)
#   Weight tying     → embedding table and LM head share weights (GPT-2 trick)
#   Real dataset     → Tiny Shakespeare (~1M characters)
#   Full training    → with eval loop, checkpointing, generation
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os
import time

# =============================================================================
# 1. CONFIG
# =============================================================================

class Config:
    # Model architecture
    embed_size  = 128        # dimension of token vectors flowing through the model
    num_heads   = 8          # parallel attention heads (head_size = 128/8 = 16)
    num_layers  = 6          # transformer blocks stacked
    block_size  = 256        # context window: tokens the model can see at once
    dropout     = 0.1        # fraction of activations zeroed during training

    # Training
    batch_size      = 64     # sequences per step
    max_steps       = 5000   # total training steps
    learning_rate   = 3e-4   # Adam learning rate (standard for transformers)
    eval_interval   = 500    # evaluate every N steps
    eval_steps      = 100    # batches to average for stable loss estimate
    train_split     = 0.9    # 90% train, 10% val

    # System
    device          = 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint_path = 'phase2_transformers/nanogpt.pt'
    data_path       = 'phase2_transformers/shakespeare.txt'
    seed            = 42

cfg = Config()
torch.manual_seed(cfg.seed)
print(f"Device: {cfg.device}")
print(f"Config: embed={cfg.embed_size}, heads={cfg.num_heads}, "
      f"layers={cfg.num_layers}, block={cfg.block_size}")


# =============================================================================
# 2. DATA — Download Tiny Shakespeare
# =============================================================================

print("\n" + "=" * 60)
print("DATA")
print("=" * 60)

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

if not os.path.exists(cfg.data_path):
    print("Downloading Tiny Shakespeare dataset...")
    urllib.request.urlretrieve(SHAKESPEARE_URL, cfg.data_path)
    print("Downloaded.")
else:
    print("Dataset already exists, skipping download.")

with open(cfg.data_path, 'r') as f:
    text = f.read()

# Build character-level vocabulary
chars         = sorted(set(text))
vocab_size    = len(chars)
char_to_int   = {ch: i for i, ch in enumerate(chars)}
int_to_char   = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [char_to_int[c] for c in s]
def decode(l): return ''.join([int_to_char[i] for i in l])

# Encode full dataset
data       = torch.tensor(encode(text), dtype=torch.long)
split_idx  = int(cfg.train_split * len(data))
train_data = data[:split_idx]
val_data   = data[split_idx:]

print(f"Text length:  {len(text):,} characters")
print(f"Vocab size:   {vocab_size} unique characters")
print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens:   {len(val_data):,}")

# Batch sampler
def get_batch(split):
    src = train_data if split == 'train' else val_data
    ix  = torch.randint(len(src) - cfg.block_size, (cfg.batch_size,))
    x   = torch.stack([src[i   : i+cfg.block_size  ] for i in ix])
    y   = torch.stack([src[i+1 : i+cfg.block_size+1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


# =============================================================================
# 3. MODEL — Full nanoGPT
# =============================================================================

class SingleHeadAttention(nn.Module):
    """One head of masked self-attention with dropout."""
    def __init__(self, embed_size, head_size, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.W_q = nn.Linear(embed_size, head_size, bias=False)
        self.W_k = nn.Linear(embed_size, head_size, bias=False)
        self.W_v = nn.Linear(embed_size, head_size, bias=False)
        # Dropout on attention weights — randomly ignore some token relationships
        # This prevents the model from over-relying on specific patterns
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores  = Q @ K.transpose(-2, -1) * (self.head_size ** -0.5)
        scores  = scores.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)   # drop some attention connections
        return weights @ V


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, embed_size, num_heads, block_size, dropout):
        super().__init__()
        self.head_size = embed_size // num_heads
        self.heads     = nn.ModuleList([
            SingleHeadAttention(embed_size, self.head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj    = nn.Linear(embed_size, embed_size)
        # Dropout after the output projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise FFN: expand to 4×, ReLU, project back."""
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One full transformer block with Pre-LN and residual connections."""
    def __init__(self, embed_size, num_heads, block_size, dropout):
        super().__init__()
        self.ln1          = nn.LayerNorm(embed_size)
        self.ln2          = nn.LayerNorm(embed_size)
        self.attention    = MultiHeadAttention(embed_size, num_heads, block_size, dropout)
        self.feed_forward = FeedForward(embed_size, dropout)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))     # attention sub-layer + residual
        x = x + self.feed_forward(self.ln2(x))  # FFN sub-layer + residual
        return x


class NanoGPT(nn.Module):
    """
    Full character-level GPT model.

    Architecture:
      Embedding (token + position)
      → N × TransformerBlock
      → LayerNorm
      → LM Head (linear projection to vocab_size)
    """
    def __init__(self, vocab_size, embed_size, num_heads,
                 block_size, num_layers, dropout):
        super().__init__()
        self.block_size = block_size

        # Embedding layers
        self.token_embedding    = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.embed_dropout      = nn.Dropout(dropout)

        # Transformer body
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])

        # Final normalization + language model head
        self.ln_final = nn.LayerNorm(embed_size)
        self.lm_head  = nn.Linear(embed_size, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and LM head.
        # Intuition: the embedding maps tokens → vectors,
        # the LM head maps vectors → token scores.
        # They are inverses of each other — sharing weights works well
        # and saves parameters. Used in GPT-2, GPT-3, and most LLMs.
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Standard GPT weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size

        # 1. Embed tokens + positions
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )
        x = self.embed_dropout(tok_emb + pos_emb)   # (B, T, embed_size)

        # 2. Transformer blocks
        x = self.blocks(x)                           # (B, T, embed_size)
        x = self.ln_final(x)                         # (B, T, embed_size)

        # 3. LM head → logits
        logits = self.lm_head(x)                     # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B*T, V),
                targets.view(B*T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive text generation.

        idx:            (B, T) starting token indices
        max_new_tokens: how many tokens to generate
        temperature:    >1 more random, <1 more focused
        top_k:          if set, only sample from the top k tokens
                        (improves quality by cutting off low-probability tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond   = idx[:, -self.block_size:]
            logits, _  = self(idx_cond)
            logits     = logits[:, -1, :] / temperature   # (B, vocab_size)

            # Top-k sampling: zero out all but top k logits
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx        = torch.cat([idx, next_token], dim=1)
        return idx


# Instantiate the model
model = NanoGPT(
    vocab_size  = vocab_size,
    embed_size  = cfg.embed_size,
    num_heads   = cfg.num_heads,
    block_size  = cfg.block_size,
    num_layers  = cfg.num_layers,
    dropout     = cfg.dropout,
).to(cfg.device)

total_params = sum(p.numel() for p in model.parameters())
# Weight tying means embedding and lm_head share weights — count uniquely
unique_params = sum(p.numel() for p in set(model.parameters()))

print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)
print(f"\nNanoGPT ready.")
print(f"  embed_size:  {cfg.embed_size}")
print(f"  num_heads:   {cfg.num_heads}  (head_size = {cfg.embed_size // cfg.num_heads})")
print(f"  num_layers:  {cfg.num_layers}")
print(f"  block_size:  {cfg.block_size}")
print(f"  vocab_size:  {vocab_size}")
print(f"  Parameters:  {total_params:,}  ({unique_params:,} unique due to weight tying)")


# =============================================================================
# 4. EVALUATION
# =============================================================================

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_steps)
        for k in range(cfg.eval_steps):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        results[split] = losses.mean().item()
    model.train()
    return results


# =============================================================================
# 5. TRAINING LOOP
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = cfg.learning_rate,
    weight_decay = 0.01   # L2 regularization — penalizes large weights
)
# AdamW vs Adam: AdamW fixes a subtle bug in weight decay implementation.
# It's the standard optimizer for training transformers.

# Expected initial loss = -log(1/vocab_size) = log(vocab_size)
expected_init_loss = torch.log(torch.tensor(float(vocab_size))).item()
print(f"\nExpected initial loss (random model): {expected_init_loss:.4f}")
print(f"Training for {cfg.max_steps:,} steps...\n")

history = {'step': [], 'train': [], 'val': []}
t_start = time.time()

for step in range(cfg.max_steps):

    # Evaluate periodically
    if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
        losses = estimate_loss()
        elapsed = time.time() - t_start
        history['step'].append(step)
        history['train'].append(losses['train'])
        history['val'].append(losses['val'])
        print(f"  step {step:5d} | train: {losses['train']:.4f} | "
              f"val: {losses['val']:.4f} | {elapsed:.0f}s elapsed")

    # Training step
    x, y          = get_batch('train')
    logits, loss  = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping: cap gradient norm at 1.0
    # Prevents occasional huge gradient steps from destabilizing training
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

total_time = time.time() - t_start
print(f"\nTraining complete. Total time: {total_time:.0f}s")


# =============================================================================
# 6. SAVE CHECKPOINT
# =============================================================================

checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': {
        'vocab_size':  vocab_size,
        'embed_size':  cfg.embed_size,
        'num_heads':   cfg.num_heads,
        'block_size':  cfg.block_size,
        'num_layers':  cfg.num_layers,
        'dropout':     0.0,   # disable dropout at inference
    },
    'char_to_int': char_to_int,
    'int_to_char': int_to_char,
    'train_loss':  history['train'][-1],
    'val_loss':    history['val'][-1],
}
torch.save(checkpoint, cfg.checkpoint_path)
size_mb = os.path.getsize(cfg.checkpoint_path) / (1024 * 1024)
print(f"\nCheckpoint saved: {cfg.checkpoint_path}  ({size_mb:.1f} MB)")


# =============================================================================
# 7. GENERATE TEXT
# =============================================================================

print("\n" + "=" * 60)
print("GENERATION")
print("=" * 60)

def generate_text(prompt, max_new_tokens=300, temperature=0.8, top_k=40):
    """Generate text from a text prompt."""
    model.eval()
    # Encode prompt — use only chars in our vocabulary
    valid_prompt = ''.join(c for c in prompt if c in char_to_int)
    if not valid_prompt:
        valid_prompt = '\n'
    tokens = torch.tensor([encode(valid_prompt)], dtype=torch.long).to(cfg.device)
    output = model.generate(tokens, max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(output[0].tolist())

# Generate from different prompts
prompts = [
    ("ROMEO:", 0.8, 40),
    ("To be or not to be", 0.8, 40),
    ("First Citizen:", 0.8, 40),
]

for prompt, temp, k in prompts:
    print(f"\nPrompt: '{prompt}'  (temp={temp}, top_k={k})")
    print("-" * 50)
    result = generate_text(prompt, max_new_tokens=250, temperature=temp, top_k=k)
    print(result)
    print()


# =============================================================================
# 8. COMPARE WITH BIGRAM
# =============================================================================

print("=" * 60)
print("COMPARISON: nanoGPT vs Bigram Model")
print("=" * 60)

bigram_ckpt_path = 'phase1_foundations/bigram_model.pt'

print(f"\n{'Metric':<25}  {'Bigram':>12}  {'nanoGPT':>12}")
print("-" * 52)
print(f"{'Parameters':<25}  {'625':>12}  {unique_params:>12,}")
print(f"{'Context window':<25}  {'1 token':>12}  {cfg.block_size:>11} tokens")
print(f"{'Architecture':<25}  {'Lookup table':>12}  {'Transformer':>12}")
print(f"{'Val loss (Phase 1)':<25}  {'~3.9':>12}  {history['val'][-1]:>12.4f}")
print(f"{'Coherent sentences':<25}  {'No':>12}  {'Yes':>12}")


# =============================================================================
# 9. LOSS CURVE
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

print(f"\n  {'Step':>6}  {'Train':>8}  {'Val':>8}  {'Gap':>8}")
print(f"  {'-'*36}")
for s, tl, vl in zip(history['step'], history['train'], history['val']):
    gap  = vl - tl
    flag = "  ← overfitting" if gap > 0.5 else ""
    print(f"  {s:6d}  {tl:8.4f}  {vl:8.4f}  {gap:+8.4f}{flag}")

print(f"""
Final train loss: {history['train'][-1]:.4f}
Final val loss:   {history['val'][-1]:.4f}

For reference — loss interpretation at character level:
  ~3.5  = random (untrained)
  ~2.5  = learned some patterns
  ~2.0  = decent character-level model
  ~1.5  = strong character-level model
  ~1.2  = near human-level for this task
""")


print("=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)
print(f"""
What you built across Phase 2:

  01_embeddings.py       → token + positional embeddings (B,T) → (B,T,C)
  02_attention.py        → self-attention Q,K,V, causal mask, multi-head
  03_transformer_block.py → residual, LayerNorm, FFN, full block
  04_nanogpt.py          → full GPT trained on Shakespeare

You now have a working GPT model that you built from scratch.
Every line of code — from tensor operations to attention weights —
you have seen, understood, and written yourself.

This is the same architecture as GPT-2.
The only difference between this and GPT-2 is scale:
  Your model:  {unique_params:,} parameters,  {cfg.block_size} context
  GPT-2 small: 117,000,000 parameters, 1024 context
  GPT-4:       ~1,000,000,000,000 parameters (estimated)

Same architecture. More data. More parameters. More compute.
That's it. You understand the whole thing.

What's next (Phase 3 — Fine-tuning):
  → Take a pretrained model (Llama, Mistral)
  → Fine-tune it on YOUR data with LoRA
  → Deliver a domain-specific model
""")
