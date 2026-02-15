# =============================================================================
# PHASE 2 — LESSON 2: Self-Attention
# =============================================================================
# The single mechanism that makes transformers work.
# Published in "Attention Is All You Need" (Vaswani et al., 2017).
#
# Core idea:
#   Each token looks at all previous tokens, decides which are relevant,
#   and updates its own representation by aggregating their information.
#
# We build it in 4 stages:
#   Stage 1: naive averaging (motivation — tokens need context)
#   Stage 2: weighted averaging with tril masking (the causal constraint)
#   Stage 3: learned weights via Q, K, V projections (real attention)
#   Stage 4: multi-head attention (run attention in parallel, multiple "views")
#
# Full formula: Attention(Q, K, V) = softmax( Q @ K.T / sqrt(head_size) ) @ V
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

B, T, C = 2, 6, 8   # batch=2, sequence_length=6, embed_size=8
x = torch.randn(B, T, C)   # pretend this is the embedding output

print("=" * 60)
print("PART 1: The Problem — Tokens Are Isolated")
print("=" * 60)
print(f"""
After embedding, we have x of shape {list(x.shape)}:
  B={B} batches, T={T} token positions, C={C} channels (embed_size)

Each x[b, t, :] is the vector for token t in batch b.
They are completely independent — token 3 knows nothing about tokens 0,1,2.

For a language model to be useful, each token needs context:
  - "bank" needs to see "river" or "savings" to know its meaning
  - "it" needs to find what noun "it" refers to
  - "not" needs to see what it's negating

Naive idea: let each token see the AVERAGE of all previous tokens.
This destroys information but shows the basic concept.
""")


print("=" * 60)
print("PART 2: Naive Context — Simple Averaging (Motivation)")
print("=" * 60)

# For each position t, compute the average of x[0..t] (all tokens up to t)
# This is the simplest way to give tokens context.
# Problem: averages destroy the specific signal. But the pattern is right.

# Method 1: explicit loop (slow but clear)
x_avg_loop = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        # Take mean of tokens 0..t (inclusive) — only PAST tokens, not future
        x_prev = x[b, :t+1]                    # (t+1, C)
        x_avg_loop[b, t] = x_prev.mean(dim=0)  # (C,)

# Method 2: matrix multiplication (fast, and reveals the attention pattern)
# Create a lower-triangular weight matrix (tril):
#   position 0 averages only token 0
#   position 1 averages tokens 0 and 1
#   position 2 averages tokens 0, 1, and 2
#   etc.
weights = torch.tril(torch.ones(T, T))  # lower triangle of 1s
print(f"\nLower triangular matrix (T={T}):")
print(weights)

# Normalize each ROW to sum to 1 — so it's a weighted average
weights = weights / weights.sum(dim=1, keepdim=True)
print(f"\nNormalized (each row sums to 1):")
print(weights.round(decimals=3))

# Apply to x: weights (T,T) @ x (B,T,C) → (B,T,C)
x_avg_matmul = weights @ x   # broadcast over batch

print(f"\nAre both methods identical? {torch.allclose(x_avg_loop, x_avg_matmul)}")
print(f"\nThis weight matrix IS the attention pattern — we'll make it LEARNED next.")
print(f"Key insight: tril ensures token t only sees tokens 0..t, NOT the future.")


print("\n" + "=" * 60)
print("PART 3: The Causal Mask — Why Tokens Cannot See the Future")
print("=" * 60)

# Language models are AUTOREGRESSIVE — they generate one token at a time.
# During training, we must prevent token t from seeing tokens t+1, t+2...
# If it could, it would just copy the answer — no learning happens.
#
# The mask: replace future positions with -infinity BEFORE softmax.
# -infinity → 0 after softmax → those positions contribute nothing.
#
# This is called "causal masking" or "autoregressive masking".

# Start with raw scores (will be Q@K.T later, for now just random)
raw_scores = torch.randn(T, T)
print(f"\nRaw scores (T×T):\n{raw_scores.round(decimals=2)}")

# Create the causal mask: True where we want to BLOCK (upper triangle)
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
print(f"\nCausal mask (True = blocked):\n{mask}")

# Apply mask: fill blocked positions with -inf
masked_scores = raw_scores.masked_fill(mask, float('-inf'))
print(f"\nAfter masking (-inf in future positions):\n{masked_scores.round(decimals=2)}")

# Softmax: -inf becomes 0, valid scores become probabilities
attn_weights = F.softmax(masked_scores, dim=-1)
print(f"\nAfter softmax (attention weights):\n{attn_weights.round(decimals=3)}")
print(f"\nRow sums (should all be 1.0): {attn_weights.sum(dim=-1).round(decimals=3)}")
print(f"Upper triangle is 0 — future tokens contribute nothing.")


print("\n" + "=" * 60)
print("PART 4: Queries, Keys, Values — Learned Attention")
print("=" * 60)

# The naive average used fixed equal weights.
# Real attention LEARNS which tokens to attend to, per token, per head.
#
# Three learned linear projections applied to x:
#
#   Query (Q): "What am I looking for?"
#              token t projects itself into a query vector
#
#   Key   (K): "What do I contain / what am I about?"
#              every token projects itself into a key vector
#
#   Value (V): "What information will I share if attended to?"
#              every token projects itself into a value vector
#
# Attention score between token i and token j:
#   score(i, j) = Q[i] · K[j]   ← dot product = "how well do they match?"
#
# High score = token i should pay a lot of attention to token j.
# These scores → softmax → weights → weighted sum of Values.

head_size = 4   # dimension of Q, K, V vectors (can be < C)
              # in GPT, head_size = embed_size / num_heads

# Three linear projections (no bias — standard in attention)
W_q = nn.Linear(C, head_size, bias=False)   # projects x → queries
W_k = nn.Linear(C, head_size, bias=False)   # projects x → keys
W_v = nn.Linear(C, head_size, bias=False)   # projects x → values

# Project x into Q, K, V spaces
Q = W_q(x)   # (B, T, head_size) — each token's query
K = W_k(x)   # (B, T, head_size) — each token's key
V = W_v(x)   # (B, T, head_size) — each token's value

print(f"\nInput x shape:  {x.shape}       (B={B}, T={T}, C={C})")
print(f"Q shape:        {Q.shape}   (B={B}, T={T}, head_size={head_size})")
print(f"K shape:        {K.shape}   same")
print(f"V shape:        {V.shape}   same")

# Compute attention scores: Q @ K^T
# For each batch: (T, head_size) @ (head_size, T) → (T, T)
# scores[t1, t2] = dot(Q[t1], K[t2]) = how much token t1 attends to token t2
scores = Q @ K.transpose(-2, -1)   # (B, T, T)
print(f"\nAttention scores shape: {scores.shape}   (B, T, T)")
print(f"\nScores for batch 0:\n{scores[0].detach().round(decimals=2)}")


print("\n" + "=" * 60)
print("PART 5: Scaling — Why Divide by sqrt(head_size)?")
print("=" * 60)

# As head_size grows, dot products get larger in magnitude.
# Large values push softmax into saturation (outputs near 0 or 1).
# Saturated softmax = vanishing gradients = training breaks.
#
# Fix: divide scores by sqrt(head_size) before softmax.
# This keeps the variance of the scores ~1 regardless of head_size.
#
# This is called "scaled dot-product attention".

scale = head_size ** -0.5   # = 1 / sqrt(head_size)
print(f"\nhead_size = {head_size}")
print(f"scale     = {scale:.4f}  (= 1/sqrt({head_size}))")

scores_unscaled = scores[0]
scores_scaled   = scores[0] * scale

print(f"\nBatch 0 scores — variance BEFORE scaling: {scores_unscaled.var().item():.4f}")
print(f"Batch 0 scores — variance AFTER  scaling: {scores_scaled.var().item():.4f}")
print(f"Scaling keeps variance ~1 — prevents softmax saturation.")


print("\n" + "=" * 60)
print("PART 6: Full Single-Head Attention — All Steps Together")
print("=" * 60)

# Now the complete formula:
#   Attention(Q, K, V) = softmax( Q @ K^T / sqrt(head_size) ) @ V
#
# Step by step:
#   1. scores  = Q @ K^T / sqrt(head_size)   → (B, T, T) raw scores
#   2. mask    = fill upper triangle with -inf → prevents seeing future
#   3. weights = softmax(scores, dim=-1)      → (B, T, T) attention weights
#   4. output  = weights @ V                  → (B, T, head_size) context vectors

class SingleHeadAttention(nn.Module):
    """One head of self-attention."""
    def __init__(self, embed_size, head_size, block_size):
        super().__init__()
        self.head_size = head_size

        # Learned projections
        self.W_q = nn.Linear(embed_size, head_size, bias=False)
        self.W_k = nn.Linear(embed_size, head_size, bias=False)
        self.W_v = nn.Linear(embed_size, head_size, bias=False)

        # Causal mask — registered as buffer (not a parameter, won't be trained)
        # tril is stored with the model but doesn't get gradients
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        # Project into Q, K, V
        Q = self.W_q(x)   # (B, T, head_size)
        K = self.W_k(x)   # (B, T, head_size)
        V = self.W_v(x)   # (B, T, head_size)

        # Scaled dot-product scores
        scores = Q @ K.transpose(-2, -1)          # (B, T, T)
        scores = scores * (self.head_size ** -0.5) # scale

        # Causal mask — block future positions
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,               # upper triangle = 0 in tril
            float('-inf')
        )

        # Softmax → attention weights
        weights = F.softmax(scores, dim=-1)        # (B, T, T)

        # Weighted sum of values
        output = weights @ V                       # (B, T, head_size)
        return output, weights


# Test it
head = SingleHeadAttention(embed_size=C, head_size=head_size, block_size=T)
out, attn_w = head(x)

print(f"\nInput  shape: {x.shape}       (B, T, embed_size)")
print(f"Output shape: {out.shape}   (B, T, head_size)")
print(f"\nAttention weights for batch 0 (rows=query tokens, cols=key tokens):")
print(attn_w[0].detach().round(decimals=3))
print(f"\nEach row = how much that token attends to previous tokens.")
print(f"Upper triangle = 0 (causal mask in effect).")
print(f"Row sums: {attn_w[0].sum(dim=-1).detach().round(decimals=3)}")


print("\n" + "=" * 60)
print("PART 7: Multi-Head Attention — Multiple Perspectives")
print("=" * 60)

# One attention head captures one type of relationship.
# But language has many simultaneous relationships:
#   Head 1 might track syntactic structure ("what is the subject?")
#   Head 2 might track coreference ("what does 'it' refer to?")
#   Head 3 might track positional proximity ("what is next to me?")
#
# Solution: run H attention heads IN PARALLEL, each with its own Q/K/V weights.
# Each head produces (B, T, head_size).
# Concatenate all heads: (B, T, H * head_size) = (B, T, embed_size).
# Project back to (B, T, embed_size) with a final linear layer.
#
# head_size = embed_size / num_heads  (standard convention)

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention.
    Runs num_heads attention heads in parallel, concatenates outputs.
    """
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        self.head_size = embed_size // num_heads   # size per head
        self.num_heads = num_heads

        # Create num_heads independent attention heads
        self.heads = nn.ModuleList([
            SingleHeadAttention(embed_size, self.head_size, block_size)
            for _ in range(num_heads)
        ])

        # Final projection: maps concatenated heads back to embed_size
        # This lets heads "talk to each other" after concatenation
        self.proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # Run all heads in parallel, collect outputs
        head_outputs = [h(x)[0] for h in self.heads]  # each: (B, T, head_size)

        # Concatenate along the channel dimension
        out = torch.cat(head_outputs, dim=-1)          # (B, T, embed_size)

        # Final projection
        out = self.proj(out)                           # (B, T, embed_size)
        return out


# Test multi-head
num_heads = 4
mha = MultiHeadAttention(embed_size=C, num_heads=num_heads, block_size=T)
mha_out = mha(x)

print(f"\nMultiHeadAttention(embed_size={C}, num_heads={num_heads}):")
print(f"  head_size per head: {C // num_heads}")
print(f"  Input shape:  {x.shape}       (B, T, embed_size)")
print(f"  Output shape: {mha_out.shape}       (B, T, embed_size)")
print(f"  Output shape SAME as input — ready to be passed to next layer")

total_params = sum(p.numel() for p in mha.parameters())
print(f"\nParameters in MultiHeadAttention: {total_params}")
print(f"  {num_heads} heads × 3 projections × ({C}×{C//num_heads}) = {num_heads * 3 * C * (C//num_heads)}")
print(f"  + output projection ({C}×{C}) = {C*C}")


print("\n" + "=" * 60)
print("PART 8: Visualizing What Attention Learns")
print("=" * 60)

# Attention weights show WHICH tokens each token is paying attention to.
# In a trained model you see interpretable patterns:
#   Noun → attending to its adjectives
#   Pronoun → attending to the noun it refers to
#   Verb → attending to its subject
#
# Right now (untrained) the weights are effectively random.
# After training, they become structured and interpretable.

head_solo = SingleHeadAttention(embed_size=C, head_size=head_size, block_size=T)
_, attn_weights = head_solo(x)

print(f"\nAttention weights — batch 0 (untrained, so effectively random):")
print(f"Rows = query tokens (asking), Cols = key tokens (being attended to)")
print()

# Make a pretty grid
header = "       " + "  ".join([f"tok{j}" for j in range(T)])
print(header)
for i in range(T):
    row = attn_weights[0, i].detach()
    bar = "  ".join([f"{v:.2f}" if j <= i else " --- " for j, v in enumerate(row)])
    print(f"  tok{i}  {bar}")

print(f"\n'---' = future tokens, blocked by causal mask")
print(f"Values = how much each token attends to past tokens (sum to 1 per row)")


print("\n" + "=" * 60)
print("PART 9: Complexity — Why Attention Is Expensive")
print("=" * 60)

# The Q@K^T operation creates a (T, T) matrix.
# Memory and compute scale as O(T^2).
# Doubling the context window = 4x the attention cost.
# This is why long-context models are expensive.

print(f"\nAttention complexity = O(T²) in memory and compute:")
print(f"\n  {'Context (T)':>15}  {'Matrix size (T×T)':>18}  {'Relative cost':>14}")
print(f"  {'-'*50}")
base = 512
for T_size in [512, 1024, 2048, 4096, 8192, 32768]:
    matrix = T_size * T_size
    relative = matrix / (base * base)
    print(f"  {T_size:>15,}  {matrix:>18,}  {relative:>13.0f}x")

print(f"\nGPT-2: 1024 context  → manageable on a single GPU")
print(f"GPT-4: ~32k context  → requires significant engineering")
print(f"Your M2: comfortably handles up to ~2k tokens")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
The full attention formula — memorize this:

  Attention(Q, K, V) = softmax( Q @ K^T / sqrt(head_size) ) @ V

Step by step:
  1. Q = W_q @ x       — each token asks: "what am I looking for?"
  2. K = W_k @ x       — each token says: "here is what I contain"
  3. scores = Q @ K^T  — how well does each query match each key?
  4. scores /= sqrt(d) — scale to prevent softmax saturation
  5. mask upper tril   — block future tokens (causal constraint)
  6. weights = softmax — convert scores to probabilities (sum to 1)
  7. output = weights @ V — weighted sum: "gather info from relevant tokens"

Multi-head = run H heads in parallel, concatenate, project back.
  Each head can specialize in different relationships.
  Output shape = input shape (B, T, embed_size) — same as input.

Key numbers:
  embed_size {C} = channels flowing through the model
  head_size  {head_size}  = embed_size / num_heads = dimension of Q, K, V
  num_heads  {num_heads}  = parallel attention heads
  block_size {T}  = max sequence length

Next: 03_transformer_block.py
      Wrap MultiHeadAttention + FeedForward into one reusable block.
      Stack N of these blocks = the full transformer.
""")
