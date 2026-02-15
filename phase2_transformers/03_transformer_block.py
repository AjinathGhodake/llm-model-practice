# =============================================================================
# PHASE 2 — LESSON 3: The Transformer Block
# =============================================================================
# One transformer block = the repeating unit of every GPT model.
# Stack N of them = the full transformer.
#
# A single block contains:
#   1. Layer Normalization    → stabilize activations before each sub-layer
#   2. Multi-Head Attention   → tokens communicate with each other
#   3. Residual connection    → add input back to output (skip connection)
#   4. Layer Normalization    → stabilize again
#   5. Feed-Forward Network   → each token "thinks" about what it gathered
#   6. Residual connection    → add input back to output again
#
# Formula:
#   x = x + MultiHeadAttention(LayerNorm(x))
#   x = x + FeedForward(LayerNorm(x))
#
# This file builds each piece from scratch, explains why each exists,
# then assembles them into a stackable block.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Configuration we'll use throughout
B          = 2    # batch size
T          = 8    # sequence length (block_size)
C          = 32   # embed_size (channels)
num_heads  = 4    # attention heads
head_size  = C // num_heads   # = 8 per head


print("=" * 60)
print("PART 1: Residual Connections — Solving the Depth Problem")
print("=" * 60)

# Problem: deep networks suffer from vanishing gradients.
# By the time gradients flow back from layer 10 to layer 1,
# they've been multiplied together so many times they become ~0.
# Layer 1 barely learns anything.
#
# Solution: residual (skip) connections, introduced in ResNet (2015).
#
#   Without residual:  output = Layer(x)
#   With residual:     output = x + Layer(x)
#
# The input x is added DIRECTLY to the layer's output.
# This creates a "highway" for gradients to flow back unobstructed.
# Even if Layer(x) has vanishing gradients, the gradient of x is just 1.
#
# In a transformer: EVERY sub-layer uses a residual connection.

print("\nWithout residual (gradient must pass through every layer):")
print("  Layer1 → Layer2 → Layer3 → ... → LayerN → Loss")
print("  Gradient: dL/dLayer1 = tiny product of N Jacobians")
print("  Result:   early layers barely learn")

print("\nWith residual connections:")
print("  x → [x + Layer1(x)] → [x + Layer2(x)] → ... → Loss")
print("  Gradient: dL/dx has a direct path (derivative of x+f(x) = 1 + df/dx)")
print("  Result:   every layer learns effectively")

# Simple demonstration: residual vs no-residual gradient magnitude
torch.manual_seed(0)

class DeepNoResidual(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(depth)])
    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))  # no residual
        return x.sum()

class DeepWithResidual(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(depth)])
    def forward(self, x):
        for layer in self.layers:
            x = x + torch.tanh(layer(x))  # residual: add input back
        return x.sum()

x_test = torch.randn(4, 8, requires_grad=True)
depth = 12

# No residual
model_no_res = DeepNoResidual(depth)
loss_no_res = model_no_res(x_test.clone().detach().requires_grad_(True))
loss_no_res.backward()
grad_no_res = x_test.grad.abs().mean().item() if x_test.grad else 0

x_res = torch.randn(4, 8, requires_grad=True)
model_res = DeepWithResidual(depth)
loss_res = model_res(x_res)
loss_res.backward()
grad_res = x_res.grad.abs().mean().item()

print(f"\nGradient magnitude at input (depth={depth}):")
print(f"  Without residual: {grad_no_res:.8f}  ← nearly vanished")
print(f"  With residual:    {grad_res:.8f}  ← healthy")


print("\n" + "=" * 60)
print("PART 2: Layer Normalization — Stabilizing Activations")
print("=" * 60)

# Problem: as data flows through many layers, the scale of activations
# can grow or shrink dramatically. This makes training unstable.
#
# Layer Normalization: for each token vector, normalize to mean=0, std=1,
# then scale and shift with learned parameters gamma and beta.
#
# Applied BEFORE each sub-layer (this is "Pre-LN" — the modern standard).
# GPT-2 and nanoGPT both use Pre-LN.
#
# LayerNorm operates on the LAST dimension (the channel dimension C).
# Each token is normalized independently.
# This is different from BatchNorm which normalizes across the batch.

ln = nn.LayerNorm(C)

print(f"\nLayerNorm(C={C}):")
print(f"  Learnable parameters: gamma (scale) + beta (shift), each shape ({C},)")
print(f"  Total parameters: {sum(p.numel() for p in ln.parameters())}")

# Show what LayerNorm does to activations
x_bad = torch.randn(B, T, C) * 10 + 5   # badly scaled: mean=5, std=10
print(f"\nBefore LayerNorm (token 0, batch 0):")
print(f"  mean = {x_bad[0,0].mean().item():.4f}")
print(f"  std  = {x_bad[0,0].std().item():.4f}")

x_normed = ln(x_bad)
print(f"\nAfter LayerNorm (token 0, batch 0):")
print(f"  mean = {x_normed[0,0].mean().item():.4f}  (≈ 0)")
print(f"  std  = {x_normed[0,0].std().item():.4f}  (≈ 1)")
print(f"\nEach token's vector is normalized independently.")
print(f"gamma/beta let the model rescale/shift after normalization.")


print("\n" + "=" * 60)
print("PART 3: Feed-Forward Network — Token-Level Thinking")
print("=" * 60)

# After attention, tokens have gathered context from each other.
# But what do they DO with that context?
#
# The Feed-Forward Network (FFN) processes each token INDEPENDENTLY.
# It doesn't mix tokens — each token's vector goes through the same 2-layer MLP.
#
# Structure:
#   Linear(C → 4C) → ReLU → Linear(4C → C)
#
# The 4× expansion is the standard from "Attention Is All You Need".
# This larger hidden layer gives the model capacity to store knowledge.
# Interestingly, research suggests the FFN layers act as "key-value memories"
# — they store factual associations learned during training.
#
# In GPT-2 small (C=768):
#   FFN hidden = 768 × 4 = 3072 neurons
#   FFN params = 768×3072 + 3072×768 = ~4.7M per layer
#   Attention params = ~2.4M per layer
#   → FFN has ~2x more parameters than attention

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Applied to each token independently and identically.
    """
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),   # expand to 4×
            nn.ReLU(),                                 # non-linearity
            nn.Linear(4 * embed_size, embed_size),    # project back down
        )

    def forward(self, x):
        # x: (B, T, C) — same linear layers applied to every token
        return self.net(x)    # (B, T, C) — same shape out


ffn = FeedForward(embed_size=C)
x_test = torch.randn(B, T, C)
ffn_out = ffn(x_test)

ffn_params = sum(p.numel() for p in ffn.parameters())
print(f"\nFeedForward(embed_size={C}):")
print(f"  Layer 1: {C} → {4*C}  ({C * 4*C + 4*C:,} params)")
print(f"  ReLU")
print(f"  Layer 2: {4*C} → {C}  ({4*C * C + C:,} params)")
print(f"  Total: {ffn_params:,} parameters")
print(f"\n  Input:  {x_test.shape}   (B, T, C)")
print(f"  Output: {ffn_out.shape}   (B, T, C) — same shape")
print(f"\nEach of the {T} tokens goes through the SAME FFN weights.")
print(f"No mixing between tokens here — attention handles that.")


print("\n" + "=" * 60)
print("PART 4: The Full Transformer Block")
print("=" * 60)

# Now we assemble everything into one block.
# The exact order matters — this is "Pre-LayerNorm" (Pre-LN):
#
#   x = x + Attention(LayerNorm(x))    ← attention sub-layer
#   x = x + FeedForward(LayerNorm(x))  ← feed-forward sub-layer
#
# Note: LayerNorm is applied to x BEFORE passing to each sub-layer.
# The residual adds the ORIGINAL x back after the sub-layer.
#
# Why Pre-LN instead of Post-LN?
#   Post-LN (original paper): LayerNorm after the residual — unstable for deep models
#   Pre-LN  (GPT-2 onwards):  LayerNorm before — more stable, easier to train deep
#
# The block is the fundamental unit. GPT-2 small stacks 12 of these.
# GPT-3 stacks 96. Each one refines the token representations further.

# Reuse the SingleHeadAttention and MultiHeadAttention from lesson 2
# (copied here so this file is self-contained)

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_size, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.W_q = nn.Linear(embed_size, head_size, bias=False)
        self.W_k = nn.Linear(embed_size, head_size, bias=False)
        self.W_v = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores  = Q @ K.transpose(-2, -1) * (self.head_size ** -0.5)
        scores  = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        self.head_size = embed_size // num_heads
        self.heads     = nn.ModuleList([
            SingleHeadAttention(embed_size, self.head_size, block_size)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """
    One transformer block = the repeating unit of GPT.

    Data flow:
      x → LayerNorm → MultiHeadAttention → + x  (attention sub-layer)
      x → LayerNorm → FeedForward        → + x  (ffn sub-layer)
    """
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()

        # Layer norms — one before each sub-layer (Pre-LN style)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

        # The two sub-layers
        self.attention   = MultiHeadAttention(embed_size, num_heads, block_size)
        self.feed_forward = FeedForward(embed_size)

    def forward(self, x):
        # Attention sub-layer with residual
        # Note: normalize FIRST (Pre-LN), then attend, then add residual
        x = x + self.attention(self.ln1(x))    # (B, T, C)

        # Feed-forward sub-layer with residual
        x = x + self.feed_forward(self.ln2(x)) # (B, T, C)

        return x   # (B, T, C) — same shape as input


block = TransformerBlock(embed_size=C, num_heads=num_heads, block_size=T)
x_in  = torch.randn(B, T, C)
x_out = block(x_in)

block_params = sum(p.numel() for p in block.parameters())
print(f"\nTransformerBlock(embed_size={C}, num_heads={num_heads}):")
print(f"  Input shape:  {x_in.shape}")
print(f"  Output shape: {x_out.shape}  ← same shape as input (stackable!)")
print(f"  Parameters:   {block_params:,}")

# Break down where params come from
attn_params = sum(p.numel() for p in block.attention.parameters())
ffn_params  = sum(p.numel() for p in block.feed_forward.parameters())
ln_params   = sum(p.numel() for p in block.ln1.parameters()) + \
              sum(p.numel() for p in block.ln2.parameters())
print(f"\n  Parameter breakdown:")
print(f"    MultiHeadAttention: {attn_params:>6,}  ({attn_params/block_params:.0%})")
print(f"    FeedForward:        {ffn_params:>6,}  ({ffn_params/block_params:.0%})")
print(f"    LayerNorms:         {ln_params:>6,}  ({ln_params/block_params:.0%})")


print("\n" + "=" * 60)
print("PART 5: Stacking Blocks — Building a Deep Transformer")
print("=" * 60)

# The transformer body = N identical blocks stacked sequentially.
# Each block refines the token representations.
# Early blocks might handle syntax, later blocks might handle semantics.
#
# Important: input and output shapes are IDENTICAL for each block (B, T, C).
# This is what makes stacking trivial — just loop through blocks.

class TransformerBody(nn.Module):
    """N stacked transformer blocks."""
    def __init__(self, embed_size, num_heads, block_size, num_layers):
        super().__init__()
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads, block_size)
            for _ in range(num_layers)
        ])
        # Final LayerNorm after all blocks (standard in GPT)
        self.ln_final = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = self.blocks(x)      # pass through all N blocks
        x = self.ln_final(x)    # final normalization
        return x


print(f"\nStacking transformer blocks (embed_size={C}, heads={num_heads}):\n")
print(f"  {'Layers':>8}  {'Parameters':>12}  {'Depth analogy'}")
print(f"  {'-'*50}")

for n_layers in [1, 2, 4, 6, 12]:
    body = TransformerBody(C, num_heads, T, n_layers)
    params = sum(p.numel() for p in body.parameters())
    if n_layers == 6:
        analogy = "← our nanoGPT will use ~6"
    elif n_layers == 12:
        analogy = "← GPT-2 small uses 12"
    else:
        analogy = ""
    print(f"  {n_layers:>8}  {params:>12,}  {analogy}")

# Test the body
body = TransformerBody(C, num_heads, T, num_layers=4)
x_in  = torch.randn(B, T, C)
x_out = body(x_in)
print(f"\nTransformerBody (4 layers):")
print(f"  Input:  {x_in.shape}")
print(f"  Output: {x_out.shape}  ← same shape, but each token has now attended")
print(f"          to all previous tokens through 4 rounds of refinement.")


print("\n" + "=" * 60)
print("PART 6: The Complete Picture — Data Flow Through the Whole Model")
print("=" * 60)

# Now we can see the FULL architecture:
#
#   Input tokens (B, T)
#       ↓
#   Embedding block → (B, T, C)
#       ↓
#   Block 1: [LN → MHA → +x] → [LN → FFN → +x]    (B, T, C)
#       ↓
#   Block 2: [LN → MHA → +x] → [LN → FFN → +x]    (B, T, C)
#       ↓
#   ...N blocks...
#       ↓
#   Final LayerNorm                                   (B, T, C)
#       ↓
#   Linear head: C → vocab_size                      (B, T, vocab_size)
#       ↓
#   Logits → CrossEntropyLoss during training
#   Logits → Softmax → sample next token during generation

class FullTransformerPreview(nn.Module):
    """
    Preview of the complete nanoGPT architecture.
    All pieces assembled — we'll build this properly in 04_nanogpt.py.
    """
    def __init__(self, vocab_size, embed_size, num_heads, block_size, num_layers):
        super().__init__()
        # Embedding layer (from lesson 1)
        self.token_embedding    = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)

        # Transformer body (N blocks)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads, block_size)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_final = nn.LayerNorm(embed_size)

        # Language model head: maps to vocabulary
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape

        # 1. Embed tokens and positions
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb                  # (B, T, C)

        # 2. Pass through transformer blocks
        x = self.blocks(x)                     # (B, T, C)
        x = self.ln_final(x)                   # (B, T, C)

        # 3. Project to vocabulary size
        logits = self.lm_head(x)               # (B, T, vocab_size)
        return logits


# Build a small version
vocab_size = 65   # ~character-level vocab (like Shakespeare dataset)
model = FullTransformerPreview(
    vocab_size  = vocab_size,
    embed_size  = C,
    num_heads   = num_heads,
    block_size  = T,
    num_layers  = 4,
)

# Count and break down parameters
total = sum(p.numel() for p in model.parameters())
emb_p = sum(p.numel() for p in model.token_embedding.parameters()) + \
        sum(p.numel() for p in model.position_embedding.parameters())
blk_p = sum(p.numel() for p in model.blocks.parameters())
head_p = sum(p.numel() for p in model.lm_head.parameters())

print(f"\nFull model (vocab={vocab_size}, C={C}, heads={num_heads}, layers=4):")
print(f"  {'Layer':<25}  {'Params':>10}")
print(f"  {'-'*38}")
print(f"  {'Embeddings':<25}  {emb_p:>10,}")
print(f"  {'Transformer blocks':<25}  {blk_p:>10,}")
print(f"  {'LM head':<25}  {head_p:>10,}")
print(f"  {'TOTAL':<25}  {total:>10,}")

# Test forward pass
tokens = torch.randint(0, vocab_size, (B, T))   # random token indices
logits = model(tokens)
print(f"\nForward pass:")
print(f"  Input tokens: {tokens.shape}  (B={B}, T={T})")
print(f"  Logits:       {logits.shape}  (B={B}, T={T}, vocab={vocab_size})")
print(f"\nAt each of the {T} positions, the model outputs {vocab_size} scores.")
print(f"The highest score = the model's prediction for the next token.")


print("\n" + "=" * 60)
print("PART 7: Scale Comparison — Our Model vs GPT-2")
print("=" * 60)

configs = [
    # name,          vocab,  C,    heads, block, layers
    ("nanoGPT (ours)", 65,   64,   4,     256,   6),
    ("GPT-2 small",    50257, 768,  12,   1024,  12),
    ("GPT-2 medium",   50257, 1024, 16,   1024,  24),
    ("GPT-2 large",    50257, 1280, 20,   1024,  36),
    ("GPT-2 XL",       50257, 1600, 25,   1024,  48),
]

print(f"\n{'Model':<20} {'C':>6} {'Heads':>6} {'Layers':>7} {'Params':>14}")
print("-" * 58)
for name, vocab, c, heads, block, layers in configs:
    # Approximate parameter count
    emb    = vocab * c + block * c
    per_block = (3 * c * (c//heads) * heads   # Q,K,V projections
              + c * c                          # output projection
              + 2 * (4*c*c + 4*c + c*4*c + c) # FFN
              + 4 * c)                         # 2 LayerNorms
    blocks_total = per_block * layers
    head   = c * vocab
    total  = emb + blocks_total + head
    marker = " ← we build this" if name.startswith("nano") else ""
    print(f"{name:<20} {c:>6} {heads:>6} {layers:>7} {total:>14,}{marker}")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
The transformer block — three ideas working together:

  1. Residual connections
       x = x + sublayer(x)
       → gradients flow directly, enabling deep networks to train

  2. Layer Normalization
       normalize each token vector to mean=0, std=1
       → applied BEFORE each sub-layer (Pre-LN, GPT-2 style)
       → stable training regardless of depth

  3. Feed-Forward Network
       Linear(C→4C) → ReLU → Linear(4C→C)
       → applied to each token independently
       → stores factual associations, processes gathered context

Full block forward pass:
  x = x + Attention(LayerNorm(x))     ← tokens communicate
  x = x + FeedForward(LayerNorm(x))   ← tokens process

Stack N blocks, add embedding layer + LM head = complete GPT.

Next: 04_nanogpt.py
      Put ALL pieces together. Train on real text. Generate coherent output.
      This is the moment everything connects.
""")
