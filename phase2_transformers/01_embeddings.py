# =============================================================================
# PHASE 2 — LESSON 1: Embeddings
# =============================================================================
# The transformer's first job: convert raw token integers into rich vectors.
#
# Two types of embeddings, always added together:
#   Token embedding     → WHAT is this token?
#   Positional embedding → WHERE is this token in the sequence?
#
# Their sum = the input to every transformer layer that follows.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

print("=" * 60)
print("PART 1: The Problem With Raw Integers")
print("=" * 60)

# In the bigram model, we passed token indices directly into nn.Embedding
# and got back a row of logits. That worked for a lookup table.
#
# But transformers need richer inputs — vectors with structure.
# Why? Because the attention mechanism computes dot products between vectors.
# The quality of those dot products depends on the quality of the vectors.
#
# Problem with using raw integers as input:
#   token 0, token 1, token 2 ...
#   The numbers imply an ordering that doesn't exist.
#   "cat"=5 and "dog"=6 are not more similar than "cat"=5 and "fish"=50.
#
# Solution: map each token to a learned VECTOR of numbers.
# Similar tokens will naturally develop similar vectors during training.

# Toy vocabulary: 6 tokens, we'll embed them into 4-dimensional vectors
vocab_size  = 6
embed_size  = 4     # each token becomes a vector of 4 numbers
              #  (GPT-2 small uses 768, GPT-3 uses 12288)

print(f"\nvocab_size = {vocab_size}  (6 possible tokens)")
print(f"embed_size = {embed_size}  (each token → vector of 4 numbers)")
print(f"\nIn GPT-2 small: vocab_size=50257, embed_size=768")
print(f"In GPT-3:       vocab_size=50257, embed_size=12288")


print("\n" + "=" * 60)
print("PART 2: nn.Embedding — The Token Lookup Table")
print("=" * 60)

# nn.Embedding is just a matrix: shape (vocab_size, embed_size)
# Each row = the vector for one token.
# Forward pass = look up the row for each token index.
# The vectors are LEARNED — they update during training via backprop.

token_embedding_table = nn.Embedding(vocab_size, embed_size)

print(f"\nEmbedding table shape: {token_embedding_table.weight.shape}")
print(f"  ({vocab_size} tokens × {embed_size} dimensions)")
print(f"\nAll token vectors (randomly initialized):")
for i in range(vocab_size):
    vec = token_embedding_table.weight[i]
    print(f"  token {i}: {vec.data.numpy().round(3)}")

# Look up embeddings for a sequence of tokens
tokens = torch.tensor([0, 3, 1, 5])   # a 4-token sequence
embedded = token_embedding_table(tokens)
print(f"\nInput tokens:    {tokens.tolist()}")
print(f"Embedded shape:  {embedded.shape}  (4 tokens × {embed_size} dims)")
print(f"Embedded:\n{embedded.data.numpy().round(3)}")


print("\n" + "=" * 60)
print("PART 3: Why Embeddings Beat One-Hot Encoding")
print("=" * 60)

# Alternative: one-hot encoding — a vector of all zeros except one 1
# Problem 1: sparse and huge (vocab_size dimensions, almost all zero)
# Problem 2: all tokens are equidistant — no notion of similarity
# Problem 3: not learnable in a useful way

# One-hot for token 2 in a vocab of 6:
one_hot = F.one_hot(torch.tensor(2), num_classes=vocab_size).float()
print(f"\nOne-hot for token 2: {one_hot.tolist()}")
print(f"  Size: {one_hot.shape[0]} dimensions")
print(f"  Every token is exactly distance sqrt(2) from every other token")
print(f"  No structure — 'king' is as far from 'queen' as from 'banana'")

# Learned embeddings develop structure:
# After training on real text, you'd see things like:
#   embed('king') - embed('man') + embed('woman') ≈ embed('queen')
# The vectors encode meaning through their geometric relationships.
print(f"\nLearned embedding for token 2: {token_embedding_table(torch.tensor(2)).data.numpy().round(3)}")
print(f"  Size: {embed_size} dimensions (much smaller)")
print(f"  Geometric relationships encode semantic meaning after training")


print("\n" + "=" * 60)
print("PART 4: Positional Embeddings — Teaching the Model About Order")
print("=" * 60)

# Critical problem: attention computes the same operation regardless of order.
# "The cat sat" and "sat the cat" would produce identical token embeddings.
# The model needs to know WHERE each token is in the sequence.
#
# Solution: add a learned vector for each POSITION.
# Position 0 gets its own vector, position 1 gets its own vector, etc.
# This is added to the token embedding before anything else.
#
# "The cat sat":
#   token embed("The")  + pos embed(0) → input for position 0
#   token embed("cat")  + pos embed(1) → input for position 1
#   token embed("sat")  + pos embed(2) → input for position 2

block_size = 8   # maximum sequence length (context window)

position_embedding_table = nn.Embedding(block_size, embed_size)

print(f"\nPositional embedding table shape: {position_embedding_table.weight.shape}")
print(f"  ({block_size} positions × {embed_size} dimensions)")
print(f"\nAll position vectors (randomly initialized):")
for i in range(block_size):
    vec = position_embedding_table.weight[i]
    print(f"  position {i}: {vec.data.numpy().round(3)}")


print("\n" + "=" * 60)
print("PART 5: Combining Token + Positional Embeddings")
print("=" * 60)

# This is EXACTLY what happens at the start of every transformer forward pass.
# Step 1: look up token embedding for each token
# Step 2: look up position embedding for each position index
# Step 3: add them element-wise
# Result: each token is represented by WHAT it is + WHERE it is

# A batch of 2 sequences, each 5 tokens long
B = 2   # batch size
T = 5   # sequence length (must be ≤ block_size)

tokens_batch = torch.tensor([
    [0, 3, 1, 5, 2],   # sequence 1
    [4, 1, 0, 2, 3],   # sequence 2
])
print(f"\nInput tokens shape: {tokens_batch.shape}  (batch={B}, seq_len={T})")

# Step 1: token embeddings — shape (B, T, embed_size)
tok_emb = token_embedding_table(tokens_batch)
print(f"Token embeddings shape: {tok_emb.shape}")

# Step 2: position indices [0, 1, 2, ..., T-1] — same for every sequence in batch
positions = torch.arange(T)              # [0, 1, 2, 3, 4]
pos_emb   = position_embedding_table(positions)   # (T, embed_size)
print(f"Position embeddings shape: {pos_emb.shape}")
print(f"  (will broadcast across the batch dimension automatically)")

# Step 3: add them — pos_emb broadcasts from (T, C) to (B, T, C)
x = tok_emb + pos_emb                   # (B, T, embed_size)
print(f"Combined (tok + pos) shape: {x.shape}")

# Show what happened for one token
print(f"\nFor sequence 0, position 2 (token {tokens_batch[0,2].item()}):")
print(f"  token embedding:    {tok_emb[0,2].data.numpy().round(3)}")
print(f"  position embedding: {pos_emb[2].data.numpy().round(3)}")
print(f"  sum (model input):  {x[0,2].data.numpy().round(3)}")


print("\n" + "=" * 60)
print("PART 6: Embedding as Part of a Model")
print("=" * 60)

# This is the standard embedding block you'll see at the top of
# every transformer, including GPT-2, GPT-3, and our nanoGPT.

class EmbeddingBlock(nn.Module):
    """
    First layer of every transformer:
    converts (B, T) token indices into (B, T, embed_size) vectors.
    """
    def __init__(self, vocab_size, embed_size, block_size):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.block_size         = block_size

    def forward(self, idx):
        # idx: (B, T) — integer token indices
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        # Token embeddings: each token → its learned vector
        tok_emb = self.token_embedding(idx)                    # (B, T, embed_size)

        # Positional embeddings: each position → its learned vector
        positions = torch.arange(T, device=idx.device)        # [0, 1, ..., T-1]
        pos_emb   = self.position_embedding(positions)        # (T, embed_size)

        # Sum: gives each token awareness of both identity and position
        return tok_emb + pos_emb                               # (B, T, embed_size)


emb_block = EmbeddingBlock(vocab_size=vocab_size, embed_size=embed_size, block_size=block_size)
output = emb_block(tokens_batch)

total_params = sum(p.numel() for p in emb_block.parameters())
print(f"\nEmbeddingBlock:")
print(f"  Input shape:  {tokens_batch.shape}   (B=2, T=5 token indices)")
print(f"  Output shape: {output.shape}  (B=2, T=5, embed_size=4 vectors)")
print(f"  Parameters:   {total_params}")
print(f"    token embedding:    {vocab_size} × {embed_size} = {vocab_size*embed_size}")
print(f"    position embedding: {block_size} × {embed_size} = {block_size*embed_size}")


print("\n" + "=" * 60)
print("PART 7: Scale — What This Looks Like in Real Models")
print("=" * 60)

# Let's see how embedding parameters scale with model size
configs = [
    ("Our nanoGPT",  65,    64,    256),
    ("GPT-2 small",  50257, 768,   1024),
    ("GPT-2 large",  50257, 1280,  1024),
    ("GPT-3",        50257, 12288, 2048),
]

print(f"\n{'Model':<18} {'Vocab':>7} {'EmbDim':>7} {'CtxLen':>7} {'EmbParams':>12}")
print("-" * 58)
for name, vocab, emb_dim, ctx_len in configs:
    tok_params = vocab * emb_dim
    pos_params = ctx_len * emb_dim
    total      = tok_params + pos_params
    print(f"{name:<18} {vocab:>7,} {emb_dim:>7,} {ctx_len:>7,} {total:>12,}")

print(f"\nOur nanoGPT embedding layer is tiny but the mechanism is identical.")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key takeaways:
  1. nn.Embedding          → a learned lookup table: integer → vector
  2. Token embedding       → encodes WHAT each token is
  3. Positional embedding  → encodes WHERE each token is in the sequence
  4. tok_emb + pos_emb     → the combined input to all transformer layers
  5. Output shape          → always (B, T, embed_size) after this step
  6. embed_size            → the channel dimension that flows through the whole model

The embedding output (B, T, embed_size) feeds into the attention mechanism.
Right now each token's vector is independent — it knows nothing about
the other tokens around it.

Attention fixes that: each token will look at every other token's vector,
weigh which ones are relevant, and update itself accordingly.

Next: 02_attention.py — self-attention from scratch
      The single most important mechanism in modern AI.
""")
