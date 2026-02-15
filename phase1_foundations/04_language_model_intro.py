# =============================================================================
# LESSON 4: Language Model Intro
# =============================================================================
# A language model is a neural network trained on one task:
#   "Given the tokens so far, predict the next token."
#
# This single task, trained on enough data, produces models that can write,
# reason, answer questions, and code — because to predict text well,
# you must understand it.
#
# This file covers:
#   1. Tokenization — text → integers
#   2. Vocabulary — the set of all tokens
#   3. Next-token prediction as classification
#   4. The Bigram model — simplest possible language model
#   5. Training and generating text
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# Small text corpus — we'll train a model to write in this style
TEXT = """to be or not to be that is the question
whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune
or to take arms against a sea of troubles
and by opposing end them to die to sleep
no more and by a sleep to say we end"""

print("=" * 60)
print("PART 1: Tokenization — Text to Numbers")
print("=" * 60)

# Tokenization = converting text into a sequence of integers.
# Every model requires this — neural networks only understand numbers.
#
# Three common levels:
#   Character-level: each character is a token  ("hello" → [h,e,l,l,o])
#   Word-level:      each word is a token        ("hello world" → [hello, world])
#   Subword (BPE):   pieces of words are tokens  ("unhappy" → [un, happy])
#
# GPT uses subword (BPE) with ~50,000 tokens.
# We start with character-level — simplest to understand.

# Build vocabulary: all unique characters in the text
chars = sorted(set(TEXT))               # sorted list of unique characters
vocab_size = len(chars)

print(f"\nText length: {len(TEXT)} characters")
print(f"Unique characters: {vocab_size}")
print(f"Vocabulary: {chars}")

# Create lookup tables: character ↔ integer
char_to_int = {ch: i for i, ch in enumerate(chars)}   # 'a' → 0, 'b' → 1, ...
int_to_char = {i: ch for i, ch in enumerate(chars)}   # 0 → 'a', 1 → 'b', ...

print(f"\nchar_to_int (first 10): {dict(list(char_to_int.items())[:10])}")
print(f"int_to_char (first 10): {dict(list(int_to_char.items())[:10])}")

# Encode: text → list of integers
def encode(text):
    return [char_to_int[ch] for ch in text]

# Decode: list of integers → text
def decode(indices):
    return ''.join([int_to_char[i] for i in indices])

sample = "to be"
encoded = encode(sample)
decoded = decode(encoded)
print(f"\nEncode: '{sample}' → {encoded}")
print(f"Decode: {encoded} → '{decoded}'")

# Encode the full text into a tensor of integers
data = torch.tensor(encode(TEXT), dtype=torch.long)
print(f"\nFull text encoded:")
print(f"  Shape: {data.shape}")
print(f"  First 20 tokens: {data[:20].tolist()}")
print(f"  Decoded back: '{decode(data[:20].tolist())}'")


print("\n" + "=" * 60)
print("PART 2: Next-Token Prediction — The Training Task")
print("=" * 60)

# The training task is simple:
#   Input:  a sequence of tokens  [t1, t2, t3, t4]
#   Target: the SAME sequence shifted by 1  [t2, t3, t4, t5]
#
# At every position, we predict what comes next.
# This gives us (sequence_length) training examples from ONE sequence.
#
# Example with block_size=4 (context window = 4 characters):

block_size = 4   # how many characters the model sees at once (context window)

print(f"\nblock_size (context window) = {block_size}")
print(f"\nFrom the sequence: {data[:block_size+1].tolist()}")
print(f"Decoded:           '{decode(data[:block_size+1].tolist())}'")
print(f"\nThis gives us {block_size} training examples:")

for t in range(block_size):
    context = data[:t+1].tolist()       # input: tokens seen so far
    target  = data[t+1].item()          # target: next token
    print(f"  input={str(context):30} → target={target} ('{int_to_char[target]}')")


print("\n" + "=" * 60)
print("PART 3: Batching — Training on Many Sequences at Once")
print("=" * 60)

# In real training, we process many sequences simultaneously (a batch).
# This is more efficient and gives better gradient estimates.

torch.manual_seed(42)
batch_size = 4     # process 4 sequences at once
block_size = 8     # each sequence is 8 tokens long

def get_batch(data, batch_size, block_size):
    """Sample a random batch of (input, target) pairs."""
    # Pick random starting positions (one per batch item)
    # Must be far enough from the end that we have block_size+1 tokens
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Stack sequences into a matrix
    # x = inputs:  each row is a sequence of block_size tokens
    # y = targets: each row is x shifted right by 1
    x = torch.stack([data[i   : i+block_size  ] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    return x, y

x_batch, y_batch = get_batch(data, batch_size, block_size)

print(f"\nBatch shapes:")
print(f"  x (inputs):  {x_batch.shape}   — {batch_size} sequences of {block_size} tokens")
print(f"  y (targets): {y_batch.shape}   — same shape, shifted by 1")

print(f"\nFirst sequence in batch:")
print(f"  input:  {x_batch[0].tolist()}")
print(f"  target: {y_batch[0].tolist()}")
print(f"  input decoded:  '{decode(x_batch[0].tolist())}'")
print(f"  target decoded: '{decode(y_batch[0].tolist())}'")
print(f"  (target is input shifted one step forward)")


print("\n" + "=" * 60)
print("PART 4: The Bigram Model — Simplest Language Model")
print("=" * 60)

# The bigram model predicts the next token using ONLY the current token.
# No memory. No context. Just: "given I see 'h', what probably comes next?"
#
# Implementation: a simple lookup table (embedding table).
# For each token, look up a row of scores (logits) for all possible next tokens.
# That's it — one matrix multiply, no layers, no attention.
#
# Despite being simple, it can learn short patterns.
# And crucially — it uses the EXACT same training loop as GPT.

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # Embedding table: shape (vocab_size, vocab_size)
        # Each row i = the logits (scores) for what token follows token i
        # This is a lookup table — model.token_embedding[i] gives row i
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (batch_size, sequence_length) — integer token indices
        # targets: (batch_size, sequence_length) — next token at each position

        # Look up logits for each token in the sequence
        # logits shape: (batch_size, sequence_length, vocab_size)
        logits = self.token_embedding(idx)

        loss = None
        if targets is not None:
            # Reshape for CrossEntropyLoss:
            # expects (N, vocab_size) and (N,) — flatten batch and sequence dims
            B, T, C = logits.shape          # B=batch, T=time/seq_len, C=vocab_size
            logits_flat  = logits.view(B*T, C)    # (B*T, vocab_size)
            targets_flat = targets.view(B*T)      # (B*T,)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens by sampling from the model's predictions."""
        for _ in range(max_new_tokens):
            # Get predictions for the current sequence
            logits, _ = self(idx)           # (B, T, vocab_size)

            # Focus only on the LAST time step — that's our next-token prediction
            logits_last = logits[:, -1, :] # (B, vocab_size)

            # Convert logits to probabilities
            probs = F.softmax(logits_last, dim=-1)  # (B, vocab_size)

            # Sample from the probability distribution
            # (don't just take the highest — sampling gives variety)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the new token to the sequence
            idx = torch.cat([idx, next_token], dim=1)   # (B, T+1)

        return idx


# Create model
model = BigramLanguageModel(vocab_size)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nBigram model created")
print(f"Vocab size: {vocab_size}")
print(f"Parameters: {total_params}  ({vocab_size} × {vocab_size} embedding table)")

# Test forward pass
x_test, y_test = get_batch(data, batch_size=2, block_size=8)
logits, loss = model(x_test, y_test)
print(f"\nForward pass:")
print(f"  Input shape:  {x_test.shape}")
print(f"  Logits shape: {logits.shape}")   # (batch, seq_len, vocab_size)
print(f"  Loss: {loss.item():.4f}")

# What's the expected initial loss?
# With vocab_size tokens, random guessing probability = 1/vocab_size
# CrossEntropy of uniform distribution = -log(1/vocab_size) = log(vocab_size)
expected_loss = torch.log(torch.tensor(vocab_size, dtype=torch.float))
print(f"  Expected loss (random): {expected_loss.item():.4f}  (model should start near this)")


print("\n" + "=" * 60)
print("PART 5: Generate Text Before Training (random gibberish)")
print("=" * 60)

torch.manual_seed(42)
start_token = torch.zeros((1, 1), dtype=torch.long)   # (1, 1) — batch=1, start with token 0
generated = model.generate(start_token, max_new_tokens=100)
generated_text = decode(generated[0].tolist())
print(f"\nGenerated (untrained model):\n'{generated_text}'")
print(f"\n(Random characters — model has no knowledge yet)")


print("\n" + "=" * 60)
print("PART 6: Train the Bigram Model")
print("=" * 60)

torch.manual_seed(42)
model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

print(f"\nTraining bigram model on Shakespeare text...\n")

for step in range(5000):
    # Sample a batch
    x_b, y_b = get_batch(data, batch_size=16, block_size=8)

    # Forward + loss
    logits, loss = model(x_b, y_b)

    # Backward + update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0 or step == 4999:
        print(f"  Step {step+1:5d}: loss={loss.item():.4f}")


print("\n" + "=" * 60)
print("PART 7: Generate Text After Training")
print("=" * 60)

model.eval()
torch.manual_seed(42)

start_token = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(start_token, max_new_tokens=200)
generated_text = decode(generated[0].tolist())
print(f"\nGenerated (trained bigram model):\n'{generated_text}'")
print(f"\nNot Shakespeare — but it learned real patterns from the text!")
print(f"Common bigrams like 'to', 'th', 'he', 'nd' should appear.")


print("\n" + "=" * 60)
print("PART 8: What the Model Learned")
print("=" * 60)

# Let's look at what the model learned for a specific token
model.eval()
with torch.no_grad():
    for char in ['t', 'h', ' ']:
        token_idx = char_to_int[char]
        # Get the probability distribution over next tokens
        logits = model.token_embedding(torch.tensor([token_idx]))
        probs = F.softmax(logits, dim=-1).squeeze()

        # Top 3 most likely next characters
        top_probs, top_indices = probs.topk(3)
        top_chars = [int_to_char[i.item()] for i in top_indices]
        top_pct   = [f"{p.item():.1%}" for p in top_probs]

        print(f"\nAfter '{char}', most likely next characters:")
        for ch, pct in zip(top_chars, top_pct):
            label = repr(ch)   # repr shows spaces/newlines clearly
            print(f"  '{label}': {pct}")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Key takeaways:
  1. Tokenization        → text becomes integers via a vocabulary lookup
  2. Vocabulary          → the set of all possible tokens (our "classes")
  3. Next-token task     → at every position, predict what comes next
  4. input/target        → same sequence, target shifted 1 step forward
  5. CrossEntropyLoss    → measures how wrong the next-token prediction is
  6. Bigram model        → simplest LLM: 1 token → next token probabilities
  7. generate()          → sample tokens one at a time using softmax + multinomial

The gap between a bigram model and GPT:
  Bigram:  sees 1 token of context
  GPT:     sees 1024+ tokens of context via ATTENTION MECHANISM

That attention mechanism — letting every token look at all previous tokens —
is what we build in Phase 2.

Next: 05_putting_it_together.py
      — combine everything from Phase 1 into one clean training pipeline
      — then Phase 2 begins: building the transformer
""")
