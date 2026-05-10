# Deep Dive Session Plan — `04_nanogpt.py`

> Goal: Understand every concept, every method, every design choice in `04_nanogpt.py`.
> Not just *what* the code does — but *why* it was written exactly that way.

## How Each Session Works

1. **Concept first** — understand the *why* before touching any code
2. **Read the relevant lines** — identify exactly which lines implement it
3. **Break it on purpose** — modify/remove the mechanism, observe what fails
4. **Restore and explain back** — explain it in your own words
5. **Quick question** — one conceptual question to confirm it landed

---

## Session 1 — The Config Class & Hyperparameters

**File location:** `04_nanogpt.py` lines 30–53

**Focus:** Why do we group settings this way? What does each number mean and what happens if you change it?

### You will understand
- Why `embed_size`, `num_heads`, `num_layers` must relate to each other
- What `block_size` really means (not just "context window")
- Why `dropout=0.1` and not `0.5` or `0.0`
- Why `learning_rate=3e-4` is called the "Goldilocks LR for transformers"
- Why device detection (`mps` → `cpu` fallback) belongs in config

### Practice
Change each hyperparameter one at a time, predict what will happen, then run and observe:
- Halve `embed_size` — what breaks?
- Set `dropout=0.0` — what changes during training?
- Set `learning_rate=0.1` — watch the loss curve

### Key question to answer before next session
> "Why can't you set `num_heads=5` when `embed_size=64`?"

---

## Session 2 — Tokenization & The Data Pipeline

**File location:** `04_nanogpt.py` lines 60–108

**Focus:** How does raw text become training examples? Why is the target just the input shifted by 1?

### You will understand
- Character-level vs subword — why we use char-level here
- Why `train_split=0.9` and what validation loss actually tells you
- `get_batch` — why random sampling, why `block_size + 1` characters are sliced
- The input/target shift — why this single idea is enough to train the whole model
- Why `dtype=torch.long` for token indices

### Practice
Print actual batches and decode `x` and `y` side by side:
```python
x, y = get_batch('train')
for i in range(4):
    print(f"input:  '{decode(x[i].tolist())}'")
    print(f"target: '{decode(y[i].tolist())}'")
    print()
```

### Key question to answer before next session
> "From one sequence of 256 tokens, how many training examples does the model actually see?"

---

## Session 3 — SingleHeadAttention: Q, K, V From First Principles

**File location:** `04_nanogpt.py` lines 115–137

**Focus:** Why three separate projections? What does each one *mean*?

### You will understand
- Why a dot product measures similarity between two vectors
- Q = "what am I looking for", K = "what do I contain", V = "what I will share"
- Why `bias=False` in the Q, K, V projections
- The causal mask — why `-inf` before softmax and not just `0`
- Why `softmax(scores, dim=-1)` operates on the last dimension
- Why the output shape is `(B, T, head_size)` not `(B, T, embed_size)`

### Practice
Add print statements to trace one token through the full head:
```python
# Inside SingleHeadAttention.forward(), add:
print(f"Q[0,0]: {Q[0,0].detach()}")        # query for token 0
print(f"K[0,:]: {K[0].detach()}")           # all keys
print(f"scores[0,0]: {scores[0,0].detach()}")  # raw scores for token 0
print(f"weights[0,0]: {weights[0,0].detach()}")  # attention weights
```

### Key question to answer before next session
> "Why does token 0 always have attention weight = 1.0 on itself and 0.0 on everything else?"

---

## Session 4 — Scaled Dot-Product: Why Divide by `sqrt(head_size)`?

**File location:** `04_nanogpt.py` line 133

**Focus:** One line of math — but without it the model can break.

### You will understand
- What happens to variance when you multiply two random matrices together
- Why large dot products push softmax into saturation (outputs near 0 or 1)
- What "saturated softmax" means for gradients during backpropagation
- Why `head_size ** -0.5` is mathematically correct (not an arbitrary constant)

### The math (plain English)
If Q and K are random with variance 1, then Q·K has variance = `head_size`.
Dividing by `sqrt(head_size)` brings variance back to 1.
Softmax with large inputs → vanishing gradients → model stops learning.

### Practice
Remove the scaling and compare:
```python
# Change this line:
scores = Q @ K.transpose(-2, -1) * (self.head_size ** -0.5)
# To this:
scores = Q @ K.transpose(-2, -1)  # no scaling
```
Train 200 steps. Compare loss curve with and without scaling.

### Key question to answer before next session
> "If `head_size=64`, what is the scale factor? What about `head_size=16`?"

---

## Session 5 — MultiHeadAttention: Why Multiple Heads?

**File location:** `04_nanogpt.py` lines 140–155

**Focus:** One head sees one type of relationship. Multiple heads see many simultaneously.

### You will understand
- Why `head_size = embed_size // num_heads` (not an arbitrary split)
- What each head can specialize in (syntax, coreference, positional proximity)
- Why we concatenate outputs then project — not just average them
- The output projection's role: letting heads "communicate" after concatenation
- Total parameter count: `num_heads × 3 × (embed_size × head_size) + embed_size²`

### Practice
```python
# Count parameters manually:
embed_size = 128
num_heads = 8
head_size = embed_size // num_heads   # = 16

qkv_params = num_heads * 3 * (embed_size * head_size)
proj_params = embed_size * embed_size
print(f"QKV params: {qkv_params}")
print(f"Proj params: {proj_params}")
print(f"Total MHA params: {qkv_params + proj_params}")
```

Then change `num_heads` from 4 → 1 → 8, observe how output quality changes.

### Key question to answer before next session
> "If you double `num_heads` but keep `embed_size` the same, does the parameter count change?"

---

## Session 6 — FeedForward: What Happens After Attention?

**File location:** `04_nanogpt.py` lines 158–170

**Focus:** Attention *gathers* context from other tokens. FFN *processes* it. Why are these two separate operations?

### You will understand
- Why each token is processed independently in the FFN (no mixing between tokens)
- Why the 4× expansion (`C → 4C → C`) — the origin of this number
- The FFN as a "key-value memory" hypothesis — what researchers believe it stores
- Why ReLU here (and why modern models like GPT-2 use GELU instead)
- Why FFN has ~2× more parameters than attention — this is intentional

### Practice
Change the expansion factor:
```python
# Original: 4 * embed_size
# Try: 1 * embed_size (no expansion)
# Try: 8 * embed_size (double expansion)
```
Train 500 steps each. What changes in the loss?

### Key question to answer before next session
> "Attention mixes information between tokens. FFN does not. Why do you need both?"

---

## Session 7 — Residual Connections & LayerNorm: Training Deep Networks

**File location:** `04_nanogpt.py` lines 173–185

**Focus:** Without these two ideas, stacking more than ~3 layers is nearly impossible to train.

### You will understand
- The vanishing gradient problem — what it is and why deep networks suffer from it
- Why `x = x + sublayer(x)` creates a "gradient highway" to early layers
- Pre-LN vs Post-LN — why GPT-2 switched to Pre-LN (normalize *before* sublayer)
- What LayerNorm normalizes: per-token vectors, not across the batch
- Why `gamma` (scale) and `beta` (shift) are learnable — not fixed at 1 and 0

### The residual math
```
Without residual:  dL/dx₁ = dL/dx_N × J_N × J_(N-1) × ... × J_1
                   (product of N Jacobians → exponentially small)

With residual:     dL/dx₁ = dL/dx_N × (1 + J_1)
                   (gradient always has a direct path of magnitude 1)
```

### Practice
Remove residual connections from `TransformerBlock`:
```python
# Change:
x = x + self.attention(self.ln1(x))
x = x + self.feed_forward(self.ln2(x))
# To:
x = self.attention(self.ln1(x))
x = self.feed_forward(self.ln2(x))
```
Train 500 steps. Watch training become unstable or fail to converge.

### Key question to answer before next session
> "Why is LayerNorm applied *before* the sublayer in GPT-2, but the original 2017 transformer paper applied it *after*?"

---

## Session 8 — NanoGPT Assembly: Weight Tying & Initialization

**File location:** `04_nanogpt.py` lines 188–236

**Focus:** Two subtle decisions that most tutorials skip — but they meaningfully affect training.

### You will understand
- Weight tying: `token_embedding.weight = lm_head.weight` — why this works
- The mathematical intuition: embedding maps tokens → vectors, lm_head maps vectors → token scores. They are approximate inverses.
- Why `std=0.02` in `_init_weights` — what happens with larger/smaller initialization
- Why `LayerNorm` weights start at `ones` and biases at `zeros`
- Dropout placement: after attention output projection and after FFN — not before

### The weight tying intuition
```
Embedding:  token_id  →  vector  (lookup: which direction in space is this word?)
LM head:    vector    →  scores  (which word does this direction point toward?)

These are inverses of each other.
Sharing weights: fewer parameters, better training signal for embeddings.
Used in GPT-2, GPT-3, LLaMA.
```

### Practice
```python
# Before weight tying — count unique parameters:
total = sum(p.numel() for p in model.parameters())
unique = sum(p.numel() for p in set(model.parameters()))
print(f"Total: {total}, Unique: {unique}, Saved: {total - unique}")
```

### Key question to answer before next session
> "If you remove weight tying, which part of the model loses the most direct training signal and why?"

---

## Session 9 — The Training Loop: AdamW, Gradient Clipping, Eval

**File location:** `04_nanogpt.py` lines 346–388

**Focus:** The training loop looks simple — but every line has a specific reason.

### You will understand
- Adam vs AdamW — the weight decay bug in Adam and how AdamW fixes it
- `weight_decay=0.01` — what it penalizes (large weights) and why transformers need it
- Gradient clipping at `1.0` — what a gradient explosion looks like and when it happens
- Why `estimate_loss` averages over 100 batches instead of just 1
- What `model.eval()` actually changes: dropout is disabled, LayerNorm uses running stats
- Why we evaluate *before* the training step (to capture the initial random loss)

### The AdamW fix (plain English)
```
Adam:   weight_update = -lr × (momentum / sqrt(variance))
        weight_decay applied to momentum → decays decayed — wrong

AdamW:  weight_update = -lr × (momentum / sqrt(variance)) - lr × wd × weight
        weight_decay applied directly to weight → correct L2 regularization
```

### Practice
Remove gradient clipping:
```python
# Comment out:
# torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
Train and watch for a step where loss suddenly spikes. That's an exploding gradient.

### Key question to answer before next session
> "Why does `estimate_loss` call `model.eval()` at the start and `model.train()` at the end?"

---

## Session 10 — Generation: Temperature, Top-k, and Autoregression

**File location:** `04_nanogpt.py` lines 268–294

**Focus:** Training is over. How does the model actually produce text one token at a time?

### You will understand
- The autoregressive loop — why exactly one token is generated per forward pass
- Why context is cropped to `block_size` during generation
- Temperature: dividing logits mathematically sharpens or flattens the distribution
- Top-k: why keeping only the top k tokens improves coherence
- Why `top_k=1` (greedy decoding) gets stuck in repetitive loops
- Nucleus (top-p) sampling — the natural next step beyond top-k

### The temperature math
```python
# temperature=0.5 (low → confident, less variety):
#   logits = [2.0, 1.0, 0.5] / 0.5 = [4.0, 2.0, 1.0]
#   → softmax pushes probability more toward the top token

# temperature=2.0 (high → uncertain, more variety):
#   logits = [2.0, 1.0, 0.5] / 2.0 = [1.0, 0.5, 0.25]
#   → softmax spreads probability more evenly
```

### Practice
Generate text at several temperatures and print the actual probability distributions:
```python
# After: logits = logits[:, -1, :] / temperature
probs = F.softmax(logits, dim=-1)
top_probs, top_idx = probs[0].topk(5)
for prob, idx in zip(top_probs, top_idx):
    print(f"  '{decode([idx.item()])}': {prob.item():.3f}")
```
Run this for `temperature = 0.3, 0.8, 1.5` and see how the distribution changes.

### Key question to answer before next session
> "Why does `top_k=1` produce repetitive output? What does the model get 'stuck' on?"

---

## Progress Tracker

| Session | Topic | Status |
|---------|-------|--------|
| 1 | Config & Hyperparameters | ⬜ Not started |
| 2 | Tokenization & Data Pipeline | ⬜ Not started |
| 3 | SingleHeadAttention — Q, K, V | ⬜ Not started |
| 4 | Scaled Dot-Product | ⬜ Not started |
| 5 | MultiHeadAttention | ⬜ Not started |
| 6 | FeedForward Network | ⬜ Not started |
| 7 | Residual Connections & LayerNorm | ⬜ Not started |
| 8 | NanoGPT Assembly & Weight Tying | ⬜ Not started |
| 9 | Training Loop — AdamW & Clipping | ⬜ Not started |
| 10 | Generation — Temperature & Top-k | ⬜ Not started |

---

## After All Sessions

Once all 10 sessions are complete, you will be able to:

- Read the original **"Attention Is All You Need"** paper and understand every equation
- Look at GPT-2's source code and recognize every component
- Debug a transformer that isn't training — and know exactly where to look
- Explain to anyone why each design decision exists, not just what it does

**The next step after this:** Phase 3 fine-tuning concepts (LoRA, PEFT) will make complete sense because you'll understand what's being frozen and what's being trained — and why.
