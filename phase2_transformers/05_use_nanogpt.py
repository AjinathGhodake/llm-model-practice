# =============================================================================
# HOW TO USE YOUR TRAINED nanoGPT
# =============================================================================
# This file shows every practical way to interact with the model you trained.
#
# Sections:
#   1. Load the model from checkpoint
#   2. Basic text generation
#   3. Controlling output with temperature + top_k
#   4. Completion mode (finish my sentence)
#   5. What the model actually learned
#   6. Limitations — being honest about what it can/can't do
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. REBUILD THE MODEL ARCHITECTURE (needed to load weights)
# =============================================================================
# PyTorch saves weights, not the class definition.
# To load, you must define the same architecture first, then load the weights.


class Head(nn.Module):
    def __init__(self, embed_size, head_size, block_size, dropout):
        super().__init__()
        self.q = nn.Linear(embed_size, head_size, bias=False)
        self.k = nn.Linear(embed_size, head_size, bias=False)
        self.v = nn.Linear(embed_size, head_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.hs = head_size
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        w = q @ k.transpose(-2, -1) * self.hs**-0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = self.drop(F.softmax(w, dim=-1))
        return w @ v


class MHA(nn.Module):
    def __init__(self, embed_size, num_heads, block_size, dropout):
        super().__init__()
        hs = embed_size // num_heads
        self.heads = nn.ModuleList(
            [Head(embed_size, hs, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_size, embed_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FFN(nn.Module):
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


class Block(nn.Module):
    def __init__(self, embed_size, num_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.attn = MHA(embed_size, num_heads, block_size, dropout)
        self.ffn = FFN(embed_size, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(
        self, vocab_size, embed_size, num_heads, block_size, num_layers, dropout=0.0
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(block_size, embed_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[
                Block(embed_size, num_heads, block_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(
            self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        )
        x = self.ln(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(n):
            idx_c = idx[:, -self.block_size :]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_t], dim=1)
        return idx


# =============================================================================
# 2. LOAD THE CHECKPOINT
# =============================================================================

device = "mps" if torch.backends.mps.is_available() else "cpu"
ckpt = torch.load(
    "phase2_transformers/nanogpt_quick.pt", map_location=device, weights_only=False
)

cfg = ckpt["config"]
itos = ckpt["itos"]
stoi = ckpt["stoi"]

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

model = NanoGPT(
    vocab_size=cfg["vocab_size"],
    embed_size=cfg["embed_size"],
    num_heads=cfg["num_heads"],
    block_size=cfg["block_size"],
    num_layers=cfg["num_layers"],
    dropout=0.0,  # always 0 at inference — we're not training
).to(device)

model.load_state_dict(ckpt["state"])
model.eval()

params = sum(p.numel() for p in set(model.parameters()))
print(f"Model loaded  |  {params:,} parameters  |  device={device}")
print(
    f"Config: embed={cfg['embed_size']}, heads={cfg['num_heads']}, "
    f"layers={cfg['num_layers']}, ctx={cfg['block_size']}"
)
print()


# =============================================================================
# HELPER: clean print utility
# =============================================================================
def run(label, prompt, n=200, temperature=0.8, top_k=40):
    tokens = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    out = model.generate(tokens, n=n, temperature=temperature, top_k=top_k)
    text = decode(out[0].tolist())
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(text)


# =============================================================================
# 3. BASIC TEXT GENERATION
# =============================================================================
# The model takes a prompt and continues it.
# Everything is character-level — it generates one character at a time.

print("\n" + "=" * 60)
print("USE CASE 1: Basic Continuation")
print("=" * 60)
print("Give the model a starting string, it continues the text.")

run("Prompt: 'ROMEO:'", prompt="ROMEO:", n=250)

run("Prompt: 'HAMLET:'", prompt="HAMLET:", n=250)


# =============================================================================
# 4. TEMPERATURE — Controlling Creativity vs Coherence
# =============================================================================
# Temperature is the single most important generation parameter.
#
# temperature < 1.0  → more focused, repeats common patterns
# temperature = 1.0  → balanced
# temperature > 1.0  → more creative, but less coherent
#
# How it works (from 02_attention.py):
#   logits = logits / temperature       ← scale scores before softmax
#   low temp  → sharpens distribution  → model picks "safe" tokens
#   high temp → flattens distribution  → model takes more risks

print("\n\n" + "=" * 60)
print("USE CASE 2: Temperature Effect")
print("=" * 60)
print("Same prompt, different temperatures — watch how output changes.\n")

base_prompt = "The king said to"

for temp in [0.3, 0.8, 1.4]:
    tokens = torch.tensor([encode(base_prompt)], dtype=torch.long).to(device)
    out = model.generate(tokens, n=120, temperature=temp, top_k=40)
    text = decode(out[0].tolist())
    label = {
        0.3: "focused (may repeat)",
        0.8: "balanced",
        1.4: "creative (may be chaotic)",
    }[temp]
    print(f"--- temperature={temp} ({label}) ---")
    print(text)
    print()


# =============================================================================
# 5. TOP-K — Cutting Off Bad Tokens
# =============================================================================
# top_k limits which tokens the model can sample from.
# Instead of picking from all vocab_size tokens, only pick from the top k.
# This blocks low-probability garbage tokens from appearing.
#
# top_k=1   → always pick the single most likely token (greedy, repetitive)
# top_k=10  → pick from top 10 candidates (focused)
# top_k=40  → pick from top 40 (good balance — default for most LLMs)
# top_k=None → pick from all tokens (risky, low prob tokens can sneak in)

print("=" * 60)
print("USE CASE 3: top_k Effect")
print("=" * 60)
print("Same prompt, same temperature, different top_k values.\n")

base_prompt = "What is the"

for k in [1, 10, 40, None]:
    tokens = torch.tensor([encode(base_prompt)], dtype=torch.long).to(device)
    out = model.generate(tokens, n=100, temperature=0.8, top_k=k)
    text = decode(out[0].tolist())
    label = {
        1: "greedy — deterministic",
        10: "focused",
        40: "balanced",
        None: "unrestricted",
    }[k]
    print(f"--- top_k={k} ({label}) ---")
    print(text)
    print()


# =============================================================================
# 6. COMPLETION MODE
# =============================================================================
# Feed the model the beginning of a Shakespearean line.
# It completes what it thinks comes next.
# This is exactly what GitHub Copilot does — just trained on code instead.

print("=" * 60)
print("USE CASE 4: Sentence Completion")
print("=" * 60)
print("Model finishes sentences from the training data.\n")

completions = [
    "To be or not to be",
    "All the world",
    "What a piece of work",
    "Now is the winter",
]

for prompt in completions:
    tokens = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    out = model.generate(tokens, n=80, temperature=0.7, top_k=30)
    full = decode(out[0].tolist())
    # Show just the completion, not the prompt
    completion = full[len(prompt) :]
    print(f'  Prompt:     "{prompt}"')
    print(f'  Completion: "{completion.strip()[:100]}"')
    print()


# =============================================================================
# 7. WHAT THE MODEL ACTUALLY LEARNED
# =============================================================================
# Look inside the model — what probability does it assign to next characters?

print("=" * 60)
print("USE CASE 5: Inspecting the Model's Beliefs")
print("=" * 60)
print("After a prompt, what characters does the model think come next?\n")

test_prompts = ["ROMEO:\n", "To be or not to ", "the king "]

for prompt in test_prompts:
    tokens = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits, _ = model(tokens)
    # Get probabilities for the LAST position (next character)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    top_p, top_i = probs.topk(5)

    print(f"After '{repr(prompt)}':")
    for prob, idx in zip(top_p, top_i):
        char = itos[idx.item()]
        bar = "█" * int(prob.item() * 40)
        print(f"  '{repr(char):8}' {prob.item():.3f}  {bar}")
    print()


# =============================================================================
# 8. WHAT THIS MODEL CAN AND CANNOT DO
# =============================================================================

print("=" * 60)
print("HONEST CAPABILITIES SUMMARY")
print("=" * 60)
print(
    f"""
CAN DO:
  ✅ Generate Shakespeare-like text
  ✅ Maintain dialogue structure (CHARACTER:\\nText)
  ✅ Continue a given prompt in the training domain
  ✅ Show the effect of temperature and top_k
  ✅ Run entirely offline, on your machine, in <1 second per token

CANNOT DO:
  ❌ Answer questions
  ❌ Follow instructions ("write a poem about X")
  ❌ Understand text outside Shakespeare
  ❌ Produce reliably grammatical English
  ❌ Remember context across separate calls

WHY:
  This model has 211K parameters and trained on 1MB of text for 1000 steps.
  GPT-2 has 117M parameters and trained on 40GB of text.
  Same architecture. ~1000x more data and parameters = qualitative leap.

THE POINT:
  You understand every layer of this model.
  When you move to Phase 3 (fine-tuning Llama 3.2),
  you will know exactly what is happening inside a model
  that CAN do all of the above — because the architecture is identical.
"""
)


# =============================================================================
# 9. INTERACTIVE LOOP (uncomment to use)
# =============================================================================
# Uncomment the block below to type your own prompts interactively.

print("\\n" + "=" * 60)
print("INTERACTIVE MODE  (type 'quit' to exit)")
print("=" * 60)
while True:
    prompt = input("\\nYour prompt: ")
    if prompt.lower() in ("quit", "exit", "q"):
        break
    temp = float(input("Temperature (0.1–1.5, default 0.8): ") or "0.8")
    n = int(input("Characters to generate (default 200): ") or "200")
    tokens = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    out = model.generate(tokens, n=n, temperature=temp, top_k=40)
    print("\\n" + decode(out[0].tolist()))
