# Quick-run version of nanoGPT — 1000 steps, smaller model
# For a full quality run: python 04_nanogpt.py  (5000 steps, ~10-15 min on M2)

import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os
import time

torch.manual_seed(42)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Smaller config for quick demo
embed_size  = 64
num_heads   = 4
num_layers  = 4
block_size  = 128
batch_size  = 32
max_steps   = 1000
eval_every  = 200
lr          = 3e-4
dropout     = 0.1

# --- Data ---
data_path = 'phase2_transformers/shakespeare.txt'
url = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
       "master/data/tinyshakespeare/input.txt")
if not os.path.exists(data_path):
    print("Downloading Shakespeare..."), urllib.request.urlretrieve(url, data_path)
with open(data_path) as f:
    text = f.read()

chars       = sorted(set(text))
vocab_size  = len(chars)
stoi        = {c: i for i, c in enumerate(chars)}
itos        = {i: c for i, c in enumerate(chars)}
encode      = lambda s: [stoi[c] for c in s]
decode      = lambda l: ''.join([itos[i] for i in l])

data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

def get_batch(split):
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i+block_size]   for i in ix]).to(device)
    y  = torch.stack([d[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# --- Model ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.q = nn.Linear(embed_size, head_size, bias=False)
        self.k = nn.Linear(embed_size, head_size, bias=False)
        self.v = nn.Linear(embed_size, head_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.hs = head_size

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        w = q @ k.transpose(-2,-1) * self.hs**-0.5
        w = w.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        w = self.drop(F.softmax(w, dim=-1))
        return w @ v

class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        hs = embed_size // num_heads
        self.heads = nn.ModuleList([Head(hs) for _ in range(num_heads)])
        self.proj  = nn.Linear(embed_size, embed_size)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size), nn.ReLU(),
            nn.Linear(4*embed_size, embed_size), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.attn = MHA()
        self.ffn  = FFN()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(block_size, embed_size)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.Sequential(*[Block() for _ in range(num_layers)])
        self.ln      = nn.LayerNorm(embed_size)
        self.head    = nn.Linear(embed_size, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight   # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        x = self.ln(self.blocks(x))
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(n):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx

model = GPT().to(device)
params = sum(p.numel() for p in set(model.parameters()))
print(f"Device: {device}")
print(f"Model: {params:,} parameters  ({embed_size}d, {num_heads}h, {num_layers}L, ctx={block_size})")
print(f"Vocab: {vocab_size} characters\n")

# --- Training ---
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

@torch.no_grad()
def eval_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = [model(*get_batch(split))[1].item() for _ in range(50)]
        out[split] = sum(losses)/len(losses)
    model.train()
    return out

print(f"Training {max_steps} steps...")
print(f"(For full quality run: edit max_steps=5000, embed_size=128, num_layers=6)\n")
t0 = time.time()

for step in range(max_steps):
    if step % eval_every == 0 or step == max_steps-1:
        L = eval_loss()
        print(f"  step {step:5d} | train: {L['train']:.4f} | val: {L['val']:.4f} | {time.time()-t0:.0f}s")
    x, y = get_batch('train')
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print(f"\nDone in {time.time()-t0:.0f}s")

# --- Generate ---
print("\n" + "="*60)
print("GENERATED TEXT")
print("="*60)

prompts = ["ROMEO:", "To be or not", "First Citizen:"]
for p in prompts:
    tokens = torch.tensor([encode(p)], dtype=torch.long).to(device)
    out    = model.generate(tokens, n=200, temperature=0.8, top_k=40)
    print(f"\n[Prompt: '{p}']\n{decode(out[0].tolist())}\n")

torch.save({'state': model.state_dict(), 'itos': itos, 'stoi': stoi,
            'config': dict(embed_size=embed_size, num_heads=num_heads,
                          num_layers=num_layers, block_size=block_size,
                          vocab_size=vocab_size)},
           'phase2_transformers/nanogpt_quick.pt')
print("Checkpoint saved: phase2_transformers/nanogpt_quick.pt")
