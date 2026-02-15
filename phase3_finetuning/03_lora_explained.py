# =============================================================================
# PHASE 3 — LESSON 3: LoRA — Low-Rank Adaptation
# =============================================================================
# Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al. 2021)
#
# Core idea:
#   Weight updates during fine-tuning have low intrinsic rank.
#   Instead of updating W directly, learn two small matrices A and B.
#   ΔW ≈ B @ A  where rank(B @ A) = r << d
#
# Result:
#   - Freeze ALL original weights (no catastrophic forgetting)
#   - Train only A and B (tiny fraction of parameters)
#   - Store only the adapter (few MB instead of hundreds)
#   - Performance matches full fine-tuning on most tasks
#
# This file:
#   1. The math — why low-rank works
#   2. LoRA from scratch — implement it in raw PyTorch
#   3. Apply LoRA to GPT-2 manually
#   4. PEFT library — how to use it in practice
#   5. Parameter and memory savings comparison
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

torch.manual_seed(42)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}\n")


# =============================================================================
# PART 1: WHY LOW-RANK? — The Core Intuition
# =============================================================================

print("=" * 60)
print("PART 1: Why Low-Rank Works")
print("=" * 60)

# A matrix has "rank r" if it can be written as the product of two
# matrices with r columns/rows respectively.
# Rank = the number of truly independent dimensions of information.
#
# Full-rank matrix: every row/column carries unique information
# Low-rank matrix:  rows/columns are combinations of just r independent vectors
#
# Research finding: when you fine-tune a large model, the weight CHANGES
# (ΔW) tend to have very low rank — often r=1 to r=8 is enough.
# This means fine-tuning mostly adjusts a small subspace of the full
# weight space. LoRA exploits this directly.

d = 768   # typical embedding dimension (GPT-2)
r = 8     # LoRA rank

# Show the parameter savings
full_params = d * d
lora_params = d * r + r * d   # A + B

print(f"\nWeight matrix W: shape ({d}, {d})")
print(f"  Full parameters:  {full_params:,}")
print(f"  LoRA parameters:  {lora_params:,}  (A: {d}×{r} + B: {r}×{d})")
print(f"  Reduction:        {full_params // lora_params}×  fewer parameters")

# Show for different ranks
print(f"\n{'Rank r':>8}  {'LoRA params':>12}  {'Reduction':>10}  {'Quality'}")
print(f"  {'-'*48}")
for rank in [1, 2, 4, 8, 16, 32, 64]:
    lp  = d * rank + rank * d
    red = full_params // lp
    quality = {1:'minimal', 2:'low', 4:'good', 8:'standard ✓',
               16:'high', 32:'very high', 64:'near-full'}[rank]
    print(f"  {rank:>6}  {lp:>12,}  {red:>9}×  {quality}")

print(f"\n  r=8 is the most common default — good quality, minimal parameters")


# =============================================================================
# PART 2: THE MATH — Forward Pass With LoRA
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: The Math — LoRA Forward Pass")
print("=" * 60)

# Without LoRA, a linear layer computes:
#   output = x @ W.T + bias
#
# With LoRA, the forward pass is:
#   output = x @ W.T + x @ A.T @ B.T * (alpha / r)
#          = x @ (W + B @ A).T  * scaling
#
# Where:
#   W: (d_out, d_in)  — original frozen weights
#   A: (r, d_in)      — LoRA matrix A, initialized with random Gaussian
#   B: (d_out, r)     — LoRA matrix B, initialized with ZEROS
#
# Critical initialization:
#   B = 0 → B @ A = 0 at step 0 → no change to output at initialization
#   This means the model starts from its pretrained state — no disruption
#   A = random (so gradients flow from the start)
#
# Scaling factor (alpha / r):
#   alpha is a hyperparameter (commonly = r, so scaling = 1.0)
#   Scaling stabilizes training across different ranks

print("""
LoRA forward pass:

  Original:  output = x @ W.T              W is frozen (no gradient)
  LoRA:      output = x @ W.T + (x @ A.T @ B.T) * (alpha/r)
                                └─────────────────────────┘
                                    the LoRA adapter output
                                    (A and B are trained, W is not)

Initialization:
  A: random Gaussian (small values, e.g. std=1/sqrt(r))
  B: all zeros  →  B@A = 0 at start  →  output unchanged at init
""")

# Demonstrate with numbers
d_in, d_out = 4, 4
r_demo      = 2

W = torch.randn(d_out, d_in)          # original frozen weight
A = torch.randn(r_demo, d_in) * 0.01  # LoRA A: small random
B = torch.zeros(d_out, r_demo)         # LoRA B: zeros

x = torch.randn(3, d_in)              # batch of 3 inputs

out_original = x @ W.T
out_lora     = x @ W.T + (x @ A.T) @ B.T

print(f"Before any training (B=0):")
print(f"  Original output:     {out_original[0].tolist()}")
print(f"  LoRA output:         {out_lora[0].tolist()}")
print(f"  Difference:          {(out_original - out_lora).abs().max().item():.8f}  (should be 0)")
print(f"\n  ✓ LoRA doesn't change the output at initialization")


# =============================================================================
# PART 3: LORA FROM SCRATCH — PyTorch Implementation
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: LoRA Layer — Built From Scratch")
print("=" * 60)

class LoRALinear(nn.Module):
    """
    A Linear layer with LoRA adaptation.

    Wraps an existing nn.Linear and adds trainable low-rank matrices A and B.
    The original weight is FROZEN — only A and B are trained.

    Forward: output = W(x) + B(A(x)) * (alpha/r)
    """
    def __init__(self, linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.r     = r
        self.alpha = alpha
        self.scale = alpha / r     # scaling factor applied to LoRA output

        # Keep the original linear layer, but FREEZE its weights
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False   # frozen — will not be updated

        # Handle both nn.Linear and HF's Conv1D (GPT-2 uses Conv1D internally)
        # Conv1D stores weight transposed: shape (d_in, d_out) vs Linear (d_out, d_in)
        if hasattr(linear, 'in_features'):
            d_in, d_out = linear.in_features, linear.out_features
        else:
            # Conv1D: weight shape is (d_in, d_out)
            d_in, d_out = linear.weight.shape

        # LoRA matrices — these ARE trained (requires_grad=True by default)
        self.lora_A = nn.Linear(d_in,  r,     bias=False)
        self.lora_B = nn.Linear(r,     d_out, bias=False)

        # Initialize: A = small random (Kaiming), B = zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Original frozen path
        original = self.linear(x)                       # W(x)

        # LoRA adapter path
        lora_out = self.lora_B(self.lora_A(x)) * self.scale  # B(A(x)) * α/r

        return original + lora_out

    def trainable_params(self):
        return sum(p.numel() for p in [self.lora_A.weight, self.lora_B.weight])

    def total_params(self):
        return sum(p.numel() for p in self.parameters())


# Test it
d_in, d_out = 768, 768
r_test      = 8

original_linear = nn.Linear(d_in, d_out)   # simulates a GPT-2 attention projection
lora_linear     = LoRALinear(original_linear, r=r_test, alpha=16.0)

# Check what's trainable
frozen_params    = sum(p.numel() for p in original_linear.parameters())
trainable_params = lora_linear.trainable_params()
print(f"\nLoRALinear({d_in} → {d_out}, r={r_test}):")
print(f"  Original params (frozen): {frozen_params:,}  (W + bias)")
print(f"  LoRA params (trainable):  {trainable_params:,}  (A + B)")
print(f"  Reduction:                {frozen_params // trainable_params}×")
print(f"  Trainable %:              {trainable_params/frozen_params*100:.2f}%")

# Verify: only lora_A and lora_B have requires_grad=True
print(f"\nWhich parameters are trainable?")
for name, param in lora_linear.named_parameters():
    grad_str = "✓ trainable" if param.requires_grad else "✗ frozen"
    print(f"  {name:<35}  {str(param.shape):<20}  {grad_str}")

# Test forward pass
x_test = torch.randn(4, d_in)
out    = lora_linear(x_test)
print(f"\nForward pass: input {x_test.shape} → output {out.shape}")


# =============================================================================
# PART 4: APPLY LORA TO GPT-2 — MANUAL INJECTION
# =============================================================================
# Now inject LoRA into GPT-2's attention layers.
# We target the Q and V projection matrices (standard LoRA practice).
# K is often skipped as it changes attention patterns, V is more stable.

print("\n" + "=" * 60)
print("PART 4: Injecting LoRA into GPT-2")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Count original trainable params
orig_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nGPT-2 BEFORE LoRA injection:")
print(f"  Total parameters:    {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable:           {orig_trainable:,}")

# Freeze ALL parameters first
for param in model.parameters():
    param.requires_grad = False

# Inject LoRA into every attention layer's c_attn (combined Q,K,V projection)
# In GPT-2, c_attn is a single Linear(768, 2304) that produces Q+K+V together
LORA_RANK  = 8
LORA_ALPHA = 16
injected   = 0

for layer_idx, block in enumerate(model.transformer.h):
    # Replace the c_attn (query/key/value projection) with LoRA version
    original_attn = block.attn.c_attn   # Linear(768, 2304)
    block.attn.c_attn = LoRALinear(original_attn, r=LORA_RANK, alpha=LORA_ALPHA)
    injected += 1

    # Also inject into the output projection
    original_proj = block.attn.c_proj   # Linear(768, 768)
    block.attn.c_proj = LoRALinear(original_proj, r=LORA_RANK, alpha=LORA_ALPHA)
    injected += 1

print(f"\nGPT-2 AFTER LoRA injection ({injected} layers replaced):")

total_params     = sum(p.numel() for p in model.parameters())
trainable_after  = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_after     = total_params - trainable_after

print(f"  Total parameters:    {total_params:,}  (original + LoRA A,B)")
print(f"  Frozen (original W): {frozen_after:,}")
print(f"  Trainable (LoRA):    {trainable_after:,}")
print(f"  Trainable %:         {trainable_after/total_params*100:.2f}%")
print(f"  Reduction vs full:   {orig_trainable // trainable_after}×  fewer trainable params")

# Show per-layer breakdown
print(f"\n  Per-layer LoRA parameters:")
for i, block in enumerate(model.transformer.h[:3]):   # show first 3 layers
    attn_params = block.attn.c_attn.trainable_params()
    proj_params = block.attn.c_proj.trainable_params()
    print(f"    Layer {i}: c_attn={attn_params:,}  c_proj={proj_params:,}  "
          f"total={attn_params+proj_params:,}")
print(f"    ...  (×{injected//2} layers)")

# Test a forward pass still works
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer("The quick brown fox", return_tensors='pt')
with torch.no_grad():
    out = model(**inputs)
print(f"\nForward pass still works: logits shape = {out.logits.shape}")


# =============================================================================
# PART 5: LORA WITH PEFT LIBRARY
# =============================================================================
# The PEFT (Parameter-Efficient Fine-Tuning) library from HuggingFace
# does exactly what we did above — but with more options and less code.
# It's the standard tool for LoRA in production.

print("\n" + "=" * 60)
print("PART 5: PEFT Library — LoRA in Practice")
print("=" * 60)

from peft import LoraConfig, get_peft_model, TaskType

# Load fresh GPT-2
model_peft = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure LoRA
lora_config = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,    # next-token prediction task
    r              = 8,                      # rank
    lora_alpha     = 16,                     # scaling (alpha/r = 2.0)
    lora_dropout   = 0.05,                   # dropout on LoRA layers
    target_modules = ["c_attn", "c_proj"],   # which layers to apply LoRA to
    bias           = "none",                 # don't train bias terms
)

print(f"\nLoraConfig:")
print(f"  r (rank):         {lora_config.r}")
print(f"  alpha:            {lora_config.lora_alpha}")
print(f"  scale (α/r):      {lora_config.lora_alpha / lora_config.r:.1f}")
print(f"  dropout:          {lora_config.lora_dropout}")
print(f"  target modules:   {lora_config.target_modules}")

# Apply LoRA to the model — this wraps it exactly like our manual injection
model_peft = get_peft_model(model_peft, lora_config)

print(f"\nPEFT model summary:")
model_peft.print_trainable_parameters()

# Show the model structure — LoRA layers visible
print(f"\nFirst transformer block structure (layer 0):")
print(model_peft.base_model.model.transformer.h[0].attn)


# =============================================================================
# PART 6: MERGING LORA BACK INTO THE BASE MODEL
# =============================================================================
# After training, LoRA weights can be MERGED back into W:
#   W_merged = W + B @ A * scale
#
# This means you can:
#   1. Fine-tune with LoRA (fast, memory efficient)
#   2. Merge the adapter into the base model (no runtime overhead)
#   3. Deploy the merged model (same speed as the original)
#
# The merged model is identical to a fully fine-tuned model
# but was trained 100× more efficiently.

print("\n" + "=" * 60)
print("PART 6: Merging LoRA Into the Base Model")
print("=" * 60)

print("""
Training workflow with LoRA:

  1. Start:      W (pretrained, frozen)   +  A, B (random/zero)
                 output = W(x) + B(A(x)) * scale

  2. Train:      Only A and B update via gradients
                 W stays frozen throughout

  3. Merge:      W_final = W + B @ A * scale
                 Discard A and B — no longer needed

  4. Deploy:     W_final model — same architecture, same speed
                 Indistinguishable from a fully fine-tuned model
                 But trained with 100× fewer trainable parameters

  Storage:
    During training:  base model + tiny adapter  (store separately)
    After merging:    just the merged model       (single file)

  Multiple tasks:
    Base model (shared)  +  adapter_legal.safetensors   (~2 MB)
                         +  adapter_medical.safetensors  (~2 MB)
                         +  adapter_code.safetensors     (~2 MB)

    vs full fine-tuning: 3 × 474 MB = 1.4 GB
""")

# Demonstrate merging (with PEFT)
from peft import LoraConfig, get_peft_model
model_merge = AutoModelForCausalLM.from_pretrained("gpt2")
cfg         = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], bias="none",
                         task_type=TaskType.CAUSAL_LM)
model_merge = get_peft_model(model_merge, cfg)

# Simulate "after training" — manually set some non-zero B weights
for name, module in model_merge.named_modules():
    if hasattr(module, 'lora_B'):
        for key in module.lora_B:
            nn.init.normal_(module.lora_B[key].weight, std=0.01)

# Merge weights back into base model
merged_model = model_merge.merge_and_unload()
print(f"Before merge: PEFT model with adapters")
print(f"After merge:  {type(merged_model).__name__}  (standard HF model, adapters gone)")
print(f"Parameters:   {sum(p.numel() for p in merged_model.parameters()):,}  (same as original GPT-2)")
print(f"\nThe merged model can be used/saved exactly like a normal GPT-2.")


# =============================================================================
# PART 7: FULL COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: Full Comparison")
print("=" * 60)

model_sizes = [
    # name,        total_M,  full_ft_GB, lora_MB, lora_trainable_M
    ("GPT-2 small", 124,     0.5,        2,       0.3),
    ("GPT-2 XL",    1560,    6.2,        10,      3.0),
    ("Llama 3.2 1B",1000,    4.0,        8,       2.0),
    ("Llama 3.2 3B",3000,    12.0,       15,      5.0),
    ("Llama 3.2 8B",8000,    32.0,       25,      10.0),
    ("Llama 3.1 70B",70000,  280.0,      100,     70.0),
]

print(f"\n  {'Model':<18} {'Params':>8} {'Full FT':>10} {'LoRA store':>12} {'Trainable':>12}")
print(f"  {'-'*65}")
for name, total, full_gb, lora_mb, lora_train in model_sizes:
    m2_ok = "✓ M2" if full_gb <= 24 else "✗ M2"
    print(f"  {name:<18} {total:>6}M  {full_gb:>7.1f} GB  {lora_mb:>8} MB  {lora_train:>8}M  {m2_ok}")

print(f"""
  Full FT  = storage size of one fully fine-tuned model
  LoRA store = storage size of one LoRA adapter (just A+B matrices)
  Trainable = parameters that need gradients (determines RAM needed)
  M2 24GB  = whether fine-tuning fits in your machine's memory

  LoRA makes Llama 3.2 3B fine-tuning possible on your M2.
  Without LoRA, even GPT-2 XL would strain your memory.
""")


print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
LoRA in one paragraph:
  Freeze all original weights. Add two tiny matrices A (r×d) and B (d×r)
  to each target layer. Train only A and B. The output becomes
  W(x) + B(A(x)) × (alpha/r). B starts at zero so training begins
  from the exact pretrained state. After training, merge B@A back into W.
  Result: same quality as full fine-tuning, ~100× fewer trainable parameters,
  ~100× smaller adapter files, no catastrophic forgetting.

Key hyperparameters:
  r (rank)   →  higher = more capacity = more params (default: 8)
  alpha      →  scaling, set to r or 2r (default: 16 for r=8)
  target     →  which layers to adapt (c_attn, c_proj for attention)
  dropout    →  regularization on LoRA layers (0.05 is common)

Next: 04_finetune_with_lora.py
      Complete pipeline: LoRA fine-tuning on a real task,
      evaluate, save adapter, load and use it.
""")
