# =============================================================================
# PHASE 3 — LESSON 1: Hugging Face Intro
# =============================================================================
# Hugging Face is the standard toolkit for working with pretrained LLMs.
# It wraps the exact same architecture you built in Phase 2,
# but pre-trained on massive data and packaged for practical use.
#
# Three core objects:
#   Tokenizer  → text → token IDs (BPE, not character-level)
#   Model      → the pretrained transformer (GPT-2, Llama, etc.)
#   Pipeline   → high-level: tokenize + model + decode in one call
#
# Model we use here: GPT-2 (117M params, ~500MB download)
#   - Same architecture as your nanoGPT, just bigger
#   - Trained on 40GB of internet text (WebText dataset)
#   - Free, widely understood, runs well on M2
# =============================================================================

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}\n")

MODEL_NAME = "gpt2"   # downloads ~500MB on first run, cached after


# =============================================================================
# PART 1: Tokenizer — Real BPE Tokenization
# =============================================================================
# In Phase 2, we used character-level tokenization: each character = 1 token.
# GPT-2 uses Byte Pair Encoding (BPE) — a smarter approach:
#
#   Character-level:  "hello" → ['h','e','l','l','o']  (5 tokens)
#   BPE:              "hello" → ['hello']               (1 token)
#   BPE:              "unhappiness" → ['un','happiness'] (2 tokens)
#
# BPE builds a vocabulary of common subword pieces.
# Common words = 1 token.  Rare words = split into pieces.
# GPT-2 vocabulary: 50,257 tokens (vs our 65 characters).
# This means the model can handle ANY text, not just Shakespeare.

print("=" * 60)
print("PART 1: BPE Tokenizer")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"\nVocabulary size: {tokenizer.vocab_size:,} tokens")
print(f"(vs our nanoGPT: 65 characters)")

# Tokenize some text
examples = [
    "Hello world",
    "unhappiness",
    "The quick brown fox",
    "def calculate_loss(predictions, targets):",
    "To be or not to be, that is the question",
]

print(f"\n{'Text':<45}  {'Tokens':<35}  {'IDs'}")
print("-" * 110)
for text in examples:
    token_ids  = tokenizer.encode(text)
    token_strs = tokenizer.convert_ids_to_tokens(token_ids)
    # Show token strings with Ġ replaced by space for readability
    readable   = [t.replace('Ġ', '▪') for t in token_strs]
    print(f"{text:<45}  {str(readable):<35}  {token_ids}")

# Decode back
ids    = tokenizer.encode("Hello world")
back   = tokenizer.decode(ids)
print(f"\nEncode then decode: 'Hello world' → {ids} → '{back}'")

# Special tokens
print(f"\nSpecial tokens:")
print(f"  EOS (end of sequence): id={tokenizer.eos_token_id}, '{tokenizer.eos_token}'")
print(f"  GPT-2 has no padding token by default — we add one for training")

# Add pad token (needed for batched training)
tokenizer.pad_token = tokenizer.eos_token
print(f"  Set pad_token = eos_token  (id={tokenizer.pad_token_id})")


# =============================================================================
# PART 2: Load the Pretrained Model
# =============================================================================
# AutoModelForCausalLM loads a decoder-only transformer (GPT-style).
# "Causal LM" = trained on next-token prediction (same as your nanoGPT).
# Other options:
#   AutoModelForSequenceClassification → for text classification
#   AutoModelForSeq2SeqLM              → for translation/summarization (T5, BART)
#   AutoModel                          → base model, no task head

print("\n" + "=" * 60)
print("PART 2: Loading a Pretrained Model")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

# Count parameters
total_params  = sum(p.numel() for p in model.parameters())
trainable     = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: {MODEL_NAME}")
print(f"  Total parameters:    {total_params:,}")
print(f"  Trainable parameters:{trainable:,}")
print(f"  Device:              {next(model.parameters()).device}")

# Inspect the architecture — same as your nanoGPT but bigger
print(f"\nArchitecture summary:")
print(f"  Embedding size:  {model.config.n_embd}")
print(f"  Attention heads: {model.config.n_head}")
print(f"  Layers:          {model.config.n_layer}")
print(f"  Context window:  {model.config.n_positions} tokens")
print(f"  Vocabulary:      {model.config.vocab_size:,} tokens")

# Compare with your nanoGPT
print(f"\nYour nanoGPT vs GPT-2 small:")
print(f"  {'Attribute':<20}  {'Your nanoGPT':>15}  {'GPT-2 small':>15}")
print(f"  {'-'*53}")
print(f"  {'Parameters':<20}  {'211,648':>15}  {total_params:>15,}")
print(f"  {'Embed size':<20}  {'64':>15}  {model.config.n_embd:>15}")
print(f"  {'Heads':<20}  {'4':>15}  {model.config.n_head:>15}")
print(f"  {'Layers':<20}  {'4':>15}  {model.config.n_layer:>15}")
print(f"  {'Context':<20}  {'128':>15}  {model.config.n_positions:>15}")
print(f"  {'Vocab':<20}  {'65 chars':>15}  {model.config.vocab_size:>15,}")


# =============================================================================
# PART 3: Manual Inference — Connecting to Phase 2
# =============================================================================
# This is the same forward pass you wrote in nanoGPT.
# Tokenize → model → logits → softmax → sample.
# Now you can see it's EXACTLY what you built, just scaled up.

print("\n" + "=" * 60)
print("PART 3: Manual Inference (connecting to Phase 2)")
print("=" * 60)

prompt     = "The meaning of life is"
input_ids  = tokenizer.encode(prompt, return_tensors='pt').to(device)
# return_tensors='pt' → return a PyTorch tensor directly

print(f"\nPrompt: '{prompt}'")
print(f"Token IDs: {input_ids.tolist()}")
print(f"Shape: {input_ids.shape}  (batch=1, seq_len={input_ids.shape[1]})")

# Forward pass — exactly like your nanoGPT's forward()
with torch.no_grad():
    outputs = model(input_ids)

logits = outputs.logits
print(f"\nLogits shape: {logits.shape}  (batch, seq_len, vocab_size)")
print(f"  = ({logits.shape[0]}, {logits.shape[1]}, {logits.shape[2]})")

# Look at the last position — that's the prediction for the NEXT token
last_logits  = logits[0, -1, :]               # shape: (vocab_size,)
probs        = torch.softmax(last_logits, dim=-1)
top_probs, top_ids = probs.topk(5)

print(f"\nTop 5 predicted next tokens after '{prompt}':")
for prob, tid in zip(top_probs, top_ids):
    token = tokenizer.decode([tid.item()])
    print(f"  '{token}' ({prob.item():.3f})")


# =============================================================================
# PART 4: Text Generation with model.generate()
# =============================================================================
# model.generate() is HF's built-in generation loop.
# Same logic as your nanoGPT generate() — just many more options.
#
# Key parameters:
#   max_new_tokens   → how many tokens to generate
#   do_sample        → True = sample (like your multinomial), False = greedy
#   temperature      → same as yours
#   top_k            → same as yours
#   top_p            → nucleus sampling: sample from tokens covering top p% probability

print("\n" + "=" * 60)
print("PART 4: Text Generation")
print("=" * 60)

prompts = [
    "Once upon a time",
    "The best way to learn programming is",
    "In the future, artificial intelligence will",
]

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens = 60,
            do_sample      = True,
            temperature    = 0.8,
            top_k          = 40,
            pad_token_id   = tokenizer.eos_token_id,
        )

    # Decode only the NEW tokens (not the prompt)
    new_tokens = output_ids[0][input_ids.shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\nPrompt:     '{prompt}'")
    print(f"Completion: '{completion}'")


# =============================================================================
# PART 5: Pipeline — The Simplest Interface
# =============================================================================
# pipeline() wraps tokenizer + model + postprocessing into one function.
# Useful for quick experiments. Under the hood: same tokenize → generate → decode.

print("\n" + "=" * 60)
print("PART 5: Pipeline — High-Level Interface")
print("=" * 60)

# text-generation pipeline
generator = pipeline(
    'text-generation',
    model     = model,
    tokenizer = tokenizer,
    device    = device,
)

results = generator(
    "Scientists have discovered",
    max_new_tokens    = 50,
    do_sample         = True,
    temperature       = 0.8,
    top_k             = 40,
    num_return_sequences = 2,   # generate 2 different continuations
    pad_token_id      = tokenizer.eos_token_id,
)

print(f"\nPrompt: 'Scientists have discovered'")
print(f"Two different completions (same prompt, sampled differently):\n")
for i, r in enumerate(results):
    print(f"  [{i+1}] {r['generated_text']}")
    print()


# =============================================================================
# PART 6: The Config — Understanding the Model
# =============================================================================
# Every HF model has a config object with all hyperparameters.
# This is exactly like the Config class you wrote in Phase 1.

print("=" * 60)
print("PART 6: Model Config — All Hyperparameters")
print("=" * 60)

print(f"\n{model.config}")


# =============================================================================
# PART 7: What Fine-tuning Changes
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: Pre-training vs Fine-tuning")
print("=" * 60)

print("""
PRE-TRAINING (done by OpenAI/Meta/etc):
  Data:     Trillions of tokens from the internet
  Goal:     Predict the next token (same as your nanoGPT training)
  Cost:     Millions of dollars, weeks of compute
  Result:   A model that understands language deeply

FINE-TUNING (what you do in Phase 3):
  Data:     Thousands of your domain-specific examples
  Goal:     Adapt the pre-trained knowledge to your task
  Cost:     Hours on your M2, nearly free
  Result:   A specialist that knows your domain

What actually changes during fine-tuning:
  The weights shift SLIGHTLY from their pre-trained values.
  The model retains its general language understanding.
  It gains specific knowledge about your domain/task.

  Pre-trained GPT-2 knows English.
  Fine-tuned GPT-2 knows English AND your specific task.

Full fine-tuning:   update ALL parameters (expensive, can forget old knowledge)
LoRA fine-tuning:   update only a tiny fraction (efficient, safer)

Next: 02_finetuning_basics.py
      Fine-tune GPT-2 on a custom dataset.
      See the weights shift. Watch domain knowledge appear.
""")
