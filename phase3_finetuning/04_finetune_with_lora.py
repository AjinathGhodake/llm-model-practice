# =============================================================================
# PHASE 3 — LESSON 4: Fine-tuning With LoRA — Complete Pipeline
# =============================================================================
# This is the production workflow. Copy and adapt this for any project.
#
# Pipeline:
#   1. Prepare dataset in instruction format
#   2. Load base model + apply LoRA with PEFT
#   3. Train with Hugging Face Trainer
#   4. Save ONLY the adapter (few MB)
#   5. Load adapter back onto the base model
#   6. Compare before vs after — see domain shift
#   7. Merge and export for deployment
#
# Task: Python code assistant
#   - Teach GPT-2 to answer Python questions in a consistent format
#   - GPT-2 has general Python knowledge but no structured Q&A behaviour
#   - After LoRA: structured, domain-focused answers
# =============================================================================

import torch
import os
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset

torch.manual_seed(42)
device     = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL_NAME = "gpt2"
ADAPTER_DIR = "phase3_finetuning/gpt2-lora-python"

print(f"Device: {device}")
print(f"Base model: {MODEL_NAME}\n")


# =============================================================================
# 1. DATASET — Instruction Format
# =============================================================================
# Instruction format is how chat models are trained.
# The model learns: "when I see [INST]...[/INST], produce a structured answer"
#
# This is a simplified version of the format used to train Llama, Mistral, etc.
# Real instruction fine-tuning uses thousands of these pairs.
# We use 20 to demonstrate the mechanism clearly.

print("=" * 60)
print("1. DATASET")
print("=" * 60)

# Each example: question → focused Python answer
# Format chosen carefully: [INST] / [/INST] are standard delimiters
EXAMPLES = [
    ("How do I reverse a list in Python?",
     "Use my_list[::-1] for a reversed copy, or my_list.reverse() to reverse in place. "
     "The slice [::-1] works on any sequence including strings and tuples."),

    ("How do I check if a key exists in a dictionary?",
     "Use 'key in my_dict' which returns True or False. "
     "Avoid using my_dict.has_key() as it was removed in Python 3. "
     "For safe access with a default use my_dict.get(key, default)."),

    ("How do I remove duplicates from a list?",
     "Use list(set(my_list)) to remove duplicates but this loses order. "
     "To preserve order use list(dict.fromkeys(my_list)) in Python 3.7+. "
     "Both approaches are O(n) time complexity."),

    ("What is the difference between deepcopy and copy?",
     "copy.copy() creates a shallow copy: new object but nested objects are shared. "
     "copy.deepcopy() recursively copies all nested objects so nothing is shared. "
     "Use deepcopy when your structure contains mutable nested objects like lists."),

    ("How do I format a string in Python?",
     "Use f-strings: f'Hello {name}' for the most readable approach. "
     "Use str.format(): 'Hello {}'.format(name) for Python 2 compatibility. "
     "Use % formatting: 'Hello %s' % name only in legacy code."),

    ("How do I write a list to a file?",
     "Open the file with open('file.txt', 'w') and use writelines() or a loop with write(). "
     "Add newlines manually: f.write(item + newline). "
     "Always use the with statement to ensure the file is properly closed."),

    ("How do I merge two dictionaries in Python?",
     "In Python 3.9+ use the merge operator: merged = dict1 | dict2. "
     "In Python 3.5+ use unpacking: merged = {**dict1, **dict2}. "
     "Use dict1.update(dict2) to merge dict2 into dict1 in place."),

    ("How do I sort a dictionary by value?",
     "Use sorted() with a key: sorted(my_dict.items(), key=lambda x: x[1]). "
     "This returns a list of tuples. Wrap in dict() to get a dictionary back. "
     "Add reverse=True for descending order."),

    ("What does the zip function do?",
     "zip() combines multiple iterables element by element into tuples. "
     "zip([1,2,3], ['a','b','c']) gives (1,'a'), (2,'b'), (3,'c'). "
     "Use dict(zip(keys, values)) to create a dictionary from two lists."),

    ("How do I find the most common element in a list?",
     "Use collections.Counter: Counter(my_list).most_common(1)[0][0]. "
     "Counter counts occurrences of each element. most_common(n) returns top n. "
     "For the full frequency table use Counter(my_list)."),

    ("How do I flatten a nested list?",
     "Use a list comprehension: [x for sub in nested for x in sub] for one level deep. "
     "Use itertools.chain.from_iterable(nested) for a memory efficient approach. "
     "For deeply nested lists write a recursive function using isinstance checks."),

    ("How do I time my Python code?",
     "Use time.perf_counter() for the most accurate timing: "
     "start = time.perf_counter() then elapsed = time.perf_counter() - start. "
     "Use the timeit module for benchmarking small code snippets reliably."),

    ("What is the walrus operator?",
     "The walrus operator := assigns and returns a value in one expression. "
     "It was introduced in Python 3.8. Example: while chunk := f.read(8192): process(chunk). "
     "Use it sparingly to avoid reducing readability."),

    ("How do I use defaultdict?",
     "from collections import defaultdict. "
     "defaultdict(list) creates a dict where missing keys return an empty list. "
     "defaultdict(int) returns 0 for missing keys, useful for counting. "
     "Use it instead of checking if a key exists before appending."),

    ("How do I chain multiple iterables?",
     "Use itertools.chain(*iterables) to combine iterables without creating a new list. "
     "chain([1,2], [3,4], [5,6]) iterates as 1,2,3,4,5,6. "
     "It is memory efficient because it does not materialise the full sequence."),

    ("What is the purpose of __slots__?",
     "__slots__ restricts a class to a fixed set of attributes, saving memory. "
     "Define __slots__ = ['x', 'y'] in the class body to allow only x and y. "
     "Useful when creating millions of instances of a simple class."),

    ("How do I run code in parallel in Python?",
     "Use concurrent.futures.ThreadPoolExecutor for I/O bound tasks. "
     "Use concurrent.futures.ProcessPoolExecutor for CPU bound tasks. "
     "The GIL prevents true thread parallelism for CPU work, so use processes."),

    ("How do I make a class iterable?",
     "Implement __iter__ returning self and __next__ returning the next value. "
     "Raise StopIteration when iteration is complete. "
     "Alternatively write a generator function and return its result from __iter__."),

    ("What is the difference between a class method and a static method?",
     "@classmethod receives the class as first argument cls and can access class state. "
     "@staticmethod receives no implicit first argument and is just a regular function. "
     "Use classmethod for alternative constructors, staticmethod for utility functions."),

    ("How do I profile my Python code?",
     "Use cProfile: python -m cProfile -s cumulative my_script.py. "
     "Use line_profiler for line by line timing of specific functions. "
     "For memory use memory_profiler. Always profile before optimising."),
]

# Format each example as an instruction pair
def format_example(question, answer):
    return f"[INST] {question} [/INST] {answer}"

formatted = [format_example(q, a) for q, a in EXAMPLES]

print(f"Examples:    {len(formatted)}")
print(f"\nFormat used:")
print(f"  [INST] {{question}} [/INST] {{answer}}")
print(f"\nFirst example:")
print(f"  {formatted[0][:120]}...")


# =============================================================================
# 2. TOKENIZE
# =============================================================================

print("\n" + "=" * 60)
print("2. TOKENIZE")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 128

def tokenize(batch):
    return tokenizer(
        batch['text'],
        truncation    = True,
        padding       = 'max_length',
        max_length    = MAX_LENGTH,
        return_tensors = None,
    )

dataset    = Dataset.from_dict({'text': formatted})
tokenized  = dataset.map(tokenize, batched=True, remove_columns=['text'])
tokenized.set_format(type='torch')

# Token length stats
lengths = [sum(1 for t in ex['attention_mask'] if t == 1) for ex in tokenized]
print(f"\nDataset:     {len(tokenized)} examples")
print(f"Max length:  {MAX_LENGTH} tokens")
print(f"Avg tokens:  {sum(lengths)//len(lengths)} per example")
print(f"Max tokens:  {max(lengths)} (any truncation?  {'Yes' if max(lengths)==MAX_LENGTH else 'No'})")


# =============================================================================
# 3. BASE MODEL + LORA
# =============================================================================

print("\n" + "=" * 60)
print("3. BASE MODEL + LORA CONFIG")
print("=" * 60)

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
base_params = sum(p.numel() for p in base_model.parameters())

lora_config = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = 8,
    lora_alpha     = 16,
    lora_dropout   = 0.05,
    target_modules = ["c_attn", "c_proj"],
    bias           = "none",
)

model = get_peft_model(base_model, lora_config)

# Count trainable vs total
total_p     = sum(p.numel() for p in model.parameters())
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nBase model:         {base_params:,} parameters")
print(f"After LoRA:")
print(f"  Total:            {total_p:,}")
print(f"  Trainable (LoRA): {trainable_p:,}  ({trainable_p/total_p*100:.3f}%)")
print(f"  Frozen:           {total_p - trainable_p:,}")
print(f"\nMemory for gradients: ~{trainable_p * 4 / 1024**2:.1f} MB  "
      f"(vs {base_params * 4 / 1024**2:.0f} MB for full fine-tuning)")


# =============================================================================
# 4. GENERATE BEFORE TRAINING (BASELINE)
# =============================================================================

print("\n" + "=" * 60)
print("4. OUTPUT BEFORE FINE-TUNING")
print("=" * 60)

model.eval()

test_questions = [
    "[INST] How do I reverse a list in Python? [/INST]",
    "[INST] How do I merge two dictionaries? [/INST]",
    "[INST] What does the zip function do? [/INST]",
]

def generate(mdl, prompt, max_new_tokens=80, temperature=0.7, top_k=40):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    mdl_device = mdl.to(device) if hasattr(mdl, 'to') else mdl
    with torch.no_grad():
        out = mdl_device.generate(
            input_ids,
            max_new_tokens = max_new_tokens,
            do_sample      = True,
            temperature    = temperature,
            top_k          = top_k,
            pad_token_id   = tokenizer.eos_token_id,
        )
    new_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

print()
for q in test_questions:
    question_text = q.replace("[INST] ", "").replace(" [/INST]", "")
    answer = generate(model, q)
    print(f"Q: {question_text}")
    print(f"A (before): {answer[:180]}")
    print()


# =============================================================================
# 5. TRAIN
# =============================================================================

print("=" * 60)
print("5. TRAINING")
print("=" * 60)

model.train()

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm       = False,
)

training_args = TrainingArguments(
    output_dir                  = ADAPTER_DIR,
    num_train_epochs            = 8,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 2,   # effective batch = 4 × 2 = 8
                                       # accumulate gradients over 2 steps before updating
                                       # simulates a larger batch without more memory
    learning_rate               = 3e-4,   # higher than full FT — LoRA params start from scratch
    warmup_ratio                = 0.1,    # warm up for 10% of steps
    weight_decay                = 0.01,
    lr_scheduler_type           = "cosine",  # gradually reduce LR (cosine curve)
    logging_steps               = 5,
    save_strategy               = "no",
    report_to                   = "none",
    use_cpu                     = (device == 'cpu'),
    optim                       = "adamw_torch",
)

steps_per_epoch = len(tokenized) // (
    training_args.per_device_train_batch_size *
    training_args.gradient_accumulation_steps
)
total_steps = steps_per_epoch * int(training_args.num_train_epochs)

print(f"\nTraining config:")
print(f"  Epochs:                 {training_args.num_train_epochs}")
print(f"  Per-device batch:       {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation:  {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size:   {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate:          {training_args.learning_rate}")
print(f"  LR schedule:            {training_args.lr_scheduler_type}")
print(f"  Steps per epoch:        {steps_per_epoch}")
print(f"  Total steps:            {total_steps}")
print(f"\nTraining only {trainable_p:,} of {total_p:,} parameters ({trainable_p/total_p*100:.3f}%)\n")

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = tokenized,
    data_collator = data_collator,
)

import time
t0     = time.time()
result = trainer.train()
elapsed = time.time() - t0

print(f"\nTraining complete.")
print(f"  Loss:     {result.training_loss:.4f}")
print(f"  Steps:    {result.global_step}")
print(f"  Time:     {elapsed:.0f}s")
print(f"  Speed:    {result.metrics.get('train_samples_per_second', 0):.1f} samples/sec")


# =============================================================================
# 6. SAVE THE ADAPTER ONLY
# =============================================================================
# This is the key difference from full fine-tuning.
# model.save_pretrained() on a PEFT model saves ONLY the adapter weights.
# The base model stays on HuggingFace hub — no need to store it locally.

print("\n" + "=" * 60)
print("6. SAVE ADAPTER")
print("=" * 60)

os.makedirs(ADAPTER_DIR, exist_ok=True)
model.save_pretrained(ADAPTER_DIR)    # saves only adapter_model.safetensors
tokenizer.save_pretrained(ADAPTER_DIR)

saved = [f for f in os.listdir(ADAPTER_DIR) if os.path.isfile(os.path.join(ADAPTER_DIR, f))]
total_size = sum(os.path.getsize(os.path.join(ADAPTER_DIR, f)) for f in saved)

print(f"\nSaved to: {ADAPTER_DIR}/")
for f in sorted(saved):
    size = os.path.getsize(os.path.join(ADAPTER_DIR, f)) / (1024 * 1024)
    note = "← LoRA adapter weights" if "adapter_model" in f else \
           "← LoRA config (rank, alpha, targets)" if "adapter_config" in f else ""
    print(f"  {f:<45}  {size:.2f} MB  {note}")

print(f"\n  Total adapter size:  {total_size/1024**2:.2f} MB")
print(f"  Full fine-tune size: ~474 MB")
print(f"  Saving:              {474 / (total_size/1024**2):.0f}× smaller")


# =============================================================================
# 7. LOAD ADAPTER AND GENERATE AFTER TRAINING
# =============================================================================
# Production pattern:
#   1. Load base model from HF hub (once)
#   2. Load adapter from your storage (few MB)
#   3. Use the combined model

print("\n" + "=" * 60)
print("7. LOAD ADAPTER + OUTPUT AFTER FINE-TUNING")
print("=" * 60)

# Load fresh base model + attach saved adapter
base   = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
loaded = PeftModel.from_pretrained(base, ADAPTER_DIR)
loaded.eval()

print(f"\nLoaded: base GPT-2 + LoRA adapter from {ADAPTER_DIR}")
print(f"Adapter size on disk: {total_size/1024**2:.2f} MB\n")

print("Fine-tuned answers:\n")
for q in test_questions:
    question_text = q.replace("[INST] ", "").replace(" [/INST]", "")
    answer = generate(loaded, q)
    print(f"Q: {question_text}")
    print(f"A (after):  {answer[:200]}")
    print()


# =============================================================================
# 8. SIDE BY SIDE COMPARISON
# =============================================================================

print("=" * 60)
print("8. SIDE BY SIDE COMPARISON")
print("=" * 60)

# Fresh base model (no adapter) for clean comparison
base_clean = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
base_clean.eval()

compare_q = "[INST] How do I remove duplicates from a list? [/INST]"
question_text = "How do I remove duplicates from a list?"

before = generate(base_clean, compare_q, max_new_tokens=100)
after  = generate(loaded, compare_q, max_new_tokens=100)

print(f"\nQuestion: {question_text}\n")
print(f"BEFORE fine-tuning:")
print(f"  {before}")
print(f"\nAFTER LoRA fine-tuning:")
print(f"  {after}")

print(f"\n--- What changed ---")
print(f"  Before: general GPT-2 text, no Python structure")
print(f"  After:  domain-focused, Python-specific vocabulary")
print(f"  Trained:  {trainable_p:,} parameters  ({trainable_p/total_p*100:.3f}% of total)")
print(f"  Adapter:  {total_size/1024**2:.2f} MB saved  (vs 474 MB for full fine-tune)")


# =============================================================================
# 9. MERGE AND EXPORT FOR DEPLOYMENT
# =============================================================================

print("\n" + "=" * 60)
print("9. MERGE FOR DEPLOYMENT")
print("=" * 60)

MERGED_DIR = "phase3_finetuning/gpt2-lora-merged"
os.makedirs(MERGED_DIR, exist_ok=True)

# Merge LoRA weights into the base model
merged = loaded.merge_and_unload()
merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

merged_size = sum(
    os.path.getsize(os.path.join(MERGED_DIR, f))
    for f in os.listdir(MERGED_DIR)
    if os.path.isfile(os.path.join(MERGED_DIR, f))
) / 1024**2

print(f"\nMerged model saved to: {MERGED_DIR}/")
print(f"  Size: {merged_size:.1f} MB  (same as original GPT-2 — adapters absorbed)")
print(f"\nDeployment options:")
print(f"  1. Load merged model directly — no PEFT dependency needed")
print(f"  2. Use with Ollama for local serving")
print(f"  3. Push to HuggingFace Hub: merged.push_to_hub('your-model-name')")
print(f"  4. Quantize with llama.cpp for faster inference")


# =============================================================================
# 10. REUSABLE TEMPLATE SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("REUSABLE TEMPLATE — Copy This For Any Project")
print("=" * 60)
print(f"""
# ── 1. Prepare your data ──────────────────────────────────
examples = [format_example(q, a) for q, a in YOUR_DATA]
dataset  = Dataset.from_dict({{'text': examples}})
tokenized = dataset.map(tokenize_fn, batched=True)

# ── 2. Load model + LoRA ──────────────────────────────────
model = AutoModelForCausalLM.from_pretrained("YOUR_BASE_MODEL")
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],   # adjust per model
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 3. Train ──────────────────────────────────────────────
trainer = Trainer(model=model, args=training_args,
                  train_dataset=tokenized,
                  data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
trainer.train()

# ── 4. Save adapter only (few MB) ─────────────────────────
model.save_pretrained("my-adapter/")
tokenizer.save_pretrained("my-adapter/")

# ── 5. Load and use ───────────────────────────────────────
base    = AutoModelForCausalLM.from_pretrained("YOUR_BASE_MODEL")
model   = PeftModel.from_pretrained(base, "my-adapter/")
# generate text ...

# ── 6. Merge and deploy ───────────────────────────────────
merged = model.merge_and_unload()
merged.save_pretrained("my-merged-model/")

For Llama 3.2 3B instead of GPT-2, change only:
  MODEL_NAME = "meta-llama/Llama-3.2-3B"
  target_modules = ["q_proj", "v_proj"]   ← Llama uses different layer names
  Everything else stays the same.
""")


print("=" * 60)
print("PHASE 3 COMPLETE")
print("=" * 60)
print(f"""
Phase 3 summary:

  01_huggingface_intro.py   → HF ecosystem, BPE tokenizer, GPT-2 inference
  02_finetuning_basics.py   → full fine-tune, dataset prep, weight diff
  03_lora_explained.py      → LoRA math, scratch implementation, PEFT
  04_finetune_with_lora.py  → complete LoRA pipeline, adapter save/load

What you can now do:
  ✅ Load any HuggingFace model (GPT-2, Llama, Mistral)
  ✅ Prepare a custom dataset in instruction format
  ✅ Apply LoRA and train only 0.35% of parameters
  ✅ Save a ~2-15 MB adapter instead of a 500MB+ model
  ✅ Load the adapter back for inference
  ✅ Merge and deploy a production model

Your M2 can handle:
  ✅ GPT-2 (124M)      — demonstrated here
  ✅ Llama 3.2 1B      — change MODEL_NAME, same code
  ✅ Llama 3.2 3B      — needs ~8-12 GB with LoRA
  ⚠️  Llama 3.2 8B     — needs quantization (QLoRA) — next step

Next steps to explore:
  → QLoRA: 4-bit quantization + LoRA for even larger models
  → Llama 3.2 3B fine-tuning on YOUR actual domain data
  → Building a REST API around a fine-tuned model
  → RAG (Retrieval Augmented Generation) as an alternative to fine-tuning
""")
