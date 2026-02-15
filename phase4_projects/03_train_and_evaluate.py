"""
Phase 4, Lesson 3: Training and Evaluating the Docstring Model
==============================================================

WHAT WE'RE DOING
----------------
1. Load the dataset we built in Lesson 2
2. Fine-tune GPT-2 with LoRA (same technique as Phase 3)
3. Evaluate systematically — rubric checks + before/after comparison
4. Save the final adapter

THE NEW PART: Evaluation
------------------------
In Phase 3, we just checked if the model generated text.
Here, we apply a real evaluation rubric to measure quality.
"""

import json
import math
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ============================================================
# PART 1: CONFIGURATION
# ============================================================

print("=" * 60)
print("PART 1: Configuration")
print("=" * 60)

# Pick device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

CONFIG = {
    "base_model": "gpt2",
    "adapter_output_dir": "phase4_projects/gpt2-docstring-lora",
    "train_data": "phase4_projects/data/train.jsonl",
    "val_data": "phase4_projects/data/val.jsonl",
    "max_length": 256,          # tokens per example
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["c_attn", "c_proj"],
    "num_train_epochs": 5,      # more epochs for a small dataset
    "per_device_train_batch_size": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 10,
    "device": DEVICE,
}

print(f"\nDevice: {DEVICE}")
for k, v in CONFIG.items():
    if k != "device":
        print(f"  {k}: {v}")

# ============================================================
# PART 2: LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("PART 2: Loading Dataset")
print("=" * 60)

def load_jsonl(path):
    """Load a JSONL file as a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f]

train_raw = load_jsonl(CONFIG["train_data"])
val_raw = load_jsonl(CONFIG["val_data"])

print(f"Train examples: {len(train_raw)}")
print(f"Val examples:   {len(val_raw)}")
print(f"\nFirst training example (first 200 chars of 'text'):")
print(f"  {train_raw[0]['text'][:200]}...")

# ============================================================
# PART 3: TOKENIZATION
# ============================================================

print("\n" + "=" * 60)
print("PART 3: Tokenizing")
print("=" * 60)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    """
    Tokenize examples for causal language modeling.

    We use the 'text' field which has the full [INST]...[/INST]... format.
    Truncate to max_length so all examples fit in memory.
    """
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=CONFIG["max_length"],
        padding=False,  # DataCollatorForLanguageModeling handles padding dynamically
    )
    # Don't set labels here — DataCollatorForLanguageModeling(mlm=False) does it
    return result

# Build HuggingFace datasets
train_texts = [ex["text"] for ex in train_raw]
val_texts = [ex["text"] for ex in val_raw]

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

train_tokenized = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
val_tokenized = val_dataset.map(tokenize, batched=True, remove_columns=["text"])

# Check token lengths
all_lengths = [len(x) for x in train_tokenized["input_ids"]]
print(f"Token lengths — min: {min(all_lengths)}, max: {max(all_lengths)}, avg: {sum(all_lengths)//len(all_lengths)}")

# ============================================================
# PART 4: BASELINE EVALUATION (before fine-tuning)
# ============================================================

print("\n" + "=" * 60)
print("PART 4: Baseline — What Does GPT-2 Generate Without Fine-tuning?")
print("=" * 60)

print("Loading base GPT-2 model...")
base_model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"])
base_model.eval()

# These are our test functions — NEVER seen during training
TEST_FUNCTIONS = [
    "def square(n):\n    return n * n",
    "def is_list(value):\n    return isinstance(value, list)",
    "def max_of_three(a, b, c):\n    return max(a, b, c)",
]

def generate_docstring(model, tokenizer, code, max_new_tokens=100, device="cpu"):
    """
    Given a function's source code, generate a docstring.
    Returns just the generated portion (after [/INST]).
    """
    prompt = f"[INST] Generate a Python docstring for this function:\n{code} [/INST]\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated.strip()


print("\nBaseline outputs (before fine-tuning):")
baseline_outputs = {}
for fn_code in TEST_FUNCTIONS:
    fn_name = fn_code.split("(")[0].replace("def ", "")
    output = generate_docstring(base_model, tokenizer, fn_code)
    baseline_outputs[fn_name] = output
    print(f"\n  Function: {fn_code.split(chr(10))[0]}")
    print(f"  GPT-2 says: {repr(output[:120])}")

# ============================================================
# PART 5: APPLY LORA AND TRAIN
# ============================================================

print("\n" + "=" * 60)
print("PART 5: Applying LoRA and Training")
print("=" * 60)

lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=CONFIG["target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"Frozen parameters:    {total_params - trainable_params:,}")

# Data collator — pads batches dynamically
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # we're doing causal LM, not masked LM
)

# Training arguments
# Note: MPS doesn't support fp16 well; use no fp16 on Mac
training_args = TrainingArguments(
    output_dir=CONFIG["adapter_output_dir"],
    num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=CONFIG["per_device_train_batch_size"],
    learning_rate=CONFIG["learning_rate"],
    warmup_steps=CONFIG["warmup_steps"],
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=False,         # MPS doesn't support fp16
    bf16=False,
    report_to="none",   # don't send to wandb/tensorboard
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print(f"\nStarting training ({CONFIG['num_train_epochs']} epochs)...")
start_time = time.time()

train_result = trainer.train()

elapsed = time.time() - start_time
print(f"\nTraining complete in {elapsed:.1f}s")
print(f"Final train loss: {train_result.training_loss:.4f}")
print(f"  (perplexity ≈ {math.exp(train_result.training_loss):.2f})")

# ============================================================
# PART 6: SAVE THE ADAPTER
# ============================================================

print("\n" + "=" * 60)
print("PART 6: Saving the Adapter")
print("=" * 60)

# Save only the LoRA adapter weights (not the full 474MB model!)
model.save_pretrained(CONFIG["adapter_output_dir"])
tokenizer.save_pretrained(CONFIG["adapter_output_dir"])

adapter_path = Path(CONFIG["adapter_output_dir"])
adapter_files = list(adapter_path.glob("*"))
total_size = sum(f.stat().st_size for f in adapter_files if f.is_file())
print(f"\nSaved adapter to: {adapter_path}")
print(f"Files: {[f.name for f in adapter_files if f.is_file()]}")
print(f"Total size: {total_size / 1024:.1f} KB  (vs 474MB for the full model!)")

# ============================================================
# PART 7: EVALUATION — BEFORE vs AFTER
# ============================================================

print("\n" + "=" * 60)
print("PART 7: Evaluation — Before vs After Fine-tuning")
print("=" * 60)

# Load the fine-tuned model for comparison
print("Loading fine-tuned model for evaluation...")
ft_model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"])
ft_model = PeftModel.from_pretrained(ft_model, CONFIG["adapter_output_dir"])
ft_model.eval()

print("\nComparing outputs on unseen test functions:\n")

after_outputs = {}
for fn_code in TEST_FUNCTIONS:
    fn_name = fn_code.split("(")[0].replace("def ", "")
    after = generate_docstring(ft_model, tokenizer, fn_code)
    after_outputs[fn_name] = after

    print(f"{'─' * 50}")
    print(f"Function:   {fn_code.split(chr(10))[0]}")
    print(f"\nBEFORE:  {repr(baseline_outputs[fn_name][:120])}")
    print(f"AFTER:   {repr(after[:200])}")
    print()

# ============================================================
# PART 8: RUBRIC-BASED EVALUATION
# ============================================================

print("=" * 60)
print("PART 8: Rubric Evaluation")
print("=" * 60)

print("""
Our evaluation rubric (from Lesson 1's plan):
  1. Does the first line end with a period? (complete sentence)
  2. Does it have "Args:" section when there are parameters?
  3. Does it have "Returns:" section when the function returns something?
  4. Is output length reasonable (10-500 chars)?
  5. Does the docstring mention the function's main action?
""")

def has_return_statement(code):
    """Check if a function has a return statement with a value."""
    lines = code.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("return ") and len(stripped) > 7:
            return True
    return False

def count_parameters(code):
    """Count the number of named parameters in a function definition."""
    import re
    match = re.search(r'def \w+\(([^)]*)\)', code)
    if not match:
        return 0
    params_str = match.group(1).strip()
    if not params_str:
        return 0
    params = [p.strip() for p in params_str.split(",")]
    # Exclude 'self' from count
    return sum(1 for p in params if p and p != "self")


def evaluate_docstring(code, docstring):
    """
    Apply our rubric to a generated docstring.
    Returns a score 0-5 and a list of passed/failed checks.
    """
    checks = {}
    score = 0

    # Check 1: Length reasonable
    checks["length_ok"] = 10 <= len(docstring) <= 500
    if checks["length_ok"]:
        score += 1

    # Check 2: First line looks like a sentence (starts with capital, has content)
    first_line = docstring.split("\n")[0].strip()
    checks["first_line_sentence"] = (
        len(first_line) > 5
        and first_line[0].isupper()
    )
    if checks["first_line_sentence"]:
        score += 1

    # Check 3: Has Args section (if function has parameters)
    param_count = count_parameters(code)
    if param_count > 0:
        checks["has_args"] = "Args:" in docstring
        if checks["has_args"]:
            score += 1
    else:
        checks["has_args"] = "N/A (no params)"
        score += 1  # no penalty

    # Check 4: Has Returns section (if function has return value)
    if has_return_statement(code):
        checks["has_returns"] = "Returns:" in docstring
        if checks["has_returns"]:
            score += 1
    else:
        checks["has_returns"] = "N/A (no return)"
        score += 1  # no penalty

    # Check 5: Doesn't just repeat the function name as the whole docstring
    fn_name = ""
    import re
    match = re.search(r'def (\w+)', code)
    if match:
        fn_name = match.group(1)
    checks["not_trivial"] = len(docstring) > len(fn_name) + 10
    if checks["not_trivial"]:
        score += 1

    return score, checks


print("Evaluating fine-tuned model on test functions:\n")
total_score = 0
for fn_code in TEST_FUNCTIONS:
    fn_name = fn_code.split("(")[0].replace("def ", "")
    docstring = after_outputs[fn_name]
    score, checks = evaluate_docstring(fn_code, docstring)
    total_score += score

    print(f"Function: {fn_code.split(chr(10))[0]}")
    print(f"Generated docstring (first line): {docstring.split(chr(10))[0]}")
    for check_name, result in checks.items():
        status = "✓" if result is True else ("─" if result == True or "N/A" in str(result) else "✗")
        if isinstance(result, bool):
            status = "✓" if result else "✗"
        print(f"  {status} {check_name}: {result}")
    print(f"  Score: {score}/5\n")

print(f"Average score: {total_score / len(TEST_FUNCTIONS):.1f}/5.0")

# ============================================================
# PART 9: VALIDATION LOSS SUMMARY
# ============================================================

print("=" * 60)
print("PART 9: Training Summary")
print("=" * 60)

print(f"""
Training stats:
  Total training time: {elapsed:.1f}s
  Final training loss: {train_result.training_loss:.4f}
  Final perplexity:    {math.exp(train_result.training_loss):.2f}

What this means:
  - Perplexity < 5 on training data means the model has learned the pattern
  - The rubric evaluation tells us if outputs are well-formed
  - The before/after comparison shows qualitative improvement

What we DIDN'T do (production would add these):
  - BLEU score comparison against reference docstrings
  - Testing on real-world Python code (not our training distribution)
  - A/B test with human raters
  - Adversarial examples (what if the function has a misleading name?)
""")

print(f"Adapter saved to: {CONFIG['adapter_output_dir']}")
print("\nReady for Lesson 4: Serving the model as an API!")
print("Run: uv run phase4_projects/04_serve_and_deploy.py")
