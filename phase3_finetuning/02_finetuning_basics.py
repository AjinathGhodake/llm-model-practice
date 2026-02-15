# =============================================================================
# PHASE 3 — LESSON 2: Fine-tuning Basics
# =============================================================================
# Fine-tuning = take a pretrained model and adapt it to your domain.
#
# What changes:
#   - Same architecture (GPT-2)
#   - Same training loop (forward → loss → backward → step)
#   - Different starting point: pretrained weights instead of random
#   - Different data: YOUR domain instead of internet text
#   - Different scale: small learning rate, few steps
#
# This lesson:
#   1. Prepare a custom dataset (Python Q&A — GPT-2 doesn't know these)
#   2. Tokenize it for causal LM training
#   3. Fine-tune with Hugging Face Trainer
#   4. Compare output before vs after — see the domain shift
# =============================================================================

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import os

torch.manual_seed(42)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL_NAME    = "gpt2"
OUTPUT_DIR    = "phase3_finetuning/gpt2-python-tips"

print(f"Device: {device}")


# =============================================================================
# PART 1: THE DATASET — Custom Domain Data
# =============================================================================
# This is where YOUR domain knowledge lives.
# We create synthetic Python tips in Q&A format.
# GPT-2 was trained on general internet text — it does NOT know these
# specific tip formats and explanations.
#
# After fine-tuning, ask GPT-2 "Q: What does enumerate() do?" →
# it will answer in the style of this dataset.
#
# Real-world equivalents:
#   - Replace this with company documentation
#   - Replace with customer support transcripts
#   - Replace with medical Q&A, legal clauses, code from your codebase
#   The mechanism is identical.

print("=" * 60)
print("PART 1: Dataset")
print("=" * 60)

# Custom training data: Python tips as Q&A pairs
# Format: "Q: <question>\nA: <answer>\n###\n"
# The ### acts as a separator between examples (common convention)
RAW_DATA = [
    "Q: What does enumerate() do in Python?\nA: enumerate() adds a counter to an iterable. It returns pairs of (index, value). Example: for i, v in enumerate(['a','b','c']): gives (0,'a'), (1,'b'), (2,'c'). Use it whenever you need both the index and value in a loop.\n###",
    "Q: What is the difference between a list and a tuple?\nA: Lists are mutable (you can change them after creation) while tuples are immutable (fixed once created). Use lists when data changes, tuples when data is fixed. Tuples are slightly faster and can be used as dictionary keys.\n###",
    "Q: How does list comprehension work?\nA: List comprehension creates a new list in one line: [expression for item in iterable if condition]. Example: squares = [x**2 for x in range(10) if x % 2 == 0] gives [0, 4, 16, 36, 64]. It is faster and more readable than a for loop.\n###",
    "Q: What is a Python decorator?\nA: A decorator is a function that wraps another function to add behaviour. Use @decorator_name above a function definition. Example: @staticmethod, @classmethod, @property. You can write custom decorators to add logging, timing, or authentication to any function.\n###",
    "Q: How do you handle exceptions in Python?\nA: Use try/except blocks. Put risky code in try, handle errors in except. Use finally for cleanup that always runs. Use specific exception types like ValueError, TypeError for precise handling. Avoid bare except which catches everything including system exits.\n###",
    "Q: What is the difference between append and extend?\nA: append() adds one item to the end of a list: my_list.append(4) adds a single element. extend() adds all items from an iterable: my_list.extend([4,5,6]) adds three elements. Using append with a list adds the list as a single nested element.\n###",
    "Q: How does Python's with statement work?\nA: The with statement manages resources automatically. It calls __enter__ when entering the block and __exit__ when leaving, even if an error occurs. Most commonly used for file handling: with open('file.txt') as f: ensures the file is always closed properly.\n###",
    "Q: What is *args and **kwargs?\nA: *args collects extra positional arguments into a tuple. **kwargs collects extra keyword arguments into a dictionary. Use them to write functions that accept variable numbers of arguments. Convention: you can use any name, the * and ** are what matter.\n###",
    "Q: How do you copy a list in Python?\nA: Never use b = a for a copy as both variables point to the same list. Use b = a.copy() or b = a[:] for a shallow copy. Use copy.deepcopy(a) for a deep copy when the list contains mutable objects like other lists. Modifying b will not affect a after a proper copy.\n###",
    "Q: What is a lambda function?\nA: A lambda is an anonymous single-expression function: lambda x: x * 2. Use lambdas for short throwaway functions, often passed to sorted(), map(), or filter(). For anything longer than one expression, use a regular def function for readability.\n###",
    "Q: How does Python's garbage collection work?\nA: Python uses reference counting as its primary mechanism. When an object's reference count drops to zero it is immediately freed. A cyclic garbage collector handles reference cycles. You rarely need to manage memory manually, but del removes a reference and gc.collect() forces collection.\n###",
    "Q: What is the difference between is and ==?\nA: == checks value equality: [1,2] == [1,2] is True. is checks identity, whether two variables point to the same object in memory. Only use is to compare with None, True, and False. Never use is for string or integer comparison as the result is implementation-dependent.\n###",
    "Q: How do you read a file in Python?\nA: Use with open('filename.txt', 'r') as f: to open for reading. Call f.read() for the whole file as a string, f.readlines() for a list of lines, or iterate with for line in f: for memory-efficient line-by-line reading. Always use the with statement to ensure the file is closed.\n###",
    "Q: What is a generator in Python?\nA: A generator is a function that yields values one at a time using the yield keyword instead of return. It does not compute all values upfront, making it memory efficient for large sequences. Example: def count_up(n): for i in range(n): yield i. Use next() to get values or iterate with for.\n###",
    "Q: How do you sort a list of dictionaries?\nA: Use sorted() with a key function: sorted(my_list, key=lambda x: x['name']). For multiple keys use a tuple: key=lambda x: (x['age'], x['name']). Add reverse=True for descending order. The key function is called once per element, making it efficient.\n###",
]

print(f"\nTraining examples: {len(RAW_DATA)}")
print(f"\nFirst example:")
print(f"  {RAW_DATA[0][:100]}...")
print(f"\nThis data format teaches the model:")
print(f"  - Questions start with 'Q:'")
print(f"  - Answers start with 'A:'")
print(f"  - Examples end with '###'")


# =============================================================================
# PART 2: TOKENIZE THE DATASET
# =============================================================================
# For causal LM fine-tuning, we tokenize the full text and train the model
# to predict the next token at every position — same as pretraining,
# just on your smaller domain dataset.
#
# Key decisions:
#   max_length: max tokens per example (truncate or pad to this)
#   padding:    pad shorter examples to max_length for batching

print("\n" + "=" * 60)
print("PART 2: Tokenization")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token

def tokenize(examples):
    """
    Tokenize a batch of text examples.
    truncation=True  → cut sequences longer than max_length
    padding='max_length' → pad shorter sequences to max_length
    return_tensors is NOT set here — Datasets handles that
    """
    return tokenizer(
        examples['text'],
        truncation = True,
        padding    = 'max_length',
        max_length = 128,
    )

# Build HuggingFace Dataset from our list of strings
raw_dataset = Dataset.from_dict({'text': RAW_DATA})
print(f"\nRaw dataset: {raw_dataset}")

# Tokenize all examples
tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=['text'])
print(f"Tokenized dataset: {tokenized}")
print(f"\nColumns: {tokenized.column_names}")
print(f"  input_ids:      token IDs for each example")
print(f"  attention_mask: 1 = real token, 0 = padding")

# Show one tokenized example
example = tokenized[0]
print(f"\nFirst example — token IDs (first 30): {example['input_ids'][:30]}")
decoded = tokenizer.decode(example['input_ids'][:50])
print(f"Decoded (first 50 tokens): '{decoded}'")

# For causal LM, labels = input_ids (model predicts the next token at every position)
# The DataCollator handles setting labels = input_ids automatically
print(f"\nFor causal LM: labels = input_ids (model learns to predict every next token)")


# =============================================================================
# PART 3: COMPARE OUTPUT BEFORE FINE-TUNING
# =============================================================================
# Generate output on our Q&A prompts BEFORE fine-tuning.
# This is the baseline — GPT-2's default behaviour.

print("\n" + "=" * 60)
print("PART 3: Output BEFORE Fine-tuning (Baseline)")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

test_prompts = [
    "Q: What does enumerate() do in Python?\nA:",
    "Q: What is a lambda function?\nA:",
    "Q: What is the difference between append and extend?\nA:",
]

def generate(model, prompt, max_new_tokens=80):
    """Generate a completion for a given prompt."""
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            max_new_tokens  = max_new_tokens,
            do_sample       = True,
            temperature     = 0.7,
            top_k           = 40,
            pad_token_id    = tokenizer.eos_token_id,
        )
    # Return only the newly generated tokens (not the prompt)
    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\nBase GPT-2 answers (no fine-tuning):\n")
for prompt in test_prompts:
    question = prompt.split('\n')[0]
    answer   = generate(model, prompt)
    print(f"  {question}")
    print(f"  → {answer[:150]}")
    print()


# =============================================================================
# PART 4: FINE-TUNE WITH HUGGING FACE TRAINER
# =============================================================================
# The Trainer class handles the training loop for you.
# Under the hood it does exactly what you wrote in Phase 1:
#   for batch in dataloader:
#       loss = model(**batch).loss
#       optimizer.zero_grad()
#       loss.backward()
#       optimizer.step()
#
# TrainingArguments is the Config class equivalent.

print("=" * 60)
print("PART 4: Fine-tuning")
print("=" * 60)

# Reload model in training mode (eval() would be wrong here)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# DataCollator: handles batching and setting labels = input_ids automatically
# mlm=False → causal LM (not masked LM like BERT)
data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm       = False,     # GPT-style: predict next token (not masked token)
)

# TrainingArguments: all hyperparameters in one place
training_args = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    num_train_epochs        = 5,          # pass over the full dataset 5 times
    per_device_train_batch_size = 4,      # 4 examples per step
    learning_rate           = 2e-5,       # MUCH smaller than pretraining (3e-4)
                                          # Small LR = gentle nudge, preserve prior knowledge
    warmup_steps            = 10,         # gradually increase LR at the start
    weight_decay            = 0.01,       # L2 regularization
    logging_steps           = 10,         # log loss every 10 steps
    save_strategy           = "no",       # don't auto-save checkpoints
    use_cpu                 = (device == 'cpu'),
    # Note: HF Trainer auto-detects MPS on Apple Silicon
    report_to               = "none",     # disable wandb/tensorboard logging
)

print(f"\nFine-tuning config:")
print(f"  Epochs:        {training_args.num_train_epochs}")
print(f"  Batch size:    {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}  (vs pretraining ~3e-4)")
print(f"  Examples:      {len(tokenized)}")
print(f"  Steps/epoch:   {len(tokenized) // training_args.per_device_train_batch_size}")
print(f"  Total steps:   {len(tokenized) // training_args.per_device_train_batch_size * int(training_args.num_train_epochs)}")

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = tokenized,
    data_collator = data_collator,
)

print(f"\nTraining...")
train_result = trainer.train()
print(f"\nTraining complete.")
print(f"  Final loss:     {train_result.training_loss:.4f}")
print(f"  Total steps:    {train_result.global_step}")
print(f"  Runtime:        {train_result.metrics['train_runtime']:.0f}s")


# =============================================================================
# PART 5: COMPARE OUTPUT AFTER FINE-TUNING
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Output AFTER Fine-tuning")
print("=" * 60)

model.eval()

print("\nFine-tuned GPT-2 answers:\n")
for prompt in test_prompts:
    question = prompt.split('\n')[0]
    answer   = generate(model, prompt)
    print(f"  {question}")
    print(f"  → {answer[:200]}")
    print()


# =============================================================================
# PART 6: WHAT CHANGED IN THE WEIGHTS
# =============================================================================
# Load the original weights and compare with fine-tuned weights.
# This shows the fine-tuning actually shifted the weights.

print("=" * 60)
print("PART 6: What Changed — Weight Diff")
print("=" * 60)

original_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print(f"\nWeight change after fine-tuning (sample of layers):")
print(f"\n  {'Layer':<45}  {'Max change':>12}  {'Mean change':>12}")
print(f"  {'-'*72}")

total_change = 0
n_params     = 0

for (name, orig_param), (_, ft_param) in zip(
    original_model.named_parameters(),
    model.named_parameters()
):
    diff      = (orig_param.data.cpu() - ft_param.data.cpu()).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()
    total_change += diff.sum().item()
    n_params += diff.numel()

    # Show a selection of layers
    if any(x in name for x in ['wte', 'wpe', 'h.0.', 'h.5.', 'h.11.', 'ln_f']):
        print(f"  {name:<45}  {max_diff:>12.6f}  {mean_diff:>12.8f}")

avg_change = total_change / n_params
print(f"\n  Average change across ALL {n_params:,} parameters: {avg_change:.8f}")
print(f"\n  Small changes → weights shifted gently from pretrained values.")
print(f"  The model still knows English — it just also knows Python tips now.")


# =============================================================================
# PART 7: SAVE THE FINE-TUNED MODEL
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: Saving")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Check what was saved
saved_files = os.listdir(OUTPUT_DIR)
total_size  = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in saved_files
    if os.path.isfile(os.path.join(OUTPUT_DIR, f))
)
print(f"\nSaved to: {OUTPUT_DIR}/")
for f in sorted(saved_files):
    fpath = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath) / (1024*1024)
        print(f"  {f:<40}  {size:.1f} MB")
print(f"  Total: {total_size/(1024*1024):.1f} MB")

# How to reload:
print(f"""
To reload this fine-tuned model anywhere:

  from transformers import AutoTokenizer, AutoModelForCausalLM
  model     = AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}')
  tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')
""")


# =============================================================================
# PART 8: LIMITATIONS OF FULL FINE-TUNING
# =============================================================================

print("=" * 60)
print("PART 8: Limitations of Full Fine-tuning")
print("=" * 60)
print(f"""
WHAT WORKED:
  ✅ Model learned the Q: / A: / ### format
  ✅ Answers became more relevant to Python
  ✅ Domain vocabulary appeared in outputs

PROBLEMS WITH FULL FINE-TUNING:

  1. Catastrophic forgetting
     All 124M parameters updated → model may "forget" other knowledge
     Ask it about history or cooking after fine-tuning → degraded performance
     The more you train, the more general knowledge erodes

  2. Cost at scale
     GPT-2 (124M) → manageable
     Llama 3.2 (3B) → 3B gradients per step, high memory
     Llama 3.2 (7B) → barely fits in 24GB even with tricks
     GPT-3 (175B)   → impossible on consumer hardware

  3. Storage per fine-tuned model
     Each fine-tuned model = full copy of all weights (~500MB for GPT-2)
     10 fine-tuned models = 10 × 500MB = 5GB
     At Llama-7B scale: 10 models = 140GB

THE SOLUTION: LoRA (Low-Rank Adaptation)
  → Train only a tiny fraction of parameters (0.1% instead of 100%)
  → Freeze all original weights — no catastrophic forgetting
  → Store only the small LoRA adapter (~few MB instead of ~500MB)
  → Multiple tasks = multiple small adapters, one base model

Next: 03_lora_explained.py
      LoRA from first principles — understand the math,
      then see why it works so well.
""")
