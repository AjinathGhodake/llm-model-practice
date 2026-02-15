# =============================================================================
# LESSON 1: Tensors
# =============================================================================
# A tensor is just a container for numbers, organized in dimensions.
# Every single thing in a neural network — inputs, weights, outputs —
# is a tensor. Understanding tensors = understanding the data flow.
# =============================================================================

import torch

print("=" * 60)
print("PART 1: What is a Tensor?")
print("=" * 60)

# --- Scalar: 0 dimensions, just one number ---
scalar = torch.tensor(7)
print(f"\nScalar:       {scalar}")
print(f"  Value:      {scalar.item()}")       # .item() extracts the raw Python number
print(f"  Shape:      {scalar.shape}")         # shape = () means 0 dimensions
print(f"  Dimensions: {scalar.ndim}")

# --- Vector: 1 dimension, a list of numbers ---
# In an LLM, a single word gets represented as a vector (called an embedding)
vector = torch.tensor([1.0, 2.0, 3.0])
print(f"\nVector:       {vector}")
print(f"  Shape:      {vector.shape}")         # shape = (3,) means 3 elements in 1 dimension
print(f"  Dimensions: {vector.ndim}")

# --- Matrix: 2 dimensions, a table of numbers ---
# In an LLM, a sentence (sequence of word vectors) is a matrix
# Rows = words/tokens, Columns = features per word
matrix = torch.tensor([
    [1.0, 2.0, 3.0],   # word 1 represented as 3 numbers
    [4.0, 5.0, 6.0],   # word 2 represented as 3 numbers
    [7.0, 8.0, 9.0],   # word 3 represented as 3 numbers
])
print(f"\nMatrix:\n{matrix}")
print(f"  Shape:      {matrix.shape}")         # shape = (3, 3) means 3 rows, 3 columns
print(f"  Dimensions: {matrix.ndim}")

# --- 3D Tensor: multiple matrices stacked ---
# In an LLM, a BATCH of sentences is a 3D tensor
# Dimension 0 = batch (how many sentences)
# Dimension 1 = sequence length (how many words per sentence)
# Dimension 2 = embedding size (how many numbers represent each word)
tensor_3d = torch.tensor([
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],   # sentence 1 (3 words, each with 2 features)
    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], # sentence 2 (3 words, each with 2 features)
])
print(f"\n3D Tensor:\n{tensor_3d}")
print(f"  Shape:      {tensor_3d.shape}")     # (2, 3, 2) = 2 sentences, 3 words, 2 features
print(f"  Dimensions: {tensor_3d.ndim}")


print("\n" + "=" * 60)
print("PART 2: Creating Tensors")
print("=" * 60)

# --- Zeros: all values are 0 ---
# Used to initialize weight matrices before training
zeros = torch.zeros(2, 3)              # 2 rows, 3 columns, all 0
print(f"\nZeros (2x3):\n{zeros}")

# --- Ones: all values are 1 ---
ones = torch.ones(2, 3)
print(f"\nOnes (2x3):\n{ones}")

# --- Random: values between 0 and 1 ---
# Used to initialize weights with randomness (important for training)
torch.manual_seed(42)                  # seed makes randomness reproducible
random = torch.rand(2, 3)
print(f"\nRandom (2x3):\n{random}")

# --- Range: like Python's range() ---
range_tensor = torch.arange(0, 10, 2)  # start=0, stop=10, step=2
print(f"\nRange (0 to 10, step 2): {range_tensor}")


print("\n" + "=" * 60)
print("PART 3: Tensor Data Types")
print("=" * 60)

# Why do data types matter?
# float32 = standard for training (balance of precision and speed)
# float16 = faster, less memory (used in production on GPU)
# int64   = used for token indices (word IDs are whole numbers)

float_tensor = torch.tensor([1.0, 2.0, 3.0])          # default: float32
int_tensor = torch.tensor([1, 2, 3])                   # default: int64

print(f"\nFloat tensor dtype: {float_tensor.dtype}")    # torch.float32
print(f"Int tensor dtype:   {int_tensor.dtype}")        # torch.int64

# Convert between types
float_to_int = float_tensor.to(torch.int64)
print(f"Converted to int:   {float_to_int}")


print("\n" + "=" * 60)
print("PART 4: Tensor Operations (the math behind neural networks)")
print("=" * 60)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# --- Element-wise operations ---
# Each element in a operates on the corresponding element in b
print(f"\na = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")                  # [5, 7, 9]
print(f"a * b = {a * b}")                  # [4, 10, 18]  (element-wise, NOT dot product)

# --- Dot product ---
# Multiply each pair, then sum everything
# This is the CORE operation inside attention mechanisms
dot = torch.dot(a, b)
print(f"\nDot product (a . b) = {dot}")    # 1*4 + 2*5 + 3*6 = 4+10+18 = 32
print(f"  (manual check: 1*4 + 2*5 + 3*6 = {1*4 + 2*5 + 3*6})")

# --- Matrix multiplication ---
# This is how layers in a neural network transform data
# Input vector (words) gets multiplied by weight matrix (learned parameters)
W = torch.tensor([                         # W = weight matrix (2x3)
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])
x = torch.tensor([2.0, 3.0, 4.0])         # x = input vector (3 elements)

# Matrix multiply: W @ x  →  transforms 3-element vector into 2-element vector
result = W @ x                             # @ is the matrix multiply operator
print(f"\nMatrix multiply W @ x:")
print(f"  W shape: {W.shape}")             # (2, 3)
print(f"  x shape: {x.shape}")             # (3,)
print(f"  result:  {result}")              # (2,) — shape changed!
print(f"  result shape: {result.shape}")


print("\n" + "=" * 60)
print("PART 5: Device — CPU vs MPS (Apple Silicon)")
print("=" * 60)

# On your M2, PyTorch can use MPS (Metal Performance Shaders)
# MPS = Apple's GPU backend — much faster than CPU for tensor operations
# CUDA = NVIDIA's GPU backend (not available on Mac)

# Check what's available
print(f"\nMPS available: {torch.backends.mps.is_available()}")

# Set device — this pattern is used in EVERY training script
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU) — fast!")
else:
    device = torch.device("cpu")
    print("Using CPU — slower but works")

# Move tensor to device
t = torch.tensor([1.0, 2.0, 3.0])
t_on_device = t.to(device)
print(f"\nTensor on {device}: {t_on_device}")
print(f"  Device: {t_on_device.device}")


print("\n" + "=" * 60)
print("PART 6: Shape Manipulation (critical for LLM code)")
print("=" * 60)

# LLM code constantly reshapes tensors to match what each layer expects
t = torch.arange(1, 13, dtype=torch.float32)   # [1, 2, 3, ..., 12]
print(f"\nOriginal: {t}  shape={t.shape}")      # (12,)

# reshape: reorganize the same data into a different shape
# rule: total elements must stay the same (12 = 3*4 = 2*6 = 2*2*3)
t_2d = t.reshape(3, 4)                          # 3 rows, 4 columns
print(f"\nReshaped (3,4):\n{t_2d}  shape={t_2d.shape}")

t_3d = t.reshape(2, 2, 3)                       # 2 batches, 2 rows, 3 columns
print(f"\nReshaped (2,2,3):\n{t_3d}  shape={t_3d.shape}")

# unsqueeze: add a dimension (used to add batch dimension)
x = torch.tensor([1.0, 2.0, 3.0])              # shape (3,)
x_batched = x.unsqueeze(0)                      # shape (1, 3) — added batch dimension
print(f"\nunsqueeze: {x.shape} → {x_batched.shape}")

# squeeze: remove a dimension of size 1
x_back = x_batched.squeeze(0)                   # shape (3,) again
print(f"squeeze:   {x_batched.shape} → {x_back.shape}")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key takeaways:
  1. Tensors are n-dimensional containers for numbers
  2. Shape = (batch, sequence_length, embedding_size) in LLMs
  3. Dot product and matrix multiply are the core math operations
  4. Always move tensors to the right device (MPS on your M2)
  5. Reshaping is constant — layers expect specific shapes

Next: 02_autograd.py — how PyTorch automatically calculates gradients
(this is how models LEARN from their mistakes)
""")
