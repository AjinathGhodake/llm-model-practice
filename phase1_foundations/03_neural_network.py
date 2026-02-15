# =============================================================================
# LESSON 3: Neural Networks
# =============================================================================
# A neural network is layers of weights + activation functions stacked together.
# PyTorch's nn.Module is the base class for ALL models — from a 2-layer network
# to GPT-4. Understanding this file means you understand the skeleton of every
# LLM ever built.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("PART 1: The Linear Layer — One Layer of a Network")
print("=" * 60)

# A Linear layer does exactly one thing:
#   output = input @ weight.T + bias
#
# That's it. Matrix multiply + add bias.
# The weight matrix and bias are the LEARNABLE parameters.
#
# nn.Linear(in_features, out_features)
#   in_features  = size of each input sample
#   out_features = size of each output sample

layer = nn.Linear(in_features=3, out_features=2)
# This created:
#   weight matrix: shape (2, 3) — randomly initialized
#   bias vector:   shape (2,)   — randomly initialized

print(f"\nLinear(3 → 2):")
print(f"  weight shape: {layer.weight.shape}")   # (out, in) = (2, 3)
print(f"  bias shape:   {layer.bias.shape}")     # (out,) = (2,)
print(f"  weight:\n{layer.weight.data}")
print(f"  bias: {layer.bias.data}")

# Pass an input through the layer
x = torch.tensor([1.0, 2.0, 3.0])              # input: 3 numbers
out = layer(x)                                  # applies: x @ weight.T + bias
print(f"\nInput:  {x}  shape={x.shape}")
print(f"Output: {out}  shape={out.shape}")       # 2 numbers out

# Manual verification (same math, done by hand)
manual = x @ layer.weight.T + layer.bias
print(f"Manual: {manual}  (should match output)")


print("\n" + "=" * 60)
print("PART 2: Activation Functions — Why Non-Linearity Matters")
print("=" * 60)

# Problem: stacking linear layers = still just one linear transformation.
# Linear(Linear(x)) = Linear(x). The layers collapse.
#
# Solution: add a non-linear activation function BETWEEN layers.
# This is what lets neural networks learn curves, boundaries, complex patterns.
#
# Three most common:

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"\nInput: {x}")

# --- ReLU: max(0, x) ---
# Negative values → 0, positive values → unchanged
# Most common in deep networks. Simple and effective.
relu_out = F.relu(x)
print(f"\nReLU:    {relu_out}")
print(f"  Rule: negative → 0, positive → keep")
print(f"  Used: between hidden layers")

# --- Sigmoid: 1 / (1 + e^-x) ---
# Squashes any value into range (0, 1)
# Output can be interpreted as a probability
sigmoid_out = torch.sigmoid(x)
print(f"\nSigmoid: {sigmoid_out.round(decimals=3)}")
print(f"  Rule: any value → squashed between 0 and 1")
print(f"  Used: binary classification output")

# --- Tanh: (e^x - e^-x) / (e^x + e^-x) ---
# Squashes any value into range (-1, 1)
# Zero-centered (better than sigmoid for hidden layers)
tanh_out = torch.tanh(x)
print(f"\nTanh:    {tanh_out.round(decimals=3)}")
print(f"  Rule: any value → squashed between -1 and 1")
print(f"  Used: hidden layers, RNNs")

# --- Softmax: turns a vector into probabilities that sum to 1 ---
# CRITICAL for LLMs — the final layer outputs a probability for each token
logits = torch.tensor([2.0, 1.0, 0.5])         # raw scores (logits)
probs = F.softmax(logits, dim=0)                # convert to probabilities
print(f"\nSoftmax:")
print(f"  logits: {logits}")
print(f"  probs:  {probs.round(decimals=3)}")
print(f"  sum:    {probs.sum().item():.4f}  (always 1.0)")
print(f"  In LLMs: each position = probability of that token being next")


print("\n" + "=" * 60)
print("PART 3: nn.Module — Building a Real Network")
print("=" * 60)

# nn.Module is the base class for ALL PyTorch models.
# You inherit from it, define layers in __init__, and define
# the forward pass in forward().
#
# PyTorch automatically:
#   - tracks all parameters (weights/biases) for gradient computation
#   - handles .to(device) to move everything to GPU/MPS
#   - handles .train() / .eval() mode switching

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()              # must call parent __init__

        # Define layers — these are registered as parameters automatically
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # This defines what happens when you call model(x)
        # Data flows: input → layer1 → ReLU → layer2 → ReLU → layer3 → output

        x = self.layer1(x)              # linear transform
        x = F.relu(x)                   # non-linear activation
        x = self.layer2(x)              # another linear transform
        x = F.relu(x)                   # another activation
        x = self.layer3(x)              # final layer — no activation here
        return x                        # raw scores (logits)


# Create the network: 4 inputs → 8 hidden → 8 hidden → 2 outputs
model = SimpleNetwork(input_size=4, hidden_size=8, output_size=2)
print(f"\nModel architecture:\n{model}")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")
# layer1: 4*8 + 8 = 40
# layer2: 8*8 + 8 = 72
# layer3: 8*2 + 2 = 18
# total:  40 + 72 + 18 = 130

# Forward pass
x = torch.rand(4)                       # one sample: 4 features
out = model(x)
print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {out.shape}")     # 2 scores

# Batched forward pass (how real training works)
batch = torch.rand(16, 4)              # 16 samples, each with 4 features
batch_out = model(batch)
print(f"\nBatch input shape:  {batch.shape}")
print(f"Batch output shape: {batch_out.shape}")  # 16 samples, each with 2 scores


print("\n" + "=" * 60)
print("PART 4: Loss Functions — Measuring Error")
print("=" * 60)

# Loss = a single number measuring how wrong the model is.
# Lower = better. 0 = perfect.
# Different tasks use different loss functions.

# --- MSE Loss: for regression (predicting continuous values) ---
# Mean of (prediction - target)^2
mse = nn.MSELoss()
pred = torch.tensor([2.5, 0.0, 2.0])
target = torch.tensor([3.0, -0.5, 2.0])
loss = mse(pred, target)
print(f"\nMSE Loss:")
print(f"  predictions: {pred}")
print(f"  targets:     {target}")
print(f"  loss:        {loss.item():.4f}")

# --- CrossEntropy Loss: for classification (picking one class) ---
# CRITICAL for LLMs — at each position, the model classifies which token comes next
# Combines Softmax + log + negative — so you pass raw logits (no softmax needed)
ce = nn.CrossEntropyLoss()
logits = torch.tensor([[2.0, 0.5, 0.1],     # sample 1 scores for 3 classes
                        [0.1, 3.0, 0.2]])    # sample 2 scores for 3 classes
labels = torch.tensor([0, 1])               # sample 1 = class 0, sample 2 = class 1
loss = ce(logits, labels)
print(f"\nCrossEntropy Loss:")
print(f"  logits: {logits.tolist()}")
print(f"  labels: {labels.tolist()}  (correct class indices)")
print(f"  loss:   {loss.item():.4f}")
print(f"  In LLMs: logits = scores for each token in vocabulary")
print(f"           labels = the actual next token index")


print("\n" + "=" * 60)
print("PART 5: Optimizers — Smarter Weight Updates")
print("=" * 60)

# In lesson 2 we updated weights manually: w -= lr * w.grad
# PyTorch's optimizers do this automatically AND smarter.
#
# SGD (Stochastic Gradient Descent): basic version of what we did manually
# Adam: adapts the learning rate per-parameter — almost always works better
#       (used in training GPT, BERT, basically all modern LLMs)

model = SimpleNetwork(input_size=4, hidden_size=8, output_size=2)

# Adam optimizer — pass model.parameters() so it knows what to update
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# One training step:
x = torch.rand(4)
target = torch.tensor([1.0, 0.0])

pred = model(x)                         # forward
loss = F.mse_loss(pred, target)         # loss
loss.backward()                         # backward — fills all .grad
optimizer.step()                        # update all weights
optimizer.zero_grad()                   # clear all gradients

print(f"\nAdam optimizer step complete.")
print(f"Loss: {loss.item():.4f}")
print(f"optimizer.zero_grad() clears gradients — same as manual w.grad.zero_()")


print("\n" + "=" * 60)
print("PART 6: Full Training Loop — Classification Task")
print("=" * 60)

# Putting it all together.
# Task: classify 2D points into 2 categories.
# This is a simple problem but uses IDENTICAL structure to LLM training.

torch.manual_seed(42)

# Generate synthetic data: 200 points, 2 classes
# Class 0: points near (1, 1)
# Class 1: points near (-1, -1)
n = 100
class0 = torch.randn(n, 2) + torch.tensor([1.0, 1.0])   # cluster near (1,1)
class1 = torch.randn(n, 2) + torch.tensor([-1.0, -1.0]) # cluster near (-1,-1)
X = torch.cat([class0, class1], dim=0)                   # (200, 2)
y = torch.cat([torch.zeros(n), torch.ones(n)]).long()    # (200,) labels: 0 or 1

# Build model, loss, optimizer
model = SimpleNetwork(input_size=2, hidden_size=16, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"\nTraining on {len(X)} samples, 2 classes")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}\n")

# Training loop
for epoch in range(100):
    # ----- FORWARD -----
    logits = model(X)                           # shape: (200, 2)

    # ----- LOSS -----
    loss = criterion(logits, y)

    # ----- BACKWARD -----
    loss.backward()                             # compute gradients

    # ----- UPDATE -----
    optimizer.step()                            # update weights
    optimizer.zero_grad()                       # clear gradients

    if epoch % 20 == 0 or epoch == 99:
        # Calculate accuracy
        with torch.no_grad():                   # no gradients needed for evaluation
            predictions = logits.argmax(dim=1)  # pick class with highest score
            accuracy = (predictions == y).float().mean()
        print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}, accuracy={accuracy.item():.2%}")

# Final evaluation
model.eval()                                    # switch to eval mode (disables dropout etc.)
with torch.no_grad():
    final_logits = model(X)
    final_preds = final_logits.argmax(dim=1)
    final_acc = (final_preds == y).float().mean()
    print(f"\nFinal accuracy: {final_acc.item():.2%}")


print("\n" + "=" * 60)
print("PART 7: Inspecting a Model")
print("=" * 60)

model = SimpleNetwork(4, 8, 2)
print(f"\nAll parameters and shapes:")
for name, param in model.named_parameters():
    print(f"  {name:20s}  shape={str(param.shape):20s}  requires_grad={param.requires_grad}")

print(f"\nModel in training mode: {model.training}")
model.eval()
print(f"After model.eval():     {model.training}")
model.train()
print(f"After model.train():    {model.training}")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key takeaways:
  1. nn.Linear(in, out)         → one layer = matrix multiply + bias
  2. Activation functions       → non-linearity between layers (ReLU most common)
  3. Without activation:        → stacked linear layers collapse into one
  4. nn.Module                  → base class for ALL models, define in __init__ + forward
  5. Loss functions             → MSE for regression, CrossEntropy for classification
  6. Optimizer (Adam)           → smarter gradient descent, almost always use this
  7. model.eval() / .train()    → switches inference vs training behavior

The training loop structure (forward → loss → backward → step → zero_grad)
is IDENTICAL in a 130-param toy network and a 70B LLM.

Next: 04_language_model_intro.py
      — what makes a language model different from a classifier?
      — tokens, vocabulary, predicting the next word
""")
