# =============================================================================
# LESSON 2: Autograd — Automatic Differentiation
# =============================================================================
# This is the engine behind ALL neural network training.
# PyTorch silently watches every operation on tensors that have
# requires_grad=True. When you call .backward(), it walks back through
# every operation and computes how much each number contributed to the error.
# That information (the gradient) tells us how to adjust weights to reduce error.
# =============================================================================

import torch

print("=" * 60)
print("PART 1: What is a Gradient?")
print("=" * 60)

# Plain English:
# Imagine you're on a hilly landscape blindfolded.
# Your goal is to reach the lowest point (minimum error).
# The gradient tells you the SLOPE at your current position:
#   - positive gradient = you're on an uphill slope going right
#   - negative gradient = you're on a downhill slope going right
# To go downhill, move OPPOSITE to the gradient direction.
# That's gradient descent — the core of all model training.

# Simple example: y = x^2
# The gradient of x^2 is 2x (from calculus)
# At x=3, gradient = 6. This means: if x increases, y increases steeply.
# To minimize y, we should decrease x.

x = torch.tensor(3.0, requires_grad=True)
# requires_grad=True tells PyTorch: "watch this tensor, I'll need its gradient"

y = x ** 2   # y = x^2 = 9.0

print(f"\nx = {x}")
print(f"y = x^2 = {y}")
print(f"y.grad_fn = {y.grad_fn}")    # PyTorch recorded the operation (PowBackward)

# Compute gradients — walks backward through all recorded operations
y.backward()

print(f"\nAfter y.backward():")
print(f"x.grad = {x.grad}")          # dy/dx = 2x = 2*3 = 6.0  ← gradient!
print(f"Expected: 2 * {x.item():.0f} = {2 * x.item():.0f}")


print("\n" + "=" * 60)
print("PART 2: Computational Graph")
print("=" * 60)

# Every time you do math on a requires_grad tensor,
# PyTorch builds a "computational graph" — a record of operations.
# .backward() traverses this graph in reverse (chain rule from calculus).

# Example with multiple operations
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# Build a small computation
c = a * b       # c = 6.0,  dc/da = b = 3,  dc/db = a = 2
d = c + a       # d = 8.0,  dd/da = dc/da + 1 = 3+1 = 4,  dd/db = dc/db = 2
e = d ** 2      # e = 64.0, de/dd = 2d = 16

# de/da = de/dd * dd/da = 16 * 4 = 64
# de/db = de/dd * dd/db = 16 * 2 = 32

e.backward()    # compute all gradients at once

print(f"\na={a.item()}, b={b.item()}")
print(f"c = a*b = {c.item()}")
print(f"d = c+a = {d.item()}")
print(f"e = d^2 = {e.item()}")
print(f"\nGradients (how much each input affects e):")
print(f"  de/da = {a.grad.item()}  (expected: 64)")
print(f"  de/db = {b.grad.item()}  (expected: 32)")


print("\n" + "=" * 60)
print("PART 3: Gradient Descent — The Learning Step")
print("=" * 60)

# This is the actual update rule used in every neural network:
#
#   weight = weight - learning_rate * gradient
#
# learning_rate (lr): how big a step to take (e.g. 0.01)
# gradient: direction and steepness of the error slope
#
# Small lr = slow but stable learning
# Large lr = fast but can overshoot and diverge

# Toy problem: learn the value x that minimizes f(x) = (x - 5)^2
# The true minimum is at x=5 (error=0). Can gradient descent find it?

x = torch.tensor(0.0, requires_grad=True)   # start at x=0, far from x=5
learning_rate = 0.1

print(f"\nMinimizing f(x) = (x - 5)^2")
print(f"Starting at x={x.item():.4f}, target is x=5.0\n")

for step in range(20):
    # Forward pass: compute the loss (error)
    loss = (x - 5) ** 2

    # Backward pass: compute gradient of loss w.r.t. x
    loss.backward()

    # Gradient descent update step
    # We use torch.no_grad() here because this update is NOT a computation
    # we want PyTorch to track — it's just a manual weight update
    with torch.no_grad():
        x -= learning_rate * x.grad    # x = x - lr * gradient

    # CRITICAL: clear the gradient after using it
    # Gradients accumulate by default — must zero them out each step
    x.grad.zero_()

    if step % 5 == 0 or step == 19:
        print(f"  Step {step+1:2d}: x={x.item():.4f}, loss={loss.item():.4f}")

print(f"\nFinal x = {x.item():.4f}  (target was 5.0)")


print("\n" + "=" * 60)
print("PART 4: requires_grad — What PyTorch Tracks")
print("=" * 60)

# Only tensors with requires_grad=True get gradients computed.
# Model weights = requires_grad=True (we want to learn them)
# Input data    = requires_grad=False (we don't adjust the input)

weights = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
data    = torch.tensor([0.5, 1.5, 2.5], requires_grad=False)  # input data

output = (weights * data).sum()     # dot product
output.backward()

print(f"\nweights.grad = {weights.grad}")   # exists — we track weights
# data.grad would be None — we don't track inputs

# Check if a tensor is being tracked
print(f"\nweights requires_grad: {weights.requires_grad}")
print(f"data requires_grad:    {data.requires_grad}")


print("\n" + "=" * 60)
print("PART 5: torch.no_grad() — Turning Off Tracking")
print("=" * 60)

# During inference (using the model, not training it), we don't need gradients.
# Turning off tracking saves memory and speeds things up significantly.
# This is used in every model evaluation loop.

w = torch.tensor([1.0, 2.0], requires_grad=True)

# With tracking (training mode) — builds computational graph
out_tracked = (w * 2).sum()
print(f"\nWith grad tracking:    grad_fn={out_tracked.grad_fn}")  # has grad_fn

# Without tracking (inference mode) — no graph built, faster
with torch.no_grad():
    out_no_grad = (w * 2).sum()
    print(f"With no_grad:          grad_fn={out_no_grad.grad_fn}")  # None

# .detach() creates a new tensor with no gradient history
# useful when you want to use a tensor's value without backpropping through it
detached = out_tracked.detach()
print(f"Detached:              grad_fn={detached.grad_fn}")  # None


print("\n" + "=" * 60)
print("PART 6: Putting It Together — Mini Training Loop")
print("=" * 60)

# This is the skeleton of EVERY neural network training loop.
# Even GPT-4 training uses this exact same structure at its core.
#
# Problem: we have input x and want to learn weight w such that:
#   w * x ≈ target
# i.e., learn a single multiplication factor.

torch.manual_seed(42)

# True relationship we want to learn: target = 3.0 * x
true_weight = 3.0

# Our model's current (wrong) guess
w = torch.tensor(0.5, requires_grad=True)   # start with w=0.5, want to reach w=3.0

# Training data: 5 examples
x_data      = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
target_data = x_data * true_weight          # [3, 6, 9, 12, 15]

learning_rate = 0.01

print(f"\nLearning w such that w * x ≈ target")
print(f"Starting w={w.item():.4f}, target w={true_weight}\n")

for epoch in range(30):
    # ----- FORWARD PASS -----
    # Make prediction with current weight
    predictions = w * x_data

    # Calculate loss: Mean Squared Error (MSE)
    # MSE = average of (prediction - target)^2
    # Lower = better. 0 = perfect.
    loss = ((predictions - target_data) ** 2).mean()

    # ----- BACKWARD PASS -----
    # Compute gradient of loss w.r.t. w
    loss.backward()

    # ----- UPDATE STEP -----
    with torch.no_grad():
        w -= learning_rate * w.grad     # adjust w in direction that reduces loss

    # ----- ZERO GRADIENTS -----
    # Must clear gradients before next iteration, otherwise they accumulate
    w.grad.zero_()

    if epoch % 5 == 0 or epoch == 29:
        print(f"  Epoch {epoch+1:2d}: w={w.item():.4f}, loss={loss.item():.6f}")

print(f"\nLearned w = {w.item():.4f}  (true w = {true_weight})")
print(f"Predictions: {(w * x_data).detach().numpy().round(2)}")
print(f"Targets:     {target_data.numpy()}")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key takeaways:
  1. requires_grad=True  → PyTorch tracks all operations on this tensor
  2. loss.backward()     → computes gradient for every tracked tensor
  3. tensor.grad         → stores the computed gradient
  4. w -= lr * w.grad    → gradient descent: move weight to reduce loss
  5. w.grad.zero_()      → ALWAYS zero gradients after updating
  6. torch.no_grad()     → skip tracking for inference (faster)

The training loop (forward → loss → backward → update) is the same
in a 2-parameter model and in GPT-4. Only the scale changes.

Next: 03_neural_network.py — stack multiple layers to build a real network
""")
