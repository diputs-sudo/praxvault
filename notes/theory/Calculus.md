# Calculus

**Definition**  
- **Derivatives**: rate of change of a function w.r.t. its input.  
- **Gradients**: vector of partial derivatives for multivariate functions.  
- **Chain rule**: derivative of a composition `f(g(x))` is `f'(g(x)) * g'(x)` (generalizes to multivariate).

---

## Core Concepts

### Partial Derivatives
For `f(x1, x2, …, xd)`, the partial derivative w.r.t. `xj` treats other variables as constants:  
`∂f/∂xj`.

### Gradient Vector
`∇f(x) = [∂f/∂x1, …, ∂f/∂xd]ᵀ` — direction of steepest ascent.

### Jacobian
For vector-valued `f: ℝ^d → ℝ^m`, the Jacobian `J ∈ ℝ^{m×d}` stacks gradients of each output:  
`J_{ij} = ∂f_i/∂x_j`.

### Hessian
Matrix of second derivatives for scalar `f: ℝ^d → ℝ`:  
`H_{ij} = ∂²f / (∂x_i ∂x_j)` — captures local curvature (convexity/concavity).

---

## ML Relevance

- **Backpropagation** applies the **chain rule** through layers to compute gradients of the loss w.r.t. parameters.  
- **Optimization** (GD/SGD/Adam) uses **gradients** to minimize the loss.  
- **Second-order methods** (Newton/Quasi-Newton) use the **Hessian** or approximations for curvature-aware steps.

---

## Worked Math: Derivative of MSE wrt Linear Weights

Model: linear prediction `ŷ = Xw + b`, with data `X ∈ ℝ^{N×d}`, targets `y ∈ ℝ^N`.  
Mean Squared Error: `L(w,b) = (1/N) ||y - (Xw + b)||²`.

- Gradient w.r.t. `w`:  
  `∇_w L = (2/N) Xᵀ (Xw + b - y)`
- Gradient w.r.t. `b`:  
  `∂L/∂b = (2/N) 1ᵀ (Xw + b - y)`  (i.e., `2 * mean(residuals)`)

---

## Small Python Demos (Autograd)

### PyTorch: gradient of MSE wrt `w, b`
'''python
import torch

# synthetic data
torch.manual_seed(0)
N, d = 200, 5
X = torch.randn(N, d)
w_true = torch.randn(d)
b_true = 0.5
y = X @ w_true + b_true + 0.1 * torch.randn(N)

# parameters with gradient tracking
w = torch.zeros(d, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def mse(y_hat, y):  # (1/N) * ||y - y_hat||^2
    return torch.mean((y - y_hat)**2)

# forward
y_hat = X @ w + b
loss = mse(y_hat, y)

# backward: compute ∂loss/∂w and ∂loss/∂b
loss.backward()

print("loss:", loss.item())
print("grad w (autograd):", w.grad)
print("grad b (autograd):", b.grad)
# analytic check:
with torch.no_grad():
    grad_w_analytic = (2.0 / N) * X.t() @ (X @ w + b - y)
    grad_b_analytic = (2.0 / N) * torch.sum((X @ w + b - y))
    print("grad w (analytic):", grad_w_analytic)
    print("grad b (analytic):", grad_b_analytic)
'''

---

### PyTorch: Jacobian and Hessian (toy functions)
'''python
import torch
from torch.autograd.functional import jacobian, hessian

# Vector-valued function f: R^3 -> R^2
def f_vec(x):
    # x shape: (3,)
    x1, x2, x3 = x
    return torch.stack([
        x1**2 + 2*x2 - x3,          # f1
        torch.sin(x1) + x2*x3       # f2
    ])

# Scalar function g: R^3 -> R
def g_scalar(x):
    x1, x2, x3 = x
    return (x1**2) + 3*(x2**2) + torch.exp(x3) + x1*x2

x0 = torch.tensor([0.2, -0.3, 0.5], requires_grad=True)

J = jacobian(f_vec, x0)   # shape (2, 3)
H = hessian(g_scalar, x0) # shape (3, 3)

print("Jacobian(f) at x0:\n", J)
print("Hessian(g) at x0:\n", H)
'''

---

### Backprop-through-layers (chain rule in practice)
For a two-layer MLP: `h = σ(XW1 + b1)`, `ŷ = h W2 + b2`, loss `L(ŷ, y)`.  
Autograd applies chain rule to compute:
- `∂L/∂W2 = hᵀ ∂L/∂ŷ`
- `∂L/∂W1 = Xᵀ ((∂L/∂ŷ) W2ᵀ ⊙ σ'(XW1 + b1))`

'''python
import torch
import torch.nn as nn

torch.manual_seed(1)
N, d, hdim, k = 64, 20, 32, 1
X = torch.randn(N, d)
y = torch.randn(N, k)

mlp = nn.Sequential(
    nn.Linear(d, hdim),
    nn.ReLU(),
    nn.Linear(hdim, k)
)

opt = torch.optim.SGD(mlp.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

for step in range(5):
    opt.zero_grad(set_to_none=True)
    y_hat = mlp(X)
    loss = loss_fn(y_hat, y)
    loss.backward()    # autograd/chain rule
    opt.step()
    print(f"step {step+1}  loss={loss.item():.4f}")
'''

---

## Tips

- Prefer **automatic differentiation** (PyTorch/JAX) for complex models; verify with small analytic checks.  
- Use **double precision** (float64) when probing curvature (Hessian) on small problems.  
- For ill-conditioned losses, consider **gradient clipping**, **learning-rate schedules**, and (when appropriate) **second-order** approximations.

---
