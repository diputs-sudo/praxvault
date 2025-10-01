# Gradient Descent

**Definition**  
Iterative optimization method that updates parameters in the **negative gradient** direction to reduce an objective (loss) function.

---

## Math

- **Update rule**  
  `θ_{t+1} = θ_t - η ∇L(θ_t)`  
  where `η > 0` is the **learning rate** (step size).

- **Convergence intuition**  
  - If `η` is **too large** → overshoot/diverge.  
  - If `η` is **too small** → slow progress.  
  - For smooth convex `L`, constant `η ≤ 1/L_smooth` ensures monotone decrease; diminishing `η_t` (e.g., `c/√t`) can guarantee convergence.

- **Variants (brief)**  
  - **Momentum**: accumulate a velocity to damp oscillations.  
  - **Nesterov**: look-ahead gradient for stability.  
  - **Schedules**: step, cosine, exponential, warmup.

---

## ML Relevance

- Foundation for **SGD, Adam, RMSProp**, etc.  
- Trains **linear regression**, **logistic regression**, and **neural networks** (with backprop).  
- Core loop: compute gradients of loss w.r.t. parameters and apply the update.

---

## Worked Example: Linear Regression via Gradient Descent (NumPy)

Objective (with optional L2):  
`L(w,b) = (1/N) ||y - (Xw + b)||² + λ ||w||²`

**Gradients**  
`∇_w L = (2/N) Xᵀ (Xw + b - y) + 2λ w`  
`∂L/∂b = (2/N) 1ᵀ (Xw + b - y)`

'''python
import numpy as np

def gd_linear_regression(X, y, lr=1e-2, steps=5000, lam=0.0):
    """
    Gradient Descent for linear regression with optional L2 regularization.
    X: (N, d), y: (N,)
    Returns: (w, b)
    """
    N, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for t in range(steps):
        # predictions and residuals
        r = (X @ w + b) - y                      # (N,)
        # gradients
        grad_w = (2.0/N) * (X.T @ r) + 2.0 * lam * w
        grad_b = (2.0/N) * np.sum(r)
        # update
        w -= lr * grad_w
        b -= lr * grad_b

        # optional: simple cosine decay of lr (illustrative)
        # lr_t = lr * (0.5 * (1 + np.cos(np.pi * t / steps)))

        # (optional) early stop if gradient small
        if (np.linalg.norm(grad_w) + abs(grad_b)) < 1e-8:
            break
    return w, b

# Demo with synthetic data
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, d = 300, 5
    X = rng.normal(size=(N, d))
    w_true = rng.normal(size=d)
    b_true = 0.7
    y = X @ w_true + b_true + 0.2 * rng.normal(size=N)

    w_hat, b_hat = gd_linear_regression(X, y, lr=5e-2, steps=2000, lam=1e-3)
    rmse = np.sqrt(np.mean((y - (X @ w_hat + b_hat))**2))
    print("RMSE:", rmse)
    print("||w_hat - w_true||:", np.linalg.norm(w_hat - w_true))
    print("b_hat:", b_hat, "  b_true:", b_true)
'''

---

## Mini-Batch Stochastic Gradient Descent (SGD)

Replace full gradients with mini-batch estimates for scalability.

'''python
import numpy as np

def sgd_linear_regression(X, y, lr=1e-2, epochs=20, batch=64, lam=0.0, seed=0):
    rng = np.random.default_rng(seed)
    N, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        idx = rng.permutation(N)
        Xp, yp = X[idx], y[idx]
        for s in range(0, N, batch):
            xb = Xp[s:s+batch]
            yb = yp[s:s+batch]
            r = (xb @ w + b) - yb
            grad_w = (2.0/len(yb)) * (xb.T @ r) + 2.0 * lam * w
            grad_b = (2.0/len(yb)) * np.sum(r)
            w -= lr * grad_w
            b -= lr * grad_b
    return w, b
'''

---

## Practical Tips

- **Standardize** features for stable steps (especially with SGD/L1).  
- Use **line search** or a **learning-rate range test** to pick `η`.  
- Prefer **solves (QR/SVD)** over explicit inverses when checking closed-form baselines.  
- Monitor **loss curve**; apply **early stopping** on validation loss.

---
