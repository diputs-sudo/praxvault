# Momentum (SGD with Momentum & Nesterov)

**Momentum** augments first-order gradient methods by accumulating a running velocity of past gradients.  
This accelerates optimization along consistent directions and damps oscillations across high-curvature axes, yielding faster, more stable convergence than vanilla SGD.

---

## 1) Overview and Motivation

- **Problem with vanilla SGD:** noisy, zig-zag trajectories in ravines and slow progress in shallow directions.  
- **Momentum idea:** maintain an exponentially weighted moving average (EWMA) of gradients—called **velocity**—to keep moving where the gradient consistently points.  
- **Benefits:** faster convergence, reduced oscillations, improved conditioning tolerance, and better use of larger learning rates under schedules.

---

## 2) Algorithm and Notation

Let parameters be `θ`, learning rate `η`, momentum coefficient `μ ∈ [0,1)`, velocity `v`, gradient at step `t` be `g_t = ∇L(θ_{t-1})`.

### 2.1 Classical Momentum (Polyak)
- Velocity update:  
  `v_t = μ v_{t-1} + g_t`
- Parameter update:  
  `θ_t = θ_{t-1} - η v_t`
- Initialization: `v_0 = 0`.

Interpretation: `v_t` is an EWMA of recent gradients with decay `μ`, turning per-step “kicks” into smooth motion.

### 2.2 Nesterov Accelerated Gradient (NAG)
Nesterov evaluates the gradient at a **look-ahead** position, improving stability and sometimes speed.

Two equivalent views (common in DL practice):

- **Look-ahead gradient:**  
  `v_t = μ v_{t-1} + ∇L(θ_{t-1} - η μ v_{t-1})`  
  `θ_t = θ_{t-1} - η v_t`

- **Framework-style (PyTorch-like):**  
  1) `v_t = μ v_{t-1} + g_t`  
  2) `θ_t = θ_{t-1} - η (μ v_t + g_t)`  
  Here `g_t` is computed at the current `θ_{t-1}`; the extra `μ v_t` implements the look-ahead correction.

---

## 3) Intuition and Geometry

- **In low-curvature directions** (consistent gradients), `v` accumulates and accelerates progress.  
- **In high-curvature directions** (oscillatory gradients), past gradients cancel out, reducing step size and oscillations.  
- **Physical analogy:** a ball rolling down a landscape with friction coefficient `(1-μ)`—velocity builds along the slope and is damped by friction.

---

## 4) Hyperparameters

- **Learning rate `η`:** primary scale of updates.  
- **Momentum `μ`:** memory of past gradients. Common values: `0.9` (default), `0.95`, up to `0.99` for very smooth regimes.
- **Weight decay `λ`:** regularization; with SGD, using decoupled weight decay is straightforward: `θ ← θ - η λ θ` after the gradient step.
- **Batch size:** interacts with noise; momentum helps tame small-batch noise.

Guidance: start with `η = 0.1` on normalized vision tasks with `μ = 0.9`; tune `η` first, then `μ`.

---

## 5) Learning-Rate Schedules

Momentum shines with schedules:

- **Step decay:** reduce `η` by ×0.1 at milestones (e.g., 30/60/90 epochs).  
- **Cosine annealing:** smooth decay to near-zero; common for modern training.  
- **Warmup:** ramp `η` up over initial steps to avoid instability as `v` builds.  
- **One-cycle:** increase to a peak LR then anneal; pair with momentum scheduling (decrease `μ` when LR rises, increase `μ` when LR falls).

---

## 6) Weight Decay and Regularization

- **Decoupled weight decay:** after momentum update, apply `θ ← θ - η λ θ`.  
- **Dropout, data augmentation, label smoothing:** complementary to momentum; aim at generalization.  
- **BatchNorm/LayerNorm:** stabilize optimization and allow larger `η` with momentum.

---

## 7) Convergence Properties (Concise)

- **Convex/quadratic settings:** momentum can approach the optimal damping for eigenvalue spectra, improving effective condition numbers and convergence speed.  
- **Non-convex deep nets:** momentum reduces gradient noise variance in dominant directions and helps escape saddle points via accumulated velocity.  
- **Stability:** higher `μ` increases memory but can overshoot if paired with too large `η`; schedules balance these effects.

---

## 8) Choosing Between Classical Momentum and Nesterov

- **Classical momentum:** simple, strong baseline.  
- **Nesterov (NAG):** look-ahead reduces overshoot, often slightly better stability and final accuracy on some tasks.  
- Practical note: differences are modest; try both if chasing last points of accuracy.

---

## 9) Relations to Adaptive Methods

- **Adam/AdamW:** add per-parameter adaptivity via second-moment estimates; faster initial progress, robust defaults. Momentum is built-in via `β₁`.  
- **RMSProp:** adaptivity based on squared gradients; can be combined with momentum.  
- **SGD+Momentum vs. AdamW:** SGD+Mom often yields superior **final generalization** in vision; AdamW reaches target loss faster and is default for Transformers.

---

## 10) Diagnostics and Troubleshooting

Symptoms → Actions:

- **Divergence or loss spikes**  
  - Reduce `η`; consider warmup; optionally reduce `μ` or add gradient clipping.
- **Zig-zagging / high variance**  
  - Increase `μ` modestly (e.g., `0.9 → 0.95`); reduce `η`; consider larger batch or stronger schedule.
- **Plateaued training**  
  - Slightly increase `η`; adjust schedule; try Nesterov; verify normalization and initialization.
- **Overfitting**  
  - Increase weight decay, use stronger augmentation or label smoothing; decay LR earlier.

---

## 11) Practical Recipes

- **Vision (ResNet-like)**  
  - `η = 0.1` (scale with batch), `μ = 0.9`, weight decay `1e-4`, cosine decay with 5–10 epoch warmup, standard augmentation; consider MixUp/CutMix for robustness.
- **Tabular/MLP**  
  - `η = 1e-2` to `1e-1`, `μ = 0.9`; step decay; early stopping on validation.
- **RNNs/sequence**  
  - Use smaller `η` (`1e-3` to `1e-2`), gradient clipping (e.g., global norm 1.0), optionally Nesterov.

---

## 12) Minimal Pseudocode (Classical and Nesterov)

### 12.1 Classical Momentum
'''python
# θ: parameters; g(θ): gradient; ⊙: elementwise
initialize θ, v = 0
set η (lr), μ in [0.9, 0.99]

while not converged:
    g = gradient(θ)
    v = μ * v + g
    θ = θ - η * v
'''

### 12.2 Nesterov Momentum (look-ahead)
'''python
# Option A: explicit look-ahead gradient
initialize θ, v = 0
set η, μ

while not converged:
    g_look = gradient(θ - η * μ * v)
    v = μ * v + g_look
    θ = θ - η * v
'''

### 12.3 Nesterov Momentum (framework-style)
'''python
# Option B: update using current gradient and correction term
initialize θ, v = 0
set η, μ

while not converged:
    g = gradient(θ)
    v = μ * v + g
    θ = θ - η * (μ * v + g)
'''

---

## 13) Worked Examples

### 13.1 Toy Quadratic (scalar)
'''python
import numpy as np

def grad(theta):  # d/dθ (θ-3)^2
    return 2 * (theta - 3)

theta = 0.0
eta = 0.2
mu = 0.9
v = 0.0

for t in range(1, 21):
    g = grad(theta)
    v = mu * v + g
    theta -= eta * v
    loss = (theta - 3)**2
    print(f"step {t:02d}  theta={theta:.6f}  loss={loss:.6f}  v={v:.6f}")
'''

### 13.2 Logistic Regression (mini-batch, Nesterov)
'''python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

# synthetic dataset
np.random.seed(0)
N, D = 2000, 10
X = np.random.randn(N, D)
true_w = np.random.randn(D)
logits = X @ true_w + 0.25 * np.random.randn(N)
y = (sigmoid(logits) > 0.5).astype(np.float64)

w = np.zeros(D)
v = np.zeros(D)
eta = 5e-2
mu = 0.9
batch = 64
epochs = 10

for epoch in range(epochs):
    idx = np.random.permutation(N)
    Xp, yp = X[idx], y[idx]
    for s in range(0, N, batch):
        xb = Xp[s:s+batch]
        yb = yp[s:s+batch]
        # Nesterov: look-ahead gradient
        w_look = w - eta * mu * v
        p = sigmoid(xb @ w_look)
        g = xb.T @ (p - yb) / len(yb)
        v = mu * v + g
        w = w - eta * v

    p_all = sigmoid(X @ w)
    loss = -(y*np.log(p_all + 1e-12) + (1-y)*np.log(1-p_all + 1e-12)).mean()
    print(f"epoch {epoch+1}  loss={loss:.5f}")
'''

---

## 14) Plotly Illustration (example)

"""""js_plotly
{
  "data": [
    {
      "x": ["SGD", "SGD+Momentum", "Nesterov"],
      "y": [18, 10, 9],
      "type": "bar"
    }
  ],
  "layout": { "title": "Illustrative Steps to Target Loss (lower is better)" }
}
"""""

---

## 15) Implementation Details (Gotchas)

- **Shuffling each epoch** to avoid bias in momentum accumulation.  
- **Gradient clipping** for RNNs/very deep nets to prevent runaway velocities.  
- **Mixed precision**: monitor for overflow; scale loss as needed.  
- **Initialization**: good initial scales (He/Kaiming for ReLU) allow larger `η` with momentum.  
- **Weight-decay exclusions** (bias and normalization parameters) if applying decay.

---

## 16) Comparison Summary

- **SGD vs. SGD+Momentum:** momentum reduces zig-zag and speeds convergence with minimal overhead.  
- **Nesterov vs. Classical:** modest but reliable stability gain via look-ahead.  
- **AdamW vs. SGD+Momentum:** AdamW for quick progress and Transformers; SGD+Momentum for top-end generalization in many vision tasks.

---

## 17) Checklist Before Training

- Choose optimizer variant: classical momentum or Nesterov.  
- Set `η` and `μ` (start with `η=0.1`, `μ=0.9` for normalized vision tasks).  
- Select a schedule (cosine with warmup or step decay).  
- Configure weight decay (decoupled) and exclusions.  
- Enable gradient clipping if needed.  
- Verify normalization layers and data preprocessing.  
- Establish logging, checkpoints, and validation metrics.

---

## 18) Summary

Momentum augments SGD with a velocity term that accumulates gradient information, accelerating along consistent directions and damping oscillations in ill-conditioned landscapes. Classical momentum is a strong baseline; Nesterov’s look-ahead often improves stability further. With a sensible learning-rate schedule, normalization, and regularization, momentum-based SGD remains a competitive and frequently state-of-the-art choice for training deep networks, especially in vision.

---
