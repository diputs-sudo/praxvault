# Adam (Adaptive Moment Estimation)

**Adam** is a first-order optimization algorithm that adapts the learning rate for each parameter by maintaining **exponential moving averages** of both the gradients (first moment) and the squared gradients (second moment). It combines the ideas of **Momentum** and **RMSProp**, includes **bias correction** for early steps, and is widely used as a strong default in deep learning.

---

## 1) Overview and Motivation

Adam aims to provide:
- **Fast, stable training** through momentum-like smoothing of gradients.
- **Per-parameter adaptivity** through scaling by a running estimate of gradient variance.
- **Robust early steps** via bias correction so that moving averages are well-behaved from iteration one.

Use cases:
- Large, deep networks where curvature varies across layers.
- Problems with **sparse gradients** (e.g., embeddings, NLP).
- Rapid prototyping and hyperparameter robustness.

---

## 2) Algorithm and Notation

Let parameters be `θ`, learning rate `η`, gradient at step `t` be `g_t = ∇L(θ_{t-1})`, first and second moment accumulators `m_t`, `v_t`, with decay factors `β₁ ∈ [0,1)` and `β₂ ∈ [0,1)` and small `ε` for numerical stability.

### 2.1 First and Second Moments
- First moment (mean of gradients):  
  `m_t = β₁ m_{t-1} + (1 - β₁) g_t`
- Second moment (mean of squared gradients):  
  `v_t = β₂ v_{t-1} + (1 - β₂) (g_t ⊙ g_t)`

### 2.2 Bias Correction
- Correct the initialization bias introduced by zeros at `t=1`:  
  `m̂_t = m_t / (1 - β₁^t)`  
  `v̂_t = v_t / (1 - β₂^t)`

### 2.3 Parameter Update
- Per-coordinate adaptive step:  
  `θ_t = θ_{t-1} - η · m̂_t / (√(v̂_t) + ε)`

Initialization:
- `m_0 = 0`, `v_0 = 0`, `t = 0`.

---

## 3) Default Hyperparameters and Roles

- **Learning rate `η`**: commonly `0.001` as a starting point.
- **`β₁`** (1st moment decay): typically `0.9`; higher values add inertia.
- **`β₂`** (2nd moment decay): typically `0.999`; controls adaptivity smoothness.
- **`ε`**: `1e-8` (framework defaults vary); avoids division by zero and affects step scale when `v̂_t` is tiny.
- **Batch size**: algorithm-agnostic, but interacts with stability; Adam tolerates a wide range.

Guidance:
- Start with defaults; adjust `η` first.  
- If training is jittery, consider a smaller `η` or slightly larger `β₂`.  
- Extremely sparse gradients may benefit from `ε` tuning.

---

## 4) Learning-Rate Schedules

Adam benefits from schedules despite adaptivity.

- **Warmup**: linearly ramp `η` over the first `k` steps/epochs to stabilize early dynamics.  
- **Cosine annealing**: smooth decay of `η` to near zero; works well for long runs.  
- **Step decay**: multiply `η` by a factor (e.g., `0.1`) at milestones.  
- **One-cycle**: increase LR to a peak then anneal; pair with momentum scheduling.

Practical approach:
- Brief LR range test to bracket a safe maximum; schedule below that maximum.

---

## 5) Weight Decay and AdamW

Classical L2 regularization adds `λ||θ||²` to the loss, coupling decay with the adaptive denominator. **AdamW** decouples decay from the adaptive update:

- Adam step (on loss gradient only).  
- Then apply decay: `θ ← θ - η λ θ`.

Benefits of decoupling:
- Cleaner optimization geometry.  
- Empirically better generalization on many tasks.

Recommendation:
- Prefer **AdamW** with typical `λ` in `[1e-5, 5e-4]` depending on model/scale.

---

## 6) Behavior and Intuition

- **Momentum effect** (`β₁`): smooths noisy gradients, accelerates along persistent directions.  
- **Adaptive scaling** (`β₂`): smaller steps where gradients have high variance; larger steps where variance is small.  
- **Bias correction**: without it, early steps would be underestimated; with it, steps are calibrated from the start.  
- **Sensitivity**: Adam is less sensitive to raw feature scaling than SGD but still benefits from normalization.

---

## 7) Batch Size and Scaling

- Generally tolerant to a wide range of batch sizes.  
- Large batches may require **learning-rate warmup** and careful tuning of `β₂`.  
- For very large batches, consider **AdamW + warmup** and longer schedules.

---

## 8) Diagnostics and Troubleshooting

Symptoms and actions:

- **Divergence or NaNs**  
  - Lower `η`; verify loss scaling with mixed precision; check data preprocessing; consider gradient clipping.

- **Plateaued training**  
  - Increase `η` moderately; apply a schedule; try slightly lower `β₂` (e.g., `0.99`) to react faster to changes; verify model capacity.

- **Training improves, validation degrades**  
  - Increase regularization (weight decay), data augmentation, or early stopping; examine the schedule (too aggressive LR may overfit early features).

- **Instability at start**  
  - Add warmup; check initialization; ensure proper batch normalization or layer norm where applicable.

---

## 9) Comparisons

- **vs. SGD + Momentum**  
  - Adam: faster initial progress, robust defaults, helpful for sparse/heterogeneous gradients.  
  - SGD+momentum: often superior final **generalization** in vision; requires more LR tuning but may yield higher top-1.

- **vs. RMSProp**  
  - Adam uses both first and second moments with bias correction; typically more stable and widely preferred.

- **vs. AdaGrad**  
  - AdaGrad accumulates squared gradients without decay (can over-shrink LR). Adam’s `β₂` decay avoids monotonic shrinkage.

---

## 10) Variants and Extensions

- **AdamW**: decoupled weight decay; recommended default in many modern settings.  
- **AMSGrad**: uses a non-increasing `v̂_t` via a max operator to address theoretical convergence concerns.  
- **AdaBelief**: tracks the variance of the gradient prediction error rather than raw squared gradients; can behave closer to SGD in flat regions.  
- **Adamax**: infinity-norm variant, sometimes more stable when gradients have large spikes.

---

## 11) Theoretical Notes (Concise)

- For convex problems under standard assumptions, Adam variants with appropriate conditions (e.g., AMSGrad) admit convergence guarantees.  
- Bias correction is crucial for unbiased estimates of moments in expectation.  
- In deep, non-convex regimes, Adam converges to stationary points; adaptivity shapes the effective geometry of the loss.

---

## 12) Practical Recipes

- **General deep nets**  
  - Start: `η = 1e-3`, `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, AdamW with `λ = 1e-4`.  
  - Use **warmup (e.g., 500–3k steps)** plus cosine annealing or step decay.  
  - Enable gradient clipping for RNNs or very deep nets (e.g., clip global norm to `1.0`).

- **Transformers / NLP**  
  - AdamW, `β₁ = 0.9`, `β₂ = 0.98` is common; warmup steps proportional to dataset/sequence lengths.  
  - LayerNorm everywhere; careful LR scaling with sequence length and batch size.

- **Vision CNNs**  
  - AdamW works well for fast convergence; for best top-1, consider switching to SGD+momentum late in training.  
  - Strong data augmentation and weight decay remain important.

---

## 13) Worked Examples

### 13.1 Toy Quadratic (scalar)
'''python
import numpy as np

theta = 0.0
eta = 0.1
beta1, beta2 = 0.9, 0.999
eps = 1e-8
m, v = 0.0, 0.0

for t in range(1, 21):
    grad = 2 * (theta - 3)          # d/dθ (θ-3)^2
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    theta -= eta * m_hat / (np.sqrt(v_hat) + eps)
    loss = (theta - 3) ** 2
    print(f"step {t:02d}  theta={theta:.6f}  loss={loss:.6f}")
'''

### 13.2 Logistic Regression (mini-batch AdamW)
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

# parameters
w = np.zeros(D)
m = np.zeros(D)
v = np.zeros(D)
eta = 1e-3
beta1, beta2 = 0.9, 0.999
eps = 1e-8
weight_decay = 1e-4  # decoupled

batch = 64
epochs = 10

for epoch in range(epochs):
    idx = np.random.permutation(N)
    Xp, yp = X[idx], y[idx]
    for s in range(0, N, batch):
        xb = Xp[s:s+batch]
        yb = yp[s:s+batch]

        p = sigmoid(xb @ w)
        grad = xb.T @ (p - yb) / len(yb)  # gradient wrt logistic loss

        # Adam moments
        t = epoch * (N // batch) + (s // batch) + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Adam step
        w -= eta * m_hat / (np.sqrt(v_hat) + eps)
        # Decoupled weight decay (AdamW)
        w -= eta * weight_decay * w

    # epoch loss
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
      "x": ["SGD+Mom", "RMSProp", "AdamW"],
      "y": [12, 8, 6],
      "type": "bar"
    }
  ],
  "layout": { "title": "Illustrative Convergence Speed (lower is better)" }
}
"""""

---

## 15) Minimal Pseudocode (Adam / AdamW)

'''python
# θ: parameters; g(θ): gradient; schedule: LR schedule function
initialize θ, m = 0, v = 0, t = 0
set η0, β1=0.9, β2=0.999, ε=1e-8
set weight_decay λ (AdamW) or 0 (Adam)

while not converged:
    t ← t + 1
    g_t ← gradient(θ)
    m ← β1 * m + (1 - β1) * g_t
    v ← β2 * v + (1 - β2) * (g_t ⊙ g_t)
    m_hat ← m / (1 - β1**t)
    v_hat ← v / (1 - β2**t)
    η ← schedule(t, η0)
    θ ← θ - η * m_hat / (sqrt(v_hat) + ε)
    θ ← θ - η * λ * θ   # decoupled weight decay (AdamW)
'''

---

## 16) Checklist Before Training

- Data preprocessing and normalization verified.  
- Reasonable initial `η` with warmup configured.  
- Defaults: `β₁=0.9`, `β₂=0.999`, `ε=1e-8`; AdamW with `λ≈1e-4`.  
- Schedule chosen (cosine, step, or one-cycle).  
- Gradient clipping configured if needed.  
- Mixed precision with dynamic loss scaling where applicable.  
- Validation metric and early stopping criteria defined.

---

## 17) Summary

Adam combines momentum and adaptive per-parameter scaling with bias correction to deliver fast, stable optimization across diverse deep learning tasks. With schedules and decoupled weight decay (AdamW), it offers a robust default that trains quickly and performs well. For absolute top-end generalization in some domains, a switch to SGD + momentum later in training can still be advantageous, but Adam remains a cornerstone optimizer for modern practice.

---
