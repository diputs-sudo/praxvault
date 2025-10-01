# Stochastic Gradient Descent (SGD)

**Stochastic Gradient Descent (SGD)** is the core optimization method used to train models by minimizing an objective (loss) function.  
It replaces the full-dataset gradient with a noisy, low-cost estimate computed on a random sample (a mini-batch).  
This simple idea scales to massive datasets and deep networks, and—when combined with the right schedules and tweaks—remains competitive with or better than adaptive methods in terms of generalization.

---

## 1) Problem Setup and Derivation 

We consider Empirical Risk Minimization (ERM):

- **Objective**  
  `L(θ) = (1/N) Σ_{i=1..N} ℓ(θ; x_i, y_i)` where `θ` are parameters, `ℓ` is per-example loss.

- **Full Gradient Descent** (expensive when `N` is large)  
  `θ_{t+1} = θ_t - η ∇L(θ_t)`

- **Mini-batch SGD** (cheap stochastic estimate)  
  - Sample a mini-batch `B_t` of size `m`.  
  - Compute `g_t = (1/m) Σ_{i∈B_t} ∇ℓ(θ_t; x_i, y_i)`.  
  - Update `θ_{t+1} = θ_t - η g_t`.

**Key property**: `E[g_t | θ_t] = ∇L(θ_t)` (unbiased gradient estimate).

**Variance vs. cost trade-off**: smaller `m` ⇒ cheaper updates but noisier gradients; larger `m` ⇒ smoother but costlier.

---

## 2) Intuition, Geometry, and Stochasticity 

- **Gradient points uphill in loss; negative gradient points downhill.**
- **Noise is not a bug**: mini-batch noise shakes the parameters out of narrow, sharp valleys and can lead to flatter minima that generalize better.
- **Early in training** we want bold moves (higher learning rate). **Later** we refine (lower learning rate).

---

## 3) Algorithm Templates 

### 3.1 Vanilla Mini-batch SGD (no momentum)
- Input: `η` (learning rate), batch size `m`
- Loop over steps:
  - Sample mini-batch `B_t` of size `m`
  - Compute `g_t = (1/m) Σ_{i∈B_t} ∇ℓ(θ; x_i, y_i)`
  - Update `θ ← θ - η g_t`

### 3.2 SGD with Momentum (classical)
- Hyperparameter: momentum `μ ∈ [0, 1)` (e.g., `0.9`)
- Keep a velocity `v` (initialized to `0`)
- Each step:
  - `g_t = (1/m) Σ ∇ℓ(θ; x_i, y_i)`
  - `v ← μ v + g_t`
  - `θ ← θ - η v`

### 3.3 Nesterov Momentum (NAG)
- Look-ahead step improves stability:
  - `v ← μ v + ∇ℓ(θ - η μ v)`
  - `θ ← θ - η v`

### 3.4 Weight Decay (L2 regularization)
- Classical L2: add `λ ||θ||²` to loss ⇒ gradient adds `2λθ`.
- Decoupled weight decay (preferred):  
  - First do plain gradient step; then `θ ← θ - η λ θ`.  
  - Decoupling keeps optimization geometry cleaner.

---

## 4) Convergence and Theory 

- **Convex objectives**:  
  With diminishing step sizes (e.g., `η_t = c / √t`), SGD converges to the global optimum in expectation. Rates like `O(1/√t)` are typical.

- **Strongly convex + smooth**:  
  Faster convergence is possible with tailored schedules.

- **Non-convex (deep nets)**:  
  SGD converges to a stationary point (where gradient norm is small). Noise helps escape saddle points and narrow minima.

- **Unbiasedness and variance**:  
  - `E[g_t] = ∇L(θ_t)`.  
  - Var decreases with batch size `m`.  
  - Step-size must balance noise and progress.

---

## 5) Learning Rate: The Central Hyperparameter 

- Too large ⇒ divergence; too small ⇒ slow training.
- Typical starting points (normalized inputs):
  - Vision CNNs with momentum: `η ∈ [0.01, 0.2]`.  
  - MLPs/Tabular: `η ∈ [0.001, 0.1]`.
- Always combine with a **schedule** (see next section).

---

## 6) Learning Rate Schedules 

- **Step decay**: multiply `η` by `γ<1` at preset epochs (e.g., every 30 epochs).
- **Exponential decay**: `η_t = η_0 · γ^t`.
- **Cosine annealing**: smoothly decay to near-zero; optionally with **warm restarts**.
- **Polynomial decay**: `η_t = η_0 (1 - t/T)^p`.
- **One-cycle**: increase LR from low to high, then sharply decrease; pairs well with momentum schedules.
- **Warmup**: start small and ramp `η` up over a few hundred or thousand steps to stabilize early training (important with large batch sizes or deep nets).

**Practical rule**: use a brief LR range test to find the largest stable LR; then choose a schedule around that.

---

## 7) Batch Size Effects and Scaling 

- **Small batch** (e.g., 32–128): more noise, often better generalization.
- **Large batch** (e.g., 4k–64k across many GPUs): smoother but can harm generalization without care.
- **Linear scaling rule** (heuristic): when increasing batch size by factor `k`, scale `η ← k η` with warmup.
- **LARS/LAMB**: layer-wise adaptive scaling for very large batches to recover stability.
- **Gradient accumulation**: simulate large effective batch on limited memory by summing gradients across `K` micro-batches before an update.

---

## 8) Momentum and Nesterov: When and Why 

- **Momentum** dampens oscillations in high-curvature directions and accelerates progress in low-curvature valleys.
- Common `μ` values: `0.9` to `0.99`. Higher can be faster but riskier.
- **Nesterov** often slightly outperforms classical momentum in stability.

---

## 9) Regularization with SGD 

- **Weight decay** (decoupled) is standard: `λ ∈ [1e-5, 1e-3]` typical.
- **Data augmentation** acts like stochastic regularization (vision, audio).
- **Dropout** moderates co-adaptation in fully connected layers.
- **Label smoothing** can improve calibration for classification.
- **Early stopping** based on validation metrics to prevent overfitting.

---

## 10) Generalization: Flat vs. Sharp Minima 

- SGD’s gradient noise biases solutions toward **flatter minima**, which are robust to parameter perturbations and often generalize better.
- Very large batches reduce gradient noise and may converge to **sharper** minima unless countermeasures (data aug, weight decay, label smoothing, longer schedules) are used.

---

## 11) Stochastic Differential Equation (SDE) View 

- In the small step-size limit, SGD dynamics approximate a stochastic process:  
  `dθ = -∇L(θ) dt + √(2 T(θ)) dW_t`.  
- The **noise scale** relates to `η`, batch size, and gradient variance.  
- Tuning `η` and batch size implicitly tunes this noise, affecting exploration and the kind of minima reached.

---

## 12) Diagnostics and Troubleshooting 

- **Diverging loss or NaNs**  
  - Lower `η`; check data normalization; clip gradients if needed; verify loss implementation.
- **Plateaued training loss**  
  - Increase `η` modestly; add momentum; improve schedule; consider better initialization or normalization.
- **Training improves, validation worsens**  
  - Overfitting: increase regularization, data augmentation, or weight decay; consider early stopping.
- **Loss spikes**  
  - Too high LR or batch too small; try warmup, gradient clipping, or increase batch size.

---

## 13) Practical Recipes 

- **Vision (ResNet-like) with SGD + Momentum**  
  - Batch: 128–256 per GPU (or effective via accumulation)  
  - LR: start `0.1` (adjust for batch), momentum `0.9`, weight decay `1e-4`  
  - Schedule: cosine with warmup 5–10 epochs  
  - Augmentations: standard (rand crop/flip), possibly MixUp/CutMix

- **Tabular/MLP**  
  - LR: `1e-2` to `1e-1` with momentum `0.9`  
  - Step or cosine decay; early stopping on validation AUC/logloss

- **RNNs/Seq models (if using SGD)**  
  - Smaller LR (`1e-3` to `1e-2`), gradient clipping, momentum optional  
  - Consider layer norm or batch norm alternatives where appropriate

- **When to prefer Adam/AdamW**  
  - Very sparse gradients, complex adaptive behavior needed, rapid prototyping.  
  - Hybrid strategy: start with AdamW for quick progress, switch to SGD + momentum for final generalization.

---

## 14) Comparisons to Adaptive Methods 

- **SGD + momentum**  
  - Fewer hyperparameters; strong generalization on vision tasks; predictable behavior.
- **Adam/AdamW**  
  - Faster initial progress; adaptive per-parameter scaling; sometimes worse generalization without careful tuning.
- **RMSProp**  
  - Historically popular in RNNs; less common now versus AdamW.

**Rule of thumb**: If top-1 accuracy or test error matters most (e.g., vision), try SGD+momentum with a good schedule. If time-to-first-results matters or gradients are sparse, start with AdamW.

---

## 15) Implementation Details 

- **Shuffling** each epoch is essential.  
- **Data normalization/standardization** stabilizes gradients.  
- **Mixed precision** (float16/bfloat16) requires loss scaling; monitor for overflow.  
- **Initialization** matters (Kaiming/He for ReLU nets).  
- **Seed control** for reproducibility, acknowledging nondeterminism from parallelism and libraries.

---

## 16) Worked Examples

### 16.1 Toy Quadratic (scalar)
'''python
import numpy as np

theta = 0.0
eta = 0.1
for step in range(20):
    grad = 2 * (theta - 3)      # d/dθ (θ-3)^2
    theta -= eta * grad
    loss = (theta - 3)**2
    print(f"Step {step+1:02d}  θ={theta:.6f}  loss={loss:.6f}")
'''

### 16.2 Logistic Regression (mini-batch SGD)
'''python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

# synthetic data
np.random.seed(0)
N, D = 1000, 5
X = np.random.randn(N, D)
true_w = np.random.randn(D)
y = (sigmoid(X @ true_w + 0.25*np.random.randn(N)) > 0.5).astype(np.float64)

# parameters
w = np.zeros(D)
eta = 0.1
batch = 64
epochs = 10

for epoch in range(epochs):
    idx = np.random.permutation(N)
    Xp, yp = X[idx], y[idx]
    for s in range(0, N, batch):
        xb = Xp[s:s+batch]
        yb = yp[s:s+batch]
        p = sigmoid(xb @ w)
        grad = xb.T @ (p - yb) / len(yb)  # gradient of log-loss
        w -= eta * grad
    # simple training loss
    p_all = sigmoid(X @ w)
    loss = -(y*np.log(p_all + 1e-12) + (1-y)*np.log(1-p_all + 1e-12)).mean()
    print(f"epoch {epoch+1}  loss={loss:.4f}")
'''

---

## 17) Plotly Illustration 

"""""js_plotly
{
  "data": [{
    "x": ["Batch GD", "Mini-batch SGD", "Pure SGD"],
    "y": [5, 20, 12],
    "type": "bar"
  }],
  "layout": { "title": "Relative Update Frequency vs. Stability (illustrative)" }
}
"""""

---

## 18) Advanced Topics in Brief

- **Variance reduction**:  
  - SVRG, SAGA reduce gradient variance and can accelerate convex optimization.  
  - Less common in large deep nets due to memory/compute trade-offs.

- **Preconditioning with SGD**:  
  - Diagonal preconditioners or Shampoo-like methods bridge SGD and second-order ideas; engineering complexity rises.

- **Second-order methods**:  
  - Newton/Quasi-Newton rarely scale to modern deep nets; approximations exist but are intricate.

- **Sharpness-aware minimization (SAM)**:  
  - Perturb parameters to penalize sharp minima; often layered atop SGD/AdamW.

- **Label noise and robustness**:  
  - Small batches and regularization help; consider loss corrections for heavy noise.

---

## 19) Frequently Asked Questions

- **How do I pick the initial learning rate?**  
  Run a short LR range test: increase LR exponentially until loss blows up; pick 1/3 to 1/10 of the blow-up threshold, then add a decay schedule.

- **Is momentum always helpful?**  
  Usually yes for deep nets. Try `0.9` first.

- **Why does large batch hurt generalization?**  
  It reduces gradient noise; remedies include stronger regularization, longer training, and schedules like cosine with warmup.

- **When should I switch from AdamW to SGD?**  
  After rapid initial fitting, switch to SGD+momentum for final fine-tuning to improve generalization on many vision tasks.

---

## 20) Minimal PyTorch-Style Pseudocode

'''python
# assume model, loss_fn, dataloader
theta = [p for p in model.parameters()]
v = [np.zeros_like(p) for p in theta]  # velocity for momentum
eta, mu = 0.1, 0.9

for epoch in range(num_epochs):
    shuffle(dataloader)
    for x, y in dataloader:
        g = compute_gradients(model, loss_fn, x, y, theta)
        for i in range(len(theta)):
            v[i] = mu * v[i] + g[i]           # momentum
            theta[i] = theta[i] - eta * v[i]  # SGD step
    eta = schedule(epoch, eta)
'''

---

## 21) Checklist Before You Train

- Data normalized and shuffled every epoch  
- Reasonable initial LR and schedule chosen  
- Momentum enabled (0.9–0.99)  
- Weight decay set (decoupled)  
- Batch size feasible for memory; consider accumulation  
- Mixed precision configured with loss scaling  
- Validation metric monitored with early stopping criteria

---

## 22) Summary

SGD is simple but not simplistic.  
Its stochastic gradients, when paired with momentum, robust schedules, and basic regularization, provide a powerful, scalable, and generalization-friendly optimization path for deep learning and beyond.  
Adaptive methods are excellent tools, but SGD remains the baseline that strong systems return to when the goal is performance that holds up out of sample.

---
