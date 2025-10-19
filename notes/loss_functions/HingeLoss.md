# Hinge Loss

**Hinge loss** is a margin-based loss used primarily with **maximum-margin classifiers** like **Support Vector Machines (SVMs)**.  
It penalizes predictions that are incorrect **or** correct but **within the margin**, encouraging a large decision margin and robust separation.

---

## 1) Setup, Notation, and Intuition

- Binary labels encoded as `y ∈ {−1, +1}`.  
- A linear scorer `f(x) = w⋅x + b` (or any real-valued logit).  
- **Margin** on example `(x, y)` is `m = y · f(x)`.  
  - If `m ≥ 1`, the example lies **outside** the margin (good).  
  - If `m < 1`, it is **inside** the margin or misclassified (penalize).

**Core idea:** Encourage `m ≥ 1` by penalizing `1 − m` when positive.

---

## 2) Binary Hinge Variants

### 2.1 Standard Hinge (L1-hinge)
`L(x,y) = max(0, 1 − y·f(x))`

### 2.2 Squared Hinge (L2-hinge)
`L(x,y) = max(0, 1 − y·f(x))²`  
- Stronger penalty as points approach/violate the margin; differentiable at `m<1` but still non-smooth at `m=1`.

### 2.3 Smooth Hinge (Huberized hinge)
- Replace the kink near margin with a **Huber**-style quadratic to make it fully differentiable.  
- Useful for gradient-based solvers that prefer smooth objectives.

---

## 3) Multiclass Hinge

### 3.1 Crammer–Singer Multiclass Hinge
- Scores `f_k(x)` for classes `k=1..K`, true class `y`.  
- Loss:  
  `L = max_{k≠y} [ 1 + f_k(x) − f_y(x) ]_+`  
  where `[·]_+ = max(0, ·)`.  
- Encourages the true class score to exceed every other class by at least **1**.

### 3.2 One-vs-Rest (OvR)
- Train K binary classifiers with hinge loss: class `k` vs. rest.  
- At test time, pick `argmax_k f_k(x)`.

---

## 4) SVM Connection (Soft-Margin)

### 4.1 Primal Objective (Linear SVM, L2-regularized L1-hinge)
Minimize over `w, b`:
`(λ/2)‖w‖² + (1/N) Σ_i max(0, 1 − y_i (w⋅x_i + b))`

- `λ` is the regularization strength (inverse of `C` in classic SVM: `λ = 1/(C·N)`).

### 4.2 Interpretation
- Regularization **maximizes margin** (controls ‖w‖).  
- Hinge loss **enforces** margin constraints softly via slack.

---

## 5) Subgradients and Derivatives

### 5.1 Binary Standard Hinge
Let `m_i = y_i f(x_i)` and `I_i = 1[m_i < 1]`.
- Subgradient w.r.t. `w` (linear model):  
  `∂L/∂w = − (1/N) Σ_i I_i · y_i x_i`  
- No contribution from points with `m_i ≥ 1` (outside margin).

### 5.2 Squared Hinge
- Gradient where `m_i < 1`:  
  `∂L/∂w = − (2/N) Σ_i (1 − m_i) · y_i x_i`  
- Zero for `m_i ≥ 1`.

### 5.3 Multiclass (Crammer–Singer)
Let `k* = argmax_{k≠y} [1 + f_k − f_y]`. If the margin term ≤ 0 → gradient 0.  
Otherwise:
- `∂L/∂f_y = −1`, `∂L/∂f_{k*} = +1`, others 0 (per violating pair).

---

## 6) Practical Usage

- **Label encoding:** use `y ∈ {−1, +1}` for hinge/BCE-style binary margins. If labels are `{0,1}`, convert: `y' = 2y − 1`.  
- **Linear vs. deep models:** Hinge is classic for linear/kernels; with deep nets, CE+softmax is more common, but hinge variants remain useful for margin emphasis and some ranking tasks.  
- **Regularization:** pair hinge with **L2 weight decay**; `λ` (or `C`) controls margin–slack trade-off.  
- **Optimization:** hinge is non-smooth at the margin; SGD with subgradients works; squared/smooth hinge can stabilize.  
- **Calibration:** hinge-trained scores are not calibrated probabilities; use Platt scaling or temperature scaling if probabilities are needed.

---

## 7) Ranking and Metric Learning Relations

### 7.1 Pairwise Ranking Hinge
For pairs `(x⁺, x⁻)` with scores `s⁺, s⁻`:  
`L = max(0, 1 − (s⁺ − s⁻))`  
Enforces `s⁺ ≥ s⁻ + 1`. Used in learning-to-rank.

### 7.2 Triplet Loss (hinge-like)
`L = max(0, d(a,p) − d(a,n) + margin)`  
Hinge structure over distances; widely used in metric learning.

---

## 8) PyTorch Implementations

> PyTorch doesn’t provide a single “binary hinge loss” for logits out of the box; it provides **`MarginRankingLoss`** (pairwise), **`HingeEmbeddingLoss`** (rarely used for standard classification), and **`multi_margin_loss`** (Crammer–Singer). For binary hinge, implement directly.

### 8.1 Binary Hinge (manual)
'''python
import torch
import torch.nn.functional as F

def binary_hinge_loss(logits, targets, squared=False):
    """
    logits: shape [N], raw scores f(x)
    targets: shape [N] with values in {-1, +1}
    """
    margins = 1 - targets * logits
    losses = torch.clamp(margins, min=0.0)
    if squared:
        losses = losses ** 2
    return losses.mean()

# example
N = 8
logits = torch.randn(N)
y = torch.randint(0, 2, (N,))*2 - 1  # {0,1} -> {-1,+1}
loss = binary_hinge_loss(logits, y, squared=False)
print(loss.item())
'''

### 8.2 Multiclass (Crammer–Singer) with `multi_margin_loss`
'''python
import torch
import torch.nn as nn

N, K = 16, 5
logits = torch.randn(N, K)           # raw scores
targets = torch.randint(0, K, (N,))  # class indices [0..K-1]

# p=1 uses standard hinge; p=2 uses squared hinge; margin=1.0 default
crit = nn.MultiMarginLoss(p=1, margin=1.0)  # Crammer–Singer
loss = crit(logits, targets)
print(loss.item())
'''

### 8.3 Pairwise Ranking with `MarginRankingLoss`
'''python
import torch
import torch.nn as nn

N = 10
s_pos = torch.randn(N)  # scores for positive items
s_neg = torch.randn(N)  # scores for negative items
y = torch.ones(N)       # target: +1 enforces s_pos > s_neg

crit = nn.MarginRankingLoss(margin=1.0)
loss = crit(s_pos, s_neg, y)
print(loss.item())
'''

---

## 9) Decision Functions and Thresholds

- Binary prediction: `ŷ = sign(f(x))`.  
- Confidence: margin `m = y·f(x)`; larger is better.  
- If calibrated probabilities are required, post-hoc **Platt scaling** (logistic regression on validation logits) can map `f(x)` to `P(y=1|x)`.

---

## 10) When to Prefer Hinge vs. Cross-Entropy

- **Hinge/SVM**  
  - Pros: strong margin, robustness to a few outliers, good with kernel methods, works well with sparse high-dimensional text features.  
  - Cons: non-probabilistic, non-smooth; less common in end-to-end deep learning.

- **Cross-Entropy**  
  - Pros: probabilistic, smooth, integrates naturally with softmax/sigmoid; dominant for deep nets.  
  - Cons: may push for overconfident solutions without explicit margin unless combined with techniques (label smoothing, large-margin softmax variants).

---

## 11) Large-Margin Softmax (Related Idea)

- Modern deep learning sometimes adds **margin** inside the softmax (e.g., **ArcFace**, **CosFace**, **AM-Softmax**) to emulate SVM-like margins while keeping CE training and probability outputs.

---

## 12) Common Pitfalls

- Using labels `{0,1}` directly in hinge formulas—**convert** to `{−1,+1}`.  
- Expecting calibrated probabilities—hinge gives **margins**, not probabilities.  
- Forgetting regularization; hinge without proper `λ` can overfit.  
- For multiclass, mixing OvR and Crammer–Singer formulations inadvertently in code.

---

## 13) Plotly Illustration (example)

"""""js_plotly
{
  "data": [
    { "x": ["Correct, outside margin", "Correct, inside margin", "Misclassified"],
      "y": [0, 0.4, 1.6],
      "type": "bar",
      "name": "Hinge Loss (example)" }
  ],
  "layout": { "title": "Hinge Loss vs. Margin Violations (illustrative)" }
}
"""""

---

## 14) Minimal Linear SVM (SGD) Sketch

'''python
import torch

def sgd_linear_svm_step(w, b, X, y, lr=1e-2, lam=1e-4, squared=False):
    """
    One SGD step for linear SVM on a mini-batch.
    w: [D], b: scalar; X: [N,D], y in {-1,+1} [N]
    """
    # compute margins and masks
    scores = X @ w + b                      # [N]
    margins = 1 - y * scores                # [N]
    viol = margins > 0                      # mask of violators

    # sub/gradients from hinge
    if squared:
        # d/dw of (max(0, m))^2 = 2*max(0,m)*d m
        coeff = -2 * margins[viol] * y[viol]
    else:
        coeff = -y[viol]

    # gradients
    if viol.any():
        grad_w = X[viol].T @ coeff / X.size(0)
        grad_b = coeff.sum() / X.size(0)
    else:
        grad_w = torch.zeros_like(w)
        grad_b = torch.tensor(0., dtype=w.dtype)

    # L2 regularization gradient
    grad_w += lam * w

    # update
    w -= lr * grad_w
    b -= lr * grad_b
    return w, b

# usage (toy)
D = 20
w = torch.zeros(D)
b = torch.tensor(0.0)
X = torch.randn(64, D)
y = (torch.randint(0,2,(64,))*2 - 1).float()
w, b = sgd_linear_svm_step(w, b, X, y, lr=1e-2, lam=1e-4, squared=False)
'''

---

## 15) Checklist

- Encode labels as `{−1,+1}` for hinge-based binary classification.  
- Choose variant: standard, squared, or smooth hinge.  
- Add **L2 regularization** (`λ` or `weight_decay`).  
- For multiclass, pick **Crammer–Singer** or OvR consistently.  
- If you need probabilities, add **calibration** post-training.  
- Monitor **margin violations** and norm ‖w‖ to track training dynamics.

---

## 16) Summary

Hinge loss focuses on **margins**, penalizing points that are misclassified or lie within a safety band around the decision boundary.  
Coupled with L2 regularization, it yields **maximum-margin** classifiers (SVMs) with strong generalization.  
While cross-entropy dominates end-to-end deep learning, hinge-based objectives remain central in linear/kernalized SVMs, ranking, metric learning, and margin-augmented variants of modern classifiers.

---
