# Cross-Entropy Loss

**Cross-Entropy (CE)** measures the dissimilarity between a **true distribution** `p` and a **predicted distribution** `q`.  
In supervised learning, CE is the dominant objective for **classification**: binary, multiclass (single-label), and multi-label setups.  
It connects probability theory, information theory, and gradient-based optimization via softmax/sigmoid with numerically stable log-likelihoods.

---

## 1) Information-Theoretic Foundations

### 1.1 Entropy, Cross-Entropy, KL
- **Entropy** of `p`: `H(p) = - Σ_y p(y) log p(y)`  
- **Cross-Entropy**: `H(p, q) = - Σ_y p(y) log q(y)`  
- **Kullback–Leibler (KL) divergence**: `KL(p‖q) = H(p, q) - H(p)`  
  Minimizing CE is equivalent to minimizing KL from `p` to `q` (since `H(p)` is constant w.r.t. `q`).

### 1.2 One-Hot Labels
For a one-hot label `y*`, CE reduces to the **negative log-likelihood** (NLL) of the true class:  
`H(p, q) = - log q(y*)`.

---

## 2) Problem Settings and Canonical Forms

### 2.1 Binary Classification (single target `y ∈ {0,1}`)
- Model outputs logit `z` (real), probability `σ(z) = 1/(1+e^{-z})`.
- **Binary cross-entropy (BCE)** for one sample:  
  `L = - [ y log σ(z) + (1 - y) log (1 - σ(z)) ]`

### 2.2 Multiclass, Single-Label (`y ∈ {1..K}`)
- Model outputs logits `z ∈ ℝ^K`.  
- Softmax probabilities: `q_k = exp(z_k) / Σ_j exp(z_j)`.  
- CE for class `y`: `L = - log q_y = - z_y + log Σ_j exp(z_j)` (**log-sum-exp** form).

### 2.3 Multi-Label (independent binary targets for K classes)
- Targets `y ∈ {0,1}^K`; use **sigmoid per class** + **BCE per class**, averaged/summed.  
- Do **not** use softmax here.

---

## 3) Stable Computation from Logits (no explicit probabilities)

### 3.1 Binary (BCEWithLogits)
Use a stable formulation avoiding `σ(z)` explicitly:
- `L = max(z,0) - z*y + log(1 + exp(-|z|))`

### 3.2 Multiclass (CrossEntropy over logits)
- `L = - z_y + log Σ_j exp(z_j)`  
- Compute via `log_softmax(z)` to avoid overflow; most frameworks do this internally.

**Rule:** Prefer **`CrossEntropyLoss`** (expects **raw logits**) and **`BCEWithLogitsLoss`** (expects **raw logits**). Avoid manual `softmax`/`sigmoid` before loss to prevent double application and numerical issues.

---

## 4) Gradients (useful for intuition)

### 4.1 Multiclass (softmax + CE, logits `z`)
- Predicted prob `q = softmax(z)`.  
- For class `k`: `∂L/∂z_k = q_k - 1{k=y}`  
  This elegant form drives probability mass toward the true class.

### 4.2 Binary (sigmoid + BCE)
- `∂L/∂z = σ(z) - y`  
  Identical shape: prediction minus target.

---

## 5) Label Smoothing and Soft Targets

### 5.1 Label Smoothing (multiclass)
- Replace one-hot `y*` with smoothed `p̃`:  
  `p̃_y = 1 - ε`, and for others `p̃_k = ε/(K-1)`.  
- Loss: `L = - Σ_k p̃_k log q_k`  
- Effects: reduces overconfidence, improves calibration and robustness.

### 5.2 Soft Targets / Distillation
- Teacher provides a distribution `p^T`; student minimizes CE to `p^T`:  
  `L = - Σ_k p^T_k log q_k`.  
- Often combined with temperature scaling in softmax.

---

## 6) Class Imbalance and Weighting

### 6.1 Class Weights (multiclass)
- Weighted CE: `L = - w_y log q_y`; choose `w` inversely proportional to frequency or via focal-style heuristics.

### 6.2 Positive/Negative Balancing (binary/multi-label)
- `BCEWithLogitsLoss(pos_weight=...)` upweights positives for rare classes.  
- For highly imbalanced detection/segmentation, consider **Focal Loss**.

---

## 7) Focal Loss (relation to CE)

For binary (extendable to multiclass with one-vs-all):
- Let `p_t = σ(z)` if `y=1`, else `p_t = 1 - σ(z)`.  
- **Focal**: `FL = - α (1 - p_t)^γ log(p_t)` with focusing `γ ≥ 0`.  
  Downweights easy examples, emphasizes hard ones.

---

## 8) Calibration, Confidence, and Temperature

- CE encourages **log-likelihood maximization**; models can become **overconfident**.  
- **Temperature scaling**: rescale logits `z/T` at eval to improve calibration (does not change accuracy).  
- Track **ECE** (Expected Calibration Error) alongside accuracy.

---

## 9) Reduction, Masking, and Ignoring Classes

- **Reduction:** `mean`, `sum`, or `none` (per-sample/per-pixel).  
- **ignore_index (multiclass)**: exclude a label id (e.g., `-100`) from loss/grad (useful for padding).  
- **Masking:** multiply per-sample/per-pixel losses by a mask and renormalize.

---

## 10) PyTorch Examples

### 10.1 Binary: `BCEWithLogitsLoss` (single output)
'''python
import torch
import torch.nn as nn

logits = torch.tensor([0.3, -1.2, 2.0])   # shape [N]
targets = torch.tensor([1., 0., 1.])      # shape [N]
crit = nn.BCEWithLogitsLoss()
loss = crit(logits, targets)
print(loss.item())
'''

### 10.2 Binary with Class Imbalance (pos_weight)
'''python
import torch
import torch.nn as nn

# upweight positives (rare class)
pos_weight = torch.tensor([5.0])  # scalar or per-channel
crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

logits = torch.randn(8, 1)    # [N, 1]
targets = torch.randint(0, 2, (8, 1)).float()
loss = crit(logits, targets)
print(loss.item())
'''

### 10.3 Multiclass: `CrossEntropyLoss` (expects logits, not probabilities)
'''python
import torch
import torch.nn as nn

N, K = 4, 5
logits = torch.randn(N, K)           # raw scores
targets = torch.tensor([0, 3, 1, 4]) # class indices [0..K-1]
crit = nn.CrossEntropyLoss()
loss = crit(logits, targets)
print(loss.item())
'''

### 10.4 Multiclass with Label Smoothing
'''python
import torch
import torch.nn as nn

crit = nn.CrossEntropyLoss(label_smoothing=0.1)  # PyTorch >= 1.10
logits = torch.randn(16, 10)
targets = torch.randint(0, 10, (16,))
loss = crit(logits, targets)
print(loss.item())
'''

### 10.5 Per-Pixel Multiclass (segmentation) with ignore_index
'''python
import torch
import torch.nn as nn

N, C, H, W = 2, 4, 8, 8
logits = torch.randn(N, C, H, W)
targets = torch.randint(0, C, (N, H, W))
targets[0, :2, :2] = -100  # ignore small region
crit = nn.CrossEntropyLoss(ignore_index=-100)
loss = crit(logits, targets)
print(loss.item())
'''

### 10.6 Multi-Label BCE (K independent labels per sample)
'''python
import torch
import torch.nn as nn

N, K = 8, 6
logits = torch.randn(N, K)            # per-class logits
targets = torch.randint(0, 2, (N, K)).float()
crit = nn.BCEWithLogitsLoss()
loss = crit(logits, targets)
print(loss.item())
'''

---

## 11) From Probabilities vs. From Logits (Do’s and Don’ts)

- **Do** pass **logits** to `CrossEntropyLoss` / `BCEWithLogitsLoss`.  
- **Do not** apply `softmax`/`sigmoid` before these losses (double-squashing harms gradients and stability).  
- **If you must** implement manually, use **`logsumexp`** and the stable BCE form.

---

## 12) Practical Recipes

- Start with standard CE (multiclass) or BCE-with-logits (binary/multi-label).  
- Add **label smoothing** (`ε ≈ 0.05–0.1`) for better calibration.  
- Handle imbalance with **class weights** or **`pos_weight`**, or switch to **Focal Loss**.  
- Use **ignore_index** and masks for padded or unlabeled regions.  
- Monitor both **accuracy** and **calibration** (ECE); consider **temperature scaling** at inference.

---

## 13) Common Pitfalls

- Feeding **probabilities** instead of logits to CE/BCE-with-logits.  
- Using **softmax for multi-label** tasks (should be **sigmoid per class**).  
- Forgetting to adjust **class weights** for heavy imbalance.  
- Mixing **label smoothing** with certain regularizers without retuning learning rate/schedule.  
- Numerical under/overflow from manual `exp`/`log`; rely on framework primitives.

---

## 14) Plotly Illustration (example)

"""""js_plotly
{
  "data": [
    {
      "x": ["Hard One-Hot", "Label Smoothing (ε=0.1)"],
      "y": [0.65, 0.60],
      "type": "bar",
      "name": "NLL (lower is better)"
    }
  ],
  "layout": { "title": "Illustrative Cross-Entropy with/without Label Smoothing" }
}
"""""

---

## 15) Summary

Cross-Entropy is the standard objective for classification because it aligns maximum-likelihood estimation with stable gradient computation from logits.  
Binary tasks use **sigmoid + BCE**, multiclass uses **softmax + CE**, and multi-label uses **independent sigmoids + BCE**.  
Robust practice involves numerically stable logits-based losses, optional label smoothing, class/pos weighting for imbalance, and calibration checks.
---
