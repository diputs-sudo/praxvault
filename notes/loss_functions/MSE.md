# Mean Squared Error (MSE)

**Mean Squared Error (MSE)** is the canonical regression loss: the average of squared differences between predictions and targets.  
It is the **L2 loss** and corresponds to **maximum likelihood** under Gaussian noise. MSE emphasizes large errors (quadratic growth), leading to fast, smooth optimization but **sensitivity to outliers**.

---

## 1) Definition and Notation

Given targets `y_i` and predictions `ŷ_i` for `i=1..N`:

- **Per-sample squared error:** `(ŷ_i − y_i)²`
- **MSE:** `MSE = (1/N) Σ_i (ŷ_i − y_i)²`
- **RMSE:** `RMSE = sqrt(MSE)` (back to target units; more interpretable)

Vector form (with `y, ŷ ∈ ℝ^N`):  
`MSE = (1/N) ‖ŷ − y‖²₂`

---

## 2) Statistical Interpretation

- **Noise model:** If `Y = f(X) + ε` with i.i.d. `ε ~ 𝓝(0, σ²)`, the negative log-likelihood reduces (up to constants) to `(1/(2σ²)) Σ (ŷ − y)²`.  
  Minimizing MSE ↔ maximizing likelihood under Gaussian noise.
- **Estimator target:** Minimizing expected MSE yields the **conditional mean**:  
  `argmin_a E[(Y − a)²] = E[Y]`.  
  In supervised learning, minimizing `E[(Y − f(X))² | X=x]` targets `E[Y|X=x]`.
- **Bias–variance decomposition (population):**  
  `E[(ŷ − y)²] = (Bias[ŷ])² + Var[ŷ] + σ²` (irreducible noise).

---

## 3) Geometry, Gradients, and Curvature

Let `e = ŷ − y`.

- **Gradient (scalar w.r.t. ŷ):** `∂(e²)/∂ŷ = 2e` → errors produce gradients proportional to their magnitude.  
- **Hessian (scalar):** `∂²(e²)/∂ŷ² = 2` → constant curvature; smooth, convex in ŷ.  
- **Implication:** smooth optimization with strong signals from large errors, but also **outlier sensitivity**.

---

## 4) Closed Form for Linear Regression (OLS)

For linear model `ŷ = Xw` with full-rank `X ∈ ℝ^{N×d}`:

- **Objective:** minimize `(1/N) ‖Xw − y‖²₂`.  
- **Normal equations:** `XᵀX w = Xᵀ y`.  
- **Solution:** `w* = (XᵀX)^{-1} Xᵀ y` (if invertible).  
- **Regularized variants:**  
  - Ridge (L2): `w* = (XᵀX + λI)^{-1} Xᵀ y` (stabilizes ill-conditioning).  
  - Lasso (L1): no closed form; promotes sparsity.

---

## 5) MSE vs. MAE vs. Huber

- **MSE (L2):** targets **mean**; quadratic penalty; efficient gradients; **less robust** to outliers.  
- **MAE (L1):** targets **median**; linear penalty; robust; non-differentiable at 0 (subgradient).  
- **Huber (Smooth L1):** quadratic near 0, linear for large |e|; controlled by `δ`; balances robustness and smoothness.

**Rule of thumb:**  
- Gaussian-like noise and desire for fast convergence → MSE / Huber (small δ).  
- Heavy tails/outliers → MAE / Huber (larger linear region).

---

## 6) Scaling, Units, and Transformations

- **Units:** MSE is in **squared units** of the target; RMSE restores original units.  
- **Feature/target scaling:** Standardizing inputs helps optimization; consider **target scaling** (e.g., z-score) and invert scaling when reporting.  
- **Log-transform:** For multiplicative noise / positive targets, training on `log(y)` and evaluating RMSE in log space may better reflect relative errors.

---

## 7) Reductions, Masking, and Weights

- **Reductions:** `mean` (MSE), `sum` (SSE), or `none` (per-element errors).  
- **Masking:** handle missing labels/padded sequences by multiplying with `mask ∈ {0,1}` and normalizing by valid count.  
- **Weights:** per-sample or per-dimension weights `w_i` to prioritize critical components (e.g., large boxes, important sensors).

---

## 8) Diagnostics and Metrics

- **Learning curves:** training vs. validation MSE/RMSE to detect under/overfitting.  
- **Residual analysis:** plot `e = ŷ−y` vs. `ŷ` or features; check heteroscedasticity and outliers.  
- **R² (coefficient of determination):**  
  `R² = 1 − SSE/SST = 1 − Σ(ŷ−y)² / Σ(y−ȳ)²` (for linear regression; careful in non-linear/biased settings).  
- **Calibration for uncertainty:** If modeling aleatoric noise, predict `σ²(x)` and optimize a Gaussian NLL instead of plain MSE.

---

## 9) PyTorch Usage

### 9.1 Basic MSE
'''python
import torch
import torch.nn as nn

y_true = torch.tensor([3.0, 0.0, -2.0])
y_pred = torch.tensor([2.5, 0.2, -3.0])

mse = torch.mean((y_pred - y_true) ** 2)
rmse = torch.sqrt(mse)
print(float(mse), float(rmse))
'''

### 9.2 Built-in Loss: `MSELoss`
'''python
import torch
import torch.nn as nn

crit = nn.MSELoss(reduction='mean')  # 'sum' or 'none'
loss = crit(y_pred, y_true)
print(float(loss))
'''

### 9.3 Masked/Weighted MSE
'''python
import torch

def masked_weighted_mse(pred, target, mask=None, weight=None):
    diff2 = (pred - target) ** 2
    if mask is not None:
        diff2 = diff2 * mask
    if weight is not None:
        diff2 = diff2 * weight
    denom = (mask.sum() if mask is not None else torch.numel(diff2)).clamp_min(1)
    return diff2.sum() / denom

pred = torch.tensor([1., 2., 3.])
tgt  = torch.tensor([1.5, 1.5, 10.])
mask = torch.tensor([1., 1., 0.])  # ignore last
print(float(masked_weighted_mse(pred, tgt, mask=mask)))
'''

### 9.4 Huber as Robust Alternative
'''python
import torch
import torch.nn as nn

crit_huber = nn.HuberLoss(delta=1.0, reduction='mean')
pred = torch.tensor([1., 2., 100.])
tgt  = torch.tensor([1., 2.,   3.])
print(float(crit_huber(pred, tgt)))   # less dominated by the outlier than MSE
'''

---

## 10) Computer Vision and Sequences

### 10.1 Image Regression (per-pixel MSE)
'''python
import torch

pred = torch.rand(8, 3, 64, 64)
tgt  = torch.rand(8, 3, 64, 64)
mse = torch.mean((pred - tgt) ** 2)

# Channel-weighted MSE
weights = torch.tensor([0.5, 1.0, 1.5]).view(1,3,1,1)
mse_w = torch.mean(((pred - tgt) ** 2) * weights)
'''

### 10.2 Sequence Forecasting (mask variable lengths)
'''python
import torch

# pred, tgt: [B, T, D], mask: [B, T] 1=valid, 0=pad
def seq_mse(pred, tgt, mask):
    diff2 = ((pred - tgt) ** 2).sum(dim=-1)    # per time-step
    diff2 = diff2 * mask
    return diff2.sum() / mask.sum().clamp_min(1)

B, T, D = 4, 10, 3
pred = torch.randn(B, T, D)
tgt  = torch.randn(B, T, D)
mask = torch.ones(B, T); mask[:, -2:] = 0  # last two are padding
print(float(seq_mse(pred, tgt, mask)))
'''

---

## 11) Robustness and Extensions

- **Huber / Smooth L1:** mitigates outliers while keeping quadratic behavior near zero.  
- **ε-insensitive (SVR):** ignores small errors within tube ε; piecewise quadratic/linear penalties.  
- **Pinball (quantile) loss:** if you want conditional quantiles rather than the mean.  
- **Tukey’s biweight, Cauchy loss:** more aggressive robust alternatives (nonconvex; careful optimization).

---

## 12) Common Pitfalls

- **Unscaled targets:** very large scales → tiny gradients dominate; normalize/standardize targets.  
- **Outlier-dominated loss:** MSE can be hijacked by a few extreme points; consider Huber/MAE, outlier handling, or heavier-tailed likelihoods.  
- **Improper masking:** forgetting to renormalize by valid-count when masking.  
- **Comparing MSE across datasets:** different target scales make raw MSE/RMSE non-comparable; report standardized metrics or scale-aware baselines.  
- **Using MSE for classification logits:** prefer cross-entropy; MSE on probabilities/logits is poorly conditioned for classification.

---

## 13) Plotly Illustration (example)

"""""js_plotly
{
  "data": [
    { "x": ["MSE", "MAE", "Huber(δ=1)"], "y": [2.4, 1.3, 1.1], "type": "bar" }
  ],
  "layout": { "title": "Illustrative Sensitivity to Outliers (lower is better)" }
}
"""""

---

## 14) Minimal Training Loop Sketch (PyTorch)

'''python
import torch
import torch.nn as nn
import torch.optim as optim

model = ...  # define your regression model
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    # validation
    model.eval()
    with torch.no_grad():
        val_mse, n = 0.0, 0
        for x, y in val_loader:
            pred = model(x)
            val_mse += ((pred - y) ** 2).mean().item() * len(x)
            n += len(x)
        print(f"epoch {epoch+1}  val_MSE={val_mse/n:.4f}")
'''

---

## 15) Summary

MSE is the workhorse loss for regression: smooth, convex in predictions, and statistically grounded in Gaussian noise models.  
It targets the **conditional mean** and supplies strong gradient signals, but can be skewed by outliers.  
Good practice: scale features/targets, monitor residuals, consider Huber/MAE when robustness matters, and report **RMSE** for interpretability alongside complementary metrics.

---
