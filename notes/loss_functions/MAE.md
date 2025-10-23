# Mean Absolute Error (MAE)

**Mean Absolute Error (MAE)** measures the average absolute difference between predictions and targets.  
It corresponds to **L1 loss** in regression, is robust to outliers compared to MSE (L2), and is **median-consistent**: minimizing MAE estimates the conditional **median** of the target given the inputs.

---

## 1) Definition and Notation

Given targets `y_i` and predictions `ŷ_i` for `i=1..N`:

- **Per-sample absolute error:** `|ŷ_i - y_i|`
- **MAE (mean):** `MAE = (1/N) Σ_i |ŷ_i - y_i|`
- **Median absolute error:** `MedAE = median_i |ŷ_i - y_i|` (robust summary statistic; not the training loss but a useful metric)

**Vector/tensor form (PyTorch-style):** apply absolute value elementwise, then reduce (mean/sum/none).

---

## 2) Statistical View

- **Estimator target:** Minimizing expected MAE yields the **median**:  
  `argmin_a E[|Y - a|] = median(Y)`.  
  In supervised learning, minimizing `E[|Y - f(X)| | X=x]` yields the **conditional median** `median(Y|X=x)`.

- **Noise model:** MAE is the negative log-likelihood under a **Laplace** (double-exponential) noise model with scale `b`:  
  `p(y|μ) = (1/(2b)) exp(-|y - μ|/b)`.

- **Robustness:** Compared to MSE, MAE is **less sensitive** to heavy-tailed noise/outliers because errors grow linearly rather than quadratically.

---

## 3) Geometry and Gradients

- **Subgradient (scalar):**  
  For `e = ŷ - y`, `∂|e|/∂ŷ = sign(e)` for `e ≠ 0`, and any value in `[-1, 1]` at `e=0` (non-differentiable kink).
- **Optimization impact:** Updates are bounded (`±1`) per sample (before averaging), which makes training **stable** but potentially **slow** near the optimum compared to MSE.
- **Smoothing:** If needed, use **Huber** or **Smooth L1** to get differentiability near zero.

---

## 4) MAE vs. MSE vs. Huber (Smooth L1)

- **MSE (L2):** minimizes conditional **mean**, strongly penalizes outliers; gradients scale with error magnitude.  
- **MAE (L1):** minimizes conditional **median**, robust to outliers; constant-magnitude gradients (subgradients).  
- **Huber:** quadratic for small errors, linear for large; controlled by `δ`. Interpolates between MSE and MAE.

**Rule of thumb:**  
- Outliers/heavy tails present → prefer **MAE** or **Huber**.  
- Small Gaussian-like noise, wanting faster convergence → **MSE**.

---

## 5) Reductions, Masking, and Weighting

- **Reductions:** `mean` (default), `sum`, or `none` (per-element loss for custom aggregation).  
- **Masking:** multiply elementwise by a mask `m_i ∈ {0,1}` and normalize by `Σ m_i` (handle missing labels or padded sequences).  
- **Weights:** per-sample or per-dimension weights `w_i` to emphasize critical components (e.g., large objects in detection boxes).

---

## 6) Scale and Units

- MAE has the **same units** as the target (interpretable).  
- Not scale-invariant: rescaling targets by `α` rescales MAE by `|α|`.  
- For percentage-style interpretation, consider **MAPE** (watch zeros) or **SMAPE** (symmetrized), or compute MAE on a **log-transformed** target if appropriate.

---

## 7) PyTorch Usage

### 7.1 Basic MAE (regression)
'''python
import torch
import torch.nn as nn

y_true = torch.tensor([3.0, 0.0, -2.0])
y_pred = torch.tensor([2.5, 0.2, -3.0])

mae = torch.mean(torch.abs(y_pred - y_true))
print(float(mae))  # 0.5
'''

### 7.2 Built-in Loss: `L1Loss`
'''python
import torch
import torch.nn as nn

crit = nn.L1Loss(reduction='mean')  # 'sum' or 'none' also available
loss = crit(y_pred, y_true)
print(float(loss))
'''

### 7.3 Masked MAE
'''python
import torch

def masked_mae(pred, target, mask):
    # mask: 1 for valid, 0 for invalid
    diff = torch.abs(pred - target) * mask
    denom = mask.sum().clamp_min(1)
    return diff.sum() / denom

pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 1.5, 10.0])
mask = torch.tensor([1.0, 1.0, 0.0])  # ignore last
print(float(masked_mae(pred, target, mask)))
'''

### 7.4 Smooth L1 / Huber Alternative
'''python
import torch
import torch.nn as nn

crit_huber = nn.HuberLoss(delta=1.0, reduction='mean')      # PyTorch >= 1.10
crit_smooth = nn.SmoothL1Loss(beta=1.0, reduction='mean')   # classic
'''

---

## 8) Computer Vision Examples

### 8.1 Image Regression (per-pixel MAE)
'''python
import torch
import torch.nn as nn

# predictions and targets: [N, C, H, W]
pred = torch.rand(8, 3, 64, 64)
tgt  = torch.rand(8, 3, 64, 64)

mae = torch.mean(torch.abs(pred - tgt))
# or channel-weighted:
weights = torch.tensor([0.5, 1.0, 1.5]).view(1,3,1,1)
mae_w = torch.mean(torch.abs(pred - tgt) * weights)
'''

### 8.2 Bounding Boxes (L1 for coordinates)
'''python
import torch

# boxes as [x1, y1, x2, y2], shape [N,4]
pred = torch.tensor([[10.,10.,50.,50.],
                     [ 5., 5.,40.,45.]])
tgt  = torch.tensor([[12.,10.,52.,48.],
                     [ 4., 7.,38.,46.]])

l1_boxes = torch.mean(torch.abs(pred - tgt))  # many detectors use L1/CIoU variants
'''

---

## 9) Time Series and Forecasting

- MAE is robust to spikes; often reported alongside **RMSE**.  
- **MAE vs. RMSE:** RMSE punishes large errors more; MAE provides median-centered error magnitude.  
- **Median forecast:** For quantile regression, MAE corresponds to **quantile τ=0.5**; generalized by pinball (quantile) loss for other τ.

---

## 10) Connections and Extensions

- **Quantile (Pinball) Loss:**  
  `ρ_τ(e) = max(τe, (τ-1)e)`; τ=0.5 gives MAE. Choose τ to target conditional quantiles (e.g., τ=0.9 for P90).
- **Huber / Smooth L1:** differentiable near zero; tradeoff controlled by `δ` or `β`.
- **ε-Insensitive Loss (SVR):** `max(0, |e| - ε)` ignores small errors within a tube; robust and sparse gradients.
- **L1 Regularization (Lasso):** same absolute-value penalty shape, but applied to **parameters** rather than residuals (encourages sparsity).

---

## 11) Training Dynamics and Tips

- **Learning rate:** MAE’s constant-magnitude gradients can slow convergence near optimum; try slightly higher LR or use Huber early, switch to MAE late.  
- **Normalization:** Standardize features; optionally scale targets to a reasonable range (e.g., z-score), but remember to invert for reporting.  
- **Batch size:** Larger batches reduce gradient variance; MAE gradients are already bounded, so modest batches often suffice.  
- **Evaluation:** Report MAE with confidence intervals (e.g., bootstrap) for interpretability.

---

## 12) Plotly Illustration (example)

"""""js_plotly
{
  "data": [
    { "x": ["MSE", "MAE", "Huber(δ=1)"], "y": [1.8, 1.2, 1.0], "type": "bar" }
  ],
  "layout": { "title": "Illustrative Robustness to Outliers (lower is better)" }
}
"""""

---

## 13) Common Pitfalls

- Expecting probability-calibrated outputs: MAE is a regression loss; it does not yield probabilities.  
- Forgetting non-differentiability at zero and then using optimizers sensitive to exact gradients—use subgradients (frameworks handle this) or a smooth alternative.  
- Comparing MAE across differently scaled targets without rescaling or reporting units.  
- Using MAPE on data with zeros instead of MAE; MAPE can explode—prefer MAE or SMAPE.

---

## 14) Minimal Training Loop Sketch (PyTorch)

'''python
import torch
import torch.nn as nn
import torch.optim as optim

model = ...  # define your regression model
criterion = nn.L1Loss(reduction='mean')  # MAE
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    # validation
    with torch.no_grad():
        val_mae = 0.0; n = 0
        for x, y in val_loader:
            pred = model(x)
            val_mae += torch.abs(pred - y).mean().item() * len(x)
            n += len(x)
        print(f"epoch {epoch+1}  val_MAE={val_mae/n:.4f}")
'''

---

## 15) Summary

MAE is a robust, interpretable regression loss that targets the **median** and tolerates outliers better than MSE.  
Use MAE when heavy-tailed noise or large occasional errors are expected; consider **Huber** to smooth optimization while retaining robustness, and **quantile loss** to predict other conditional quantiles when needed.

---
