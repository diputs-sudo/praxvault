# Logistic Regression

Logistic Regression models the probability of a binary outcome conditioned on features.  
It uses a linear score transformed by a sigmoid to predict `P(y=1|x)` and optimizes **log loss** (cross-entropy).

---

## 1) Problem Setup

- **Model (binary)**  
  `p = σ(z)`, with `z = wᵀ x + b` and `σ(t) = 1 / (1 + e^{-t})`  
  Prediction: `ŷ = 1[p ≥ 0.5]` (or a chosen threshold)

- **Loss (Negative Log-Likelihood / Log Loss)**  
  For labels `y ∈ {0,1}`:  
  `L(w,b) = (1/N) Σ_i [ - y_i log p_i - (1 - y_i) log (1 - p_i) ]`

- **Gradient**  
  Let `X1 = [1, x]` include intercept and `θ = [b; w]`. With `p = σ(X1 θ)`:  
  `∇_θ L = (1/N) X1ᵀ (p - y)`

- **Regularization**  
  - L2 (Ridge): `L + λ ||w||²` (don’t penalize intercept)  
  - L1 (Lasso): `L + α ||w||₁` (sparsity)

---

## 2) Algorithms

### 2.1 Batch Gradient Descent (with L2)
'''python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def fit_logreg_gd(X, y, lr=1e-2, steps=2000, lam=0.0):
    """
    X: (N, d) features
    y: (N,) in {0,1}
    """
    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)  # add intercept
    N, D = X1.shape
    theta = np.zeros(D)
    I = np.eye(D); I[0,0] = 0  # don't penalize intercept
    for _ in range(steps):
        z = X1 @ theta
        p = sigmoid(z)
        grad = (X1.T @ (p - y)) / N + 2 * lam * (I @ theta)
        theta -= lr * grad
    return theta  # [b, w...]

# Example:
# theta = fit_logreg_gd(X, y, lr=1e-2, steps=3000, lam=1e-3)
# p = sigmoid(np.c_[np.ones(len(X)), X] @ theta)
'''

### 2.2 Mini-batch SGD (shuffle, weight decay)
'''python
import numpy as np

def fit_logreg_sgd(X, y, lr=1e-2, epochs=20, batch=64, lam=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    N, D = X1.shape
    theta = np.zeros(D)
    I = np.eye(D); I[0,0] = 0
    for _ in range(epochs):
        idx = rng.permutation(N)
        X1p, yp = X1[idx], y[idx]
        for s in range(0, N, batch):
            xb = X1p[s:s+batch]; yb = yp[s:s+batch]
            pb = 1.0 / (1.0 + np.exp(-(xb @ theta)))
            grad = (xb.T @ (pb - yb)) / len(yb) + 2 * lam * (I @ theta)
            theta -= lr * grad
    return theta
'''

### 2.3 Newton–Raphson / IRLS (fast near optimum, small–medium d)
- Uses Hessian `H = (1/N) X1ᵀ W X1 + 2λI` with `W = diag(p(1-p))`.
- Update: `θ ← θ - H⁻¹ ∇L`.

'''python
import numpy as np

def fit_logreg_newton(X, y, lam=0.0, max_iter=50, tol=1e-6):
    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    N, D = X1.shape
    theta = np.zeros(D)
    I = np.eye(D); I[0,0] = 0
    for _ in range(max_iter):
        z = X1 @ theta
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (X1.T @ (p - y)) / N + 2 * lam * (I @ theta)
        w = p * (1 - p)
        W = w  # vector, use as diag
        # H = X1.T @ diag(W) @ X1 / N + 2 lam I
        # compute X1.T * (W * X1) without forming diag:
        Xw = X1 * W[:, None]
        H = (X1.T @ Xw) / N + 2 * lam * I
        step = np.linalg.solve(H, grad)
        theta_new = theta - step
        if np.linalg.norm(step) < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta
'''

---

## 3) Inference & Prediction

- **Probabilities**: `p = σ(wᵀ x + b)`  
- **Class Decision**: threshold `τ` (default `0.5`, tune via ROC/PR trade-off)
- **Calibration**: Platt scaling, isotonic regression if needed

'''python
import numpy as np

def predict_proba(X, theta):
    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return 1.0 / (1.0 + np.exp(-(X1 @ theta)))

def predict_label(X, theta, thresh=0.5):
    return (predict_proba(X, theta) >= thresh).astype(int)
'''

---

## 4) Regularization & Feature Handling

- **L2 (Ridge)**: stabilizes, improves generalization, handles multicollinearity.  
- **L1 (Lasso)**: promotes sparsity (feature selection).  
- **Elastic Net**: mix of L1/L2.  
- **Scaling**: standardize features for GD/L1.  
- **Class Imbalance**: class weights, focal loss (approx), threshold tuning, resampling.

'''python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# L2-regularized LR with standardization and class weights
pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000))
# pipe.fit(X_train, y_train)
# p = pipe.predict_proba(X_valid)[:,1]
'''

---

## 5) Evaluation

- **Log Loss** (training objective)  
- **Accuracy**, **Precision/Recall/F1**, **ROC-AUC**, **PR-AUC** (for imbalance)  
- **Confusion Matrix**; **Calibration curve**

'''python
from sklearn.metrics import (
    log_loss, accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix
)
import numpy as np

def evaluate_binary(y_true, p, thresh=0.5):
    y_pred = (p >= thresh).astype(int)
    ll = log_loss(y_true, p, labels=[0,1])
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, p)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    return {"log_loss": ll, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc, "confusion_matrix": cm}
'''

---

## 6) End-to-End From Scratch (Synthetic Data)

'''python
import numpy as np

# synthetic separable-ish data
rng = np.random.default_rng(42)
N, d = 600, 2
X_pos = rng.normal(loc=1.0, scale=1.0, size=(N//2, d))
X_neg = rng.normal(loc=-1.0, scale=1.0, size=(N//2, d))
X = np.vstack([X_pos, X_neg])
y = np.concatenate([np.ones(N//2, dtype=int), np.zeros(N//2, dtype=int)])

# shuffle
perm = rng.permutation(N)
X, y = X[perm], y[perm]

# split
split = int(0.8 * N)
Xtr, Xva = X[:split], X[split:]
ytr, yva = y[:split], y[split:]

# train with GD (L2)
theta = fit_logreg_gd(Xtr, ytr, lr=5e-2, steps=2000, lam=1e-3)

# evaluate
p_va = predict_proba(Xva, theta)
metrics = evaluate_binary(yva, p_va, thresh=0.5)
print("Metrics:", metrics)
'''

---

## 7) PyTorch-Style Optimization (binary)

'''python
import torch

def fit_logreg_torch(X_np, y_np, lr=1e-2, steps=2000, lam=1e-3, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    # linear layer + bias
    d = X.size(1)
    W = torch.zeros((d, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([W, b], lr=lr)

    for _ in range(steps):
        z = X @ W + b
        p = torch.sigmoid(z)
        bce = torch.nn.functional.binary_cross_entropy(p, y)
        l2 = lam * torch.sum(W**2)
        loss = bce + l2
        opt.zero_grad()
        loss.backward()
        opt.step()

    return W.detach().numpy().squeeze(), b.detach().item()

# Example:
# W, b = fit_logreg_torch(Xtr, ytr, lr=1e-2, steps=2000, lam=1e-3)
# p = torch.sigmoid(torch.tensor(Xva, dtype=torch.float32) @ torch.tensor(W).reshape(-1,1) + b).numpy().ravel()
'''

---

## 8) Multiclass Extension

- **One-vs-Rest (OvR)**: train K binary models (class k vs. rest), pick max probability.  
- **Softmax Regression (Multinomial LR)**:  
  `p_k = exp(w_kᵀ x + b_k) / Σ_j exp(w_jᵀ x + b_j)`; optimize multinomial cross-entropy.

'''python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Softmax (multinomial) logistic regression
softmax_clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0, max_iter=2000)
)
# softmax_clf.fit(X_train, y_train_multi)
# proba = softmax_clf.predict_proba(X_valid)
'''

---

## 9) Thresholding, ROC, and PR

- Tune threshold to trade precision/recall based on costs.  
- Use ROC/PR curves to visualize trade-offs.

"""""js_plotly
{
  "data": [
    {
      "x": [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0],
      "y": [0.0, 0.6, 0.75, 0.82, 0.88, 0.93, 1.0],
      "mode": "lines+markers",
      "name": "ROC (TPR vs. FPR)"
    }
  ],
  "layout": { "title": "ROC Curve (Illustrative)", "xaxis": {"title": "FPR"}, "yaxis": {"title": "TPR"} }
}
"""""

---

## 10) Practical Notes

- Standardize features (esp. for L1/L2, SGD).  
- Use class weights for imbalance or resample data.  
- Prefer **LBFGS/Newton/‘liblinear’/‘saga’** solvers when feasible; SGD for streaming/large-scale.  
- Check calibration if probabilities matter.  
- Avoid penalizing the intercept.

---

## 11) Minimal Scikit-learn Example

'''python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# X, y given (binary)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)
)
clf.fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:,1]
yhat = (p >= 0.5).astype(int)

print(classification_report(yte, yhat, digits=3))
print("ROC-AUC:", roc_auc_score(yte, p))
'''

---

## 12) Summary

Optimizes **log loss** to predict `P(y=1|x)` with a linear decision function.  
Efficient solvers (Newton/Quasi-Newton, SGD) and regularization make it robust, scalable, and a strong baseline for probabilistic classification.  
Extend to multiclass via **softmax** or **OvR**, tune thresholds for application-specific costs, and validate with **ROC/PR** and calibration.
