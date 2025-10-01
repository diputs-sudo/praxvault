# Linear Regression

Linear Regression models a continuous target as a weighted sum of input features.  
We fit parameters to minimize a loss (typically MSE), using closed-form solutions or gradient-based methods.

---

## 1) Problem Setup

- **Model**  
  `ŷ = wᵀ x + b`

- **Loss (MSE)**  
  `L(w,b) = (1/N) Σ (y - (wᵀ x + b))²`

- **Matrix Form (with intercept)**  
  Let `X ∈ ℝ^{N×(d+1)}` with a first column of ones and `θ = [b; w]`.  
  `ŷ = Xθ`, `L(θ) = (1/N) ||y - Xθ||²`.

---

## 2) Solutions

### 2.1 Closed Form (Normal Equation, Ridge-ready)
'''python
import numpy as np

def fit_normal_eq(X, y, lam=0.0):
    """
    X: (N, d) features without intercept
    y: (N,) targets
    lam: L2 (ridge) strength; lam=0 -> OLS
    """
    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)  # add intercept
    D = X1.shape[1]
    I = np.eye(D); I[0, 0] = 0  # don't penalize intercept
    theta = np.linalg.solve(X1.T @ X1 + lam * I, X1.T @ y)
    return theta  # [b, w...]

# Example usage:
# theta = fit_normal_eq(X, y, lam=1e-2)
# b, w = theta[0], theta[1:]
'''

### 2.2 Batch Gradient Descent (with optional L2)
'''python
import numpy as np

def fit_gd(X, y, lr=1e-2, steps=2000, lam=0.0):
    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    N, D = X1.shape
    theta = np.zeros(D)
    I = np.eye(D); I[0,0] = 0
    for _ in range(steps):
        err = X1 @ theta - y
        grad = (2/N) * (X1.T @ err) + 2 * lam * (I @ theta)
        theta -= lr * grad
    return theta

# theta = fit_gd(X, y, lr=1e-2, steps=3000, lam=1e-2)
'''

### 2.3 Mini-batch SGD (shuffle, weight decay)
'''python
import numpy as np

def fit_sgd(X, y, lr=1e-2, epochs=20, batch=64, lam=0.0, seed=0):
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
            err = xb @ theta - yb
            grad = (2/len(yb)) * (xb.T @ err) + 2 * lam * (I @ theta)
            theta -= lr * grad
    return theta
'''

---

## 3) Regularization & Variants

### 3.1 Ridge (L2), Lasso (L1), Elastic Net (L1+L2)
'''python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Ridge with scaling
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge.fit(X, y)
y_pred_ridge = ridge.predict(X)

# Lasso (feature selection), often needs stronger scaling and more iterations
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.05, max_iter=20000))
lasso.fit(X, y)
y_pred_lasso = lasso.predict(X)

# Elastic Net
enet = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000))
enet.fit(X, y)
y_pred_enet = enet.predict(X)
'''

### 3.2 Polynomial & Interaction Features
'''python
# Linear-in-parameters but nonlinear-in-features
poly2 = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0))
poly2.fit(X, y)
y_pred_poly2 = poly2.predict(X)
'''

### 3.3 Robust Regression (Huber)
'''python
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

robust = Pipeline([("scaler", StandardScaler()), ("huber", HuberRegressor(alpha=0.0))])
robust.fit(X, y)
y_pred_robust = robust.predict(X)
'''

---

## 4) Evaluation & Diagnostics

### 4.1 Metrics (MSE, RMSE, MAE, R²)
'''python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def eval_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
'''

### 4.2 Residual Analysis (quick)
'''python
import numpy as np

def residuals(y_true, y_pred):
    return y_true - y_pred

# Example:
# res = residuals(y, y_pred_ridge)
# Inspect patterns, variance, outliers programmatically or via plots
'''

---

## 5) End-to-End From Scratch (Synthetic Data)

'''python
import numpy as np

# synthetic data
rng = np.random.default_rng(7)
N, d = 400, 3
X = rng.normal(size=(N, d))
true_w = np.array([2.0, -1.0, 0.5])
b_true = 1.5
y = X @ true_w + b_true + rng.normal(scale=0.5, size=N)

# closed-form ridge
X1 = np.concatenate([np.ones((N,1)), X], axis=1)
lam = 1e-2
I = np.eye(d+1); I[0,0] = 0
theta = np.linalg.solve(X1.T @ X1 + lam*I, X1.T @ y)

b_hat, w_hat = theta[0], theta[1:]
y_pred = X1 @ theta

rmse = np.sqrt(np.mean((y - y_pred)**2))
print("b_hat:", b_hat)
print("w_hat:", w_hat)
print("RMSE:", rmse)
'''

---

## 6) PyTorch-Style Optimization (with L2)

'''python
import torch

def fit_torch_linear(X_np, y_np, lr=1e-2, steps=2000, lam=1e-2, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    ones = torch.ones(X.size(0), 1)
    X1 = torch.cat([ones, X], dim=1)

    theta = torch.zeros(X1.size(1), 1, requires_grad=True)
    opt = torch.optim.SGD([theta], lr=lr)

    for _ in range(steps):
        y_hat = X1 @ theta
        mse = torch.mean((y_hat - y)**2)
        l2 = lam * torch.sum(theta[1:]**2)  # no penalty on intercept
        loss = mse + l2
        opt.zero_grad()
        loss.backward()
        opt.step()

    return theta.detach().squeeze()

# Example:
# theta_t = fit_torch_linear(X, y, lr=1e-2, steps=2000, lam=1e-2)
# b_t, w_t = theta_t[0].item(), theta_t[1:].numpy()
'''

---

## 7) Cross-Validation & Pipelines

'''python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, scoring="neg_root_mean_squared_error", cv=cv)
print("CV RMSE (mean ± std):", -scores.mean(), "+/-", scores.std())
'''

---

## 8) Visualization (Plotly Example)

"""""js_plotly
{
  "data": [
    {
      "x": [0, 1, 2, 3, 4, 5],
      "y": [1.1, 2.0, 2.8, 4.1, 5.0, 6.2],
      "mode": "markers",
      "name": "Data"
    },
    {
      "x": [0, 1, 2, 3, 4, 5],
      "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      "mode": "lines",
      "name": "Fitted Line"
    }
  ],
  "layout": { "title": "Linear Regression Fit (Illustrative)" }
}
"""""

---

## 9) Troubleshooting

- Diverging GD/SGD → lower learning rate; standardize features.  
- Unstable closed form → use SVD/QR solvers; add L2 regularization.  
- Overfitting → Ridge/Lasso/Elastic Net; cross-validate; reduce polynomial degree.  
- Poor residuals → add nonlinear features or switch models.  
- Multicollinearity → L2 regularization or dimensionality reduction (PCA).

---

## 10) Summary

Use closed form for small/medium problems, GD/SGD for scale and flexibility.  
Control complexity with regularization, validate with cross-validation, and assess residuals to decide when to enrich the model.
