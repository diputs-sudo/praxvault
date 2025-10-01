# Probability

**Definition**  
- **Random variables (RVs)**: variables whose values are outcomes of randomness (discrete or continuous).  
- **Distributions**: rules that assign probabilities to outcomes or ranges.  
- **Expectation** `E[X]`: average value in the long run.  
- **Variance** `Var(X) = E[(X - E[X])²]`: dispersion around the mean.

---

## Core Concepts

### Conditional Probability
`P(A|B) = P(A ∩ B) / P(B)` (when `P(B) > 0`) — probability of `A` given `B`.

### Bayes’ Rule
`P(A|B) = P(B|A) P(A) / P(B)`  
Posterior ∝ Likelihood × Prior.

### Likelihood
For parameter θ and data `D`, **likelihood** `L(θ; D) = P(D | θ)` (viewed as a function of θ).  
Maximum Likelihood Estimation (MLE): choose θ that maximizes `L(θ; D)` (or log-likelihood).

### Independence
Events `A` and `B` are independent if `P(A ∩ B) = P(A) P(B)`.  
RVs `X` and `Y` independent if their joint PDF/PMF factorizes: `p(x,y) = p(x) p(y)`.

---

## Common Distributions

- **Bernoulli(p)**: `X ∈ {0,1}`, `P(X=1)=p`, `E[X]=p`, `Var(X)=p(1-p)`.  
- **Binomial(n,p)**: sum of `n` iid Bernoullis; `E[X]=np`, `Var(X)=np(1-p)`.  
- **Categorical(π)**: one of K classes; probs `π_k`.  
- **Gaussian(μ,σ²)**: continuous; `E[X]=μ`, `Var(X)=σ²`.

---

## ML Relevance

- **Logistic regression** models `P(y=1|x) = σ(wᵀx + b)`.  
- **Loss functions**: CrossEntropy = **negative log-likelihood** under categorical/Bernoulli models.  
- **Uncertainty**: predictive probabilities, calibration, Bayesian posteriors.  
- **Regularization**: can be seen as a prior in MAP estimation.

---

## Worked Example: Coin Flip → Logistic Link

- Let `X ~ Bernoulli(p)` describe a coin (1=heads, 0=tails).  
- Log-likelihood for `N` flips `{x_i}`:  
  `ℓ(p) = Σ [ x_i log p + (1 - x_i) log(1 - p) ]`  
  MLE: `p̂ = (1/N) Σ x_i` (sample mean).  
- Logistic regression uses `p = σ(z)` with `z = wᵀx + b` so that `0 < p < 1`.

---

## Code Snippets

### Simulate coin flips; estimate p (NumPy)
'''python
import numpy as np

rng = np.random.default_rng(0)
N = 1000
p_true = 0.62
x = rng.binomial(1, p_true, size=N)  # 0/1 samples

p_hat = x.mean()
loglik = np.sum(x*np.log(p_hat + 1e-12) + (1-x)*np.log(1 - p_hat + 1e-12))

print(f"True p={p_true:.3f}  MLE p_hat={p_hat:.3f}  loglik={loglik:.2f}")
'''

### Bernoulli/Cross-Entropy = Negative Log-Likelihood
For predictions `p_i` and labels `y_i ∈ {0,1}`:  
`CE = - (1/N) Σ [ y_i log p_i + (1 - y_i) log(1 - p_i) ]`

'''python
import numpy as np

def binary_cross_entropy(p, y):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

# toy predictions vs labels
p = np.array([0.8, 0.4, 0.7, 0.1])
y = np.array([1,   0,   1,   0  ])
print("CE:", binary_cross_entropy(p, y))
'''

### Logistic Regression probability & NLL (PyTorch)
'''python
import torch
import torch.nn.functional as F

# synthetic features and labels
torch.manual_seed(0)
N, d = 200, 3
X = torch.randn(N, d)
w_true = torch.randn(d)
b_true = 0.2
logits = X @ w_true + b_true
p_true = torch.sigmoid(logits)
y = torch.bernoulli(p_true)  # 0/1 labels sampled from Bernoulli(p_true)

# model parameters
w = torch.zeros(d, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=1e-1)

for step in range(200):
    z = X @ w + b
    # BCEWithLogits combines sigmoid + CE stably
    loss = F.binary_cross_entropy_with_logits(z, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

p_pred = torch.sigmoid(X @ w + b)
print("Final NLL (BCE):", float(loss))
print("Mean predicted prob:", float(p_pred.mean()))
'''

### Expectation & Variance by Monte Carlo
'''python
import numpy as np

# Gaussian samples → estimate E[X], Var(X)
mu, sigma = 1.5, 2.0
xs = rng.normal(mu, sigma, size=100000)
print("MC mean ~", xs.mean(), "  MC var ~", xs.var())
'''

### Conditional Probability via Simulation
Estimate `P(A|B)` by counting:
'''python
import numpy as np

rng = np.random.default_rng(1)
N = 200000
# Let A = {sum of two dice == 7}, B = {at least one die is 3}
dice = rng.integers(1, 7, size=(N, 2))
A = (dice.sum(axis=1) == 7)
B = (dice[:,0] == 3) | (dice[:,1] == 3)
p_A = A.mean()
p_B = B.mean()
p_A_and_B = (A & B).mean()
p_A_given_B = p_A_and_B / max(p_B, 1e-12)

print(f"P(A)={p_A:.4f}, P(B)={p_B:.4f}, P(A∧B)={p_A_and_B:.4f}, P(A|B)={p_A_given_B:.4f}")
'''

---

## Quick Identities & Tips

- **Law of Total Probability**: `P(A) = Σ_k P(A|B_k) P(B_k)` for a partition `{B_k}`.  
- **Total Expectation**: `E[X] = E[ E[X|Y] ]`.  
- **Variance**: `Var(X) = E[X^2] - (E[X])^2`.  
- For numerical stability in NLL/CE, use **logits** (e.g., `BCEWithLogitsLoss`).  
- Calibrate probabilities if decisions depend on absolute risk (temperature scaling, isotonic).

---
