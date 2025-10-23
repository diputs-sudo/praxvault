# Kullback–Leibler Divergence (KLDiv)

**Kullback–Leibler divergence** `KL(P‖Q)` measures how one probability distribution `P` diverges from another reference distribution `Q`.  
It is **non-symmetric** and **non-negative**, and equals zero iff `P` and `Q` are the same almost everywhere.  
KL divergence appears across ML in **maximum likelihood**, **cross-entropy**, **variational inference**, **knowledge distillation**, **RL**, and **calibration**.

---

## 1) Definitions and Basic Properties

### 1.1 Discrete form
`KL(P‖Q) = Σ_x P(x) log( P(x) / Q(x) )`, with the convention `0·log(0/q)=0` and requiring `Q(x)>0` whenever `P(x)>0`.

### 1.2 Continuous form
`KL(P‖Q) = ∫ p(x) log( p(x) / q(x) ) dx`, assuming absolute continuity of `P` w.r.t. `Q`.

### 1.3 Non-negativity and equality
`KL(P‖Q) ≥ 0` with equality iff `P=Q` a.e. (Gibbs’ inequality).

### 1.4 Asymmetry
Generally `KL(P‖Q) ≠ KL(Q‖P)`. Direction matters:
- `KL(P‖Q)` penalizes when `Q` **misses** probability mass where `P` has support (mode-covering).
- `KL(Q‖P)` penalizes assigning mass where `P` has **none** (mode-seeking).

---

## 2) Relation to Entropy, Cross-Entropy, and Likelihood

### 2.1 CE decomposition
`H(P, Q) = H(P) + KL(P‖Q)` where  
- Entropy: `H(P) = -Σ P log P`  
- Cross-entropy: `H(P, Q) = -Σ P log Q`  
Minimizing cross-entropy in supervised learning is equivalent to minimizing `KL(P‖Q)` since `H(P)` is constant.

### 2.2 Maximum likelihood
Given data from `P_data`, minimizing `KL(P_data‖Q_θ)` over model `Q_θ` is equivalent to maximizing the (average) log-likelihood under `Q_θ`.

---

## 3) Closed Forms for Common Families

### 3.1 Bernoulli
For `P=Bern(p)`, `Q=Bern(q)`:
`KL = p log(p/q) + (1-p) log((1-p)/(1-q))`.

### 3.2 Categorical
`KL = Σ_k p_k log(p_k / q_k)`.

### 3.3 Gaussian (multivariate, full covariance)
For `P=𝓝(μ₁, Σ₁)`, `Q=𝓝(μ₂, Σ₂)` in `d` dimensions:

`KL = 0.5 * [ tr(Σ₂^{-1} Σ₁) + (μ₂−μ₁)^T Σ₂^{-1} (μ₂−μ₁) − d + log( det Σ₂ / det Σ₁ ) ]`

Diagonal covariances reduce this elementwise.

---

## 4) Symmetrized Alternatives

- **Jensen–Shannon divergence**: `JS(P‖Q) = 0.5 KL(P‖M) + 0.5 KL(Q‖M)` with `M=(P+Q)/2`. Symmetric, bounded, √JS is a metric.
- **Symmetric KL**: `KL(P‖Q) + KL(Q‖P)` (not a metric but often used).

---

## 5) Practical Appearances in ML

### 5.1 Variational Inference (VI)
- **ELBO**: `log p(x) ≥ 𝐄_{q(z|x)}[log p(x,z) − log q(z|x)] = 𝐄[log p(x|z)] − KL(q(z|x)‖p(z))`.
- The KL term regularizes the approximate posterior toward the prior.

### 5.2 VAEs
- Reconstruction loss + KL between encoder `q(z|x)` and prior `p(z)` (often standard normal).  
- `β-VAE` scales the KL to control disentanglement and capacity.

### 5.3 Knowledge Distillation
- Student matches teacher distribution via `KL(P_teacher‖Q_student)` (or CE with soft targets), often with **temperature scaling**.

### 5.4 Reinforcement Learning
- Policy optimization with a **trust region**: constrain `KL(π_old‖π_new)` or `KL(π_new‖π_old)` (TRPO, PPO) to keep updates stable.

### 5.5 Calibration and Temperature
- Calibrate logits by minimizing `KL(P_val‖Q_T)` or CE on validation with `softmax(z/T)`; `T>1` usually reduces overconfidence.

---

## 6) Numerical Concerns and Estimation

- **Support mismatch**: if `Q(x)=0` where `P(x)>0`, `KL(P‖Q)=∞`. Add **smoothing** or ensure support overlap.
- **Stability**: compute from **log probabilities** using `logsumexp` tricks.
- **Continuous KL**: if closed form is unavailable, estimate via **Monte Carlo** samples from `P`:  
  `KL ≈ (1/N) Σ_i [log p(x_i) − log q(x_i)]`.

---

## 7) Direction Choice: When P‖Q vs Q‖P?

- **Mode covering** (don’t miss true modes): minimize `KL(P_data‖Q)`; common in MLE.
- **Mode seeking** (avoid spurious modes): sometimes minimize `KL(Q‖P)`; appears in certain VI choices (reverse vs forward KL).

---

## 8) PyTorch Usage and Gotchas

**Key rule:** `torch.nn.KLDivLoss` expects **log-probabilities** as input and **probabilities** (or optionally log-probabilities) as target.

### 8.1 Multiclass KL (forward direction) per sample
'''python
import torch
import torch.nn as nn
import torch.nn.functional as F

# logits from model and reference target distribution p (soft targets)
logits_q = torch.randn(4, 10)           # student logits
p = F.softmax(torch.randn(4, 10), dim=1)  # teacher probs or labels -> probs

log_q = F.log_softmax(logits_q, dim=1)  # log-probs for input
kld = nn.KLDivLoss(reduction='batchmean')  # averages KL over batch
loss = kld(log_q, p)                    # ≈ mean_i Σ p_i log(p_i / q_i)
print(loss.item())
'''

### 8.2 Using `log_target=True` (PyTorch ≥ 1.10)
If the target is in **log-space**, you can avoid converting to probs:
'''python
log_q = F.log_softmax(logits_q, dim=1)
log_p = F.log_softmax(torch.randn(4,10), dim=1)  # teacher in log-space
kld = nn.KLDivLoss(reduction='batchmean', log_target=True)
loss = kld(log_q, log_p)  # computes KL(P‖Q) from log_p and log_q
'''

### 8.3 Symmetric KL (use with care)
'''python
def symmetric_kl(logits_a, logits_b):
    log_a = F.log_softmax(logits_a, dim=1)
    log_b = F.log_softmax(logits_b, dim=1)
    a = log_a.exp(); b = log_b.exp()
    kld = nn.KLDivLoss(reduction='batchmean')
    return kld(log_a, b) + kld(log_b, a)
'''

### 8.4 Temperature scaling for distillation
'''python
T = 4.0
log_q_T = F.log_softmax(logits_q / T, dim=1)
p_T     = F.softmax(logits_teacher / T, dim=1)
kld = nn.KLDivLoss(reduction='batchmean')
distill_loss = (T*T) * kld(log_q_T, p_T)   # scale by T^2 (gradient correction)
'''

### 8.5 Continuous KL via Monte Carlo (example)
Suppose `p` and `q` are tractable densities on ℝᵈ; sample from `p`:
'''python
import torch

def mc_kl_from_p(p_sampler, logp, logq, N=8192, device='cpu'):
    x = p_sampler(N).to(device)
    return (logp(x) - logq(x)).mean()

# Example: Gaussian with different means, diagonal covariances
d = 4
mu_p = torch.zeros(d);  logstd_p = torch.zeros(d)
mu_q = torch.ones(d);   logstd_q = torch.zeros(d)

def sample_p(n):
    return mu_p + torch.randn(n, d) * torch.exp(logstd_p)

def log_gauss(x, mu, logstd):
    var = torch.exp(2*logstd)
    return -0.5*( ((x-mu)**2/var).sum(-1) + (2*logstd).sum() + d*torch.log(torch.tensor(2*3.1415926535)) )

kl_mc = mc_kl_from_p(sample_p,
                     lambda x: log_gauss(x, mu_p, logstd_p),
                     lambda x: log_gauss(x, mu_q, logstd_q),
                     N=20000)
print(float(kl_mc))
'''

---

## 9) Applications and Recipes

- **Distillation**: use KL with temperature between teacher and student; combine with standard CE on hard labels.  
- **VAE/VI**: if latent posteriors are Gaussian, use the closed form KL; otherwise reparameterize and estimate with MC.  
- **RL (PPO/TRPO)**: add a penalty or constraint on policy KL; monitor `KL(π_old‖π_new)` online.  
- **Calibration**: fit temperature by minimizing NLL (equivalently KL from empirical label distribution).  
- **Domain adaptation**: minimize KL between feature distributions or output posteriors across domains.

---

## 10) Common Pitfalls

- **Wrong inputs to `KLDivLoss`**: passing probabilities instead of log-probs as the input. Use `log_softmax` for the input side.  
- **Forgetting support overlap**: zeros in `Q` where `P>0` cause infinite KL—add smoothing or label smoothing.  
- **Direction confusion**: `KL(P‖Q)` vs `KL(Q‖P)` lead to different behavior (mode covering vs seeking).  
- **Temperature scaling without `T^2` factor** in distillation—gradients get mis-scaled.  
- **Batch reduction**: `reduction='batchmean'` is often preferred to represent the mean KL per sample.

---

## 11) Plotly Illustration (example)

"""""js_plotly
{
  "data": [
    { "x": ["KL(P‖Q)", "KL(Q‖P)", "JS(P‖Q)"], "y": [0.52, 1.34, 0.43], "type": "bar" }
  ],
  "layout": { "title": "Illustrative Divergences Between Two Distributions" }
}
"""""

---

## 12) Checklist

- Decide **direction** of KL based on behavior you want (covering vs seeking).  
- Use **log-probabilities** for numerical stability; prefer `logsumexp` operations.  
- For **categorical** logits: use `log_softmax` and pass soft targets correctly.  
- For **VAEs**: exploit closed-form Gaussian KL; otherwise MC with reparameterization.  
- For **distillation**: apply temperature and multiply loss by `T²`.  
- Monitor KL in training when used as a **constraint/penalty** (e.g., PPO).

---

## 13) Summary

KL divergence quantifies how a model distribution differs from a reference.  
It connects cross-entropy and maximum likelihood, shapes approximate inference via ELBO, stabilizes RL policy updates, and aligns student models to teachers.  
Practical success hinges on choosing the **direction**, using **log-space computations**, ensuring **support overlap**, and leveraging **closed forms** or **Monte Carlo** estimation where appropriate.

---
