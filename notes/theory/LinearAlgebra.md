# Linear Algebra

**Definition**  
- **Vectors**: ordered lists of numbers (elements in ℝⁿ).  
- **Matrices**: 2D arrays (elements in ℝ^{m×n}).  
- **Tensors**: generalization to ≥3 dimensions (e.g., images: `(C,H,W)`, batches: `(N,…)`).  

---

## Notation & Shapes

- Vector `x ∈ ℝ^d` (column by default).  
- Matrix `A ∈ ℝ^{m×n}` maps `ℝ^n → ℝ^m`.  
- Dataset matrix `X ∈ ℝ^{N×d}`: `N` samples, `d` features.  
- Identity `I_d`, zeros `0`, ones `1`.  
- Transpose `Aᵀ`, inverse `A⁻¹` (exists iff `A` is square and full-rank).  

---

## Core Operations

### Dot Product (inner product)
- For `x,y ∈ ℝ^d`: `x·y = Σ_i x_i y_i = xᵀ y`.  
- Geometric: `x·y = ||x|| ||y|| cos θ`.  
- Use cases: similarity, projections, attention scores.

### Matrix–Vector, Matrix–Matrix Multiplication
- `y = A x`, shape `(m×n)·(n) → (m)`.  
- `C = A B`, shape `(m×n)·(n×p) → (m×p)`.  
- Associative: `(AB)C = A(BC)`; **not** commutative in general.

### Transpose
- `(Aᵀ)_{ij} = A_{ji}`.  
- `(AB)ᵀ = Bᵀ Aᵀ`, `(xᵀ)ᵀ = x`.  
- For vectors, `xᵀ y` is scalar; `y xᵀ` is rank-1 matrix.

### Inverse (when it exists)
- `A A⁻¹ = I`.  
- Solving `Ax=b`: `x = A⁻¹ b` (but prefer linear solves/QR/SVD in practice).  
- For non-square: Moore–Penrose pseudoinverse `A⁺`.

### Eigenvalues / Eigenvectors
- `A v = λ v` for square `A`.  
- Diagonalizable `A = V Λ V⁻¹` (if enough eigenvectors).  
- For symmetric `Σ = Σᵀ`: `Σ = Q Λ Qᵀ` with orthonormal `Q`.

### (Bonus) SVD (very useful)
- `A = U Σ Vᵀ` for any `A`.  
- Low-rank approximation, PCA, denoising.

---

## ML Relevance

- **Datasets as matrices**: rows = samples, cols = features → vectorized compute.  
- **Weight matrices**: linear/regression layers are matrix multiplies: `ŷ = XW + b`.  
- **PCA**: eigen-decomposition of covariance to find principal components: directions of max variance.  
- **Distances/similarity**: dot products and norms drive nearest neighbors, kernels, attention.  

---

## Worked Examples

### Represent images as vectors/matrices
- Grayscale image `I ∈ ℝ^{H×W}`; flatten to `x ∈ ℝ^{HW}`.  
- Color image `I ∈ ℝ^{H×W×3}`; treat channels as stacked features or use `(C,H,W)` tensors.

'''python
# NumPy: image as matrix and vector
import numpy as np

H, W = 28, 28
I = np.arange(H*W, dtype=np.float32).reshape(H, W)       # toy "image"
x = I.reshape(-1)                                        # flatten to vector (784,)
I_recon = x.reshape(H, W)                                # back to 2D

print(I.shape, x.shape, I_recon.shape)
'''

---

### Matrix multiplication in NumPy and PyTorch
'''python
# NumPy matmul: C = A @ B
import numpy as np

A = np.random.randn(4, 3)
B = np.random.randn(3, 2)
C = A @ B  # (4x2)
print(C.shape)
'''

'''python
# PyTorch matmul and linear layer equivalence
import torch
A = torch.randn(4, 3)
B = torch.randn(3, 2)
C = A @ B  # (4x2)

# Linear layer: y = XWᵀ + b in PyTorch's Linear
lin = torch.nn.Linear(3, 2, bias=True)
with torch.no_grad():
    lin.weight.copy_(B.T)  # Wᵀ has shape (out,in) = (2,3)
    lin.bias.zero_()
C2 = lin(A)  # equals A @ B
print(torch.allclose(C, C2, atol=1e-6))
'''

---

### Linear Regression as matrix math
- Model: `ŷ = X w` (assume centered, no bias)  
- Normal equation (when feasible): `w* = (XᵀX)⁻¹ Xᵀ y` (use solves/QR/SVD in practice)

'''python
# NumPy: closed-form linear regression via solve
import numpy as np

N, d = 200, 5
X = np.random.randn(N, d)
w_true = np.random.randn(d)
y = X @ w_true + 0.1*np.random.randn(N)

XtX = X.T @ X
Xty = X.T @ y
w_hat = np.linalg.solve(XtX, Xty)  # better than computing inverse explicitly
print("RMSE:", np.sqrt(np.mean((y - X @ w_hat)**2)))
'''

---

### PCA via eigen-decomposition (covariance) and SVD
- Center data: `X₀ = X - mean(X)`  
- Covariance: `Σ = (1/N) X₀ᵀ X₀`  
- Eigen-decomposition: `Σ = Q Λ Qᵀ`  
- Principal components = columns of `Q` (top-`k` eigenvectors)

'''python
# NumPy: PCA from covariance (eigh) and via SVD
import numpy as np

# synthetic data with correlated features
N, d = 500, 3
Z = np.random.randn(N, d)
M = np.array([[1.0, 0.9, 0.2],
              [0.9, 1.0, 0.1],
              [0.2, 0.1, 1.0]])  # correlation-ish
X = Z @ np.linalg.cholesky(M).T

# center
X0 = X - X.mean(axis=0, keepdims=True)

# Covariance eigen-decomposition (symmetric)
Sigma = (X0.T @ X0) / (N - 1)
eigvals, eigvecs = np.linalg.eigh(Sigma)   # ascending order
idx = eigvals.argsort()[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

# Project onto first k components
k = 2
W = eigvecs[:, :k]        # (d,k)
X_proj = X0 @ W           # (N,k)

# Alternatively: SVD of centered data
U, S, Vt = np.linalg.svd(X0, full_matrices=False)  # X0 = U S Vᵀ
# Right singular vectors V are PCs; eigenvalues = (S^2)/(N-1)
V = Vt.T
X_proj_svd = X0 @ V[:, :k]

print("Explained variance ratios (eig):", eigvals / eigvals.sum())
print("Subspace match (cosine of principal angles):",
      np.abs(np.linalg.svd(W.T @ V[:, :k], compute_uv=False)))
'''

---

### Eigenvalues / Eigenvectors in practice
- Symmetric (Hermitian) matrices have **real** eigenvalues and orthonormal eigenvectors → numerically stable with `eigh`.  
- Use cases: PCA, spectral clustering, quadratic forms, Laplacians.

'''python
# NumPy: eigenvalues of a symmetric matrix
import numpy as np

A = np.random.randn(4,4)
S = (A + A.T)/2  # make symmetric
vals, vecs = np.linalg.eigh(S)
recon = vecs @ np.diag(vals) @ vecs.T
print("Spectral recon error:", np.linalg.norm(S - recon))
'''

---

## PyTorch Tensor Shapes & Broadcasting Cheatsheet

- **Batch-first**: `(N, C, H, W)` for images; `(N, T, D)` for sequences.  
- **Broadcasting**: singleton dims expand without copying: `(N,1) + (N,d) → (N,d)`.  
- **Transpose/permute**: `x.transpose(dim0, dim1)`, `x.permute(...)`.  
- **Matrix ops**: `@`, `matmul`, `mm`, `bmm` (batched).  

'''python
import torch

X = torch.randn(32, 128)       # (N,d)
w = torch.randn(128, 10)       # (d,k)
b = torch.randn(10)            # (k,)
Y = X @ w + b                  # broadcasting adds bias to each row
print(Y.shape)                 # (32,10)
'''

---

## Tips & Best Practices

- Prefer linear **solves** (`solve`, `lstsq`, `qr`, `svd`) over explicit inverses.  
- Standardize/center features when using PCA or solving ill-conditioned systems.  
- Inspect condition numbers; use **regularization** (e.g., ridge) when `XᵀX` is near-singular.  
- Keep track of shapes; assert early to catch dim mismatches.

---
