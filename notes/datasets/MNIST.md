# MNIST

**MNIST** (Modified National Institute of Standards and Technology) is a benchmark dataset of **handwritten digits** 0–9.  
It contains **70,000 grayscale images** at **28×28** resolution: **60,000** training images and **10,000** test images.  
MNIST is widely used for teaching fundamentals, prototyping models, and verifying training pipelines end-to-end.

---

## 1) Dataset Overview

### 1.1 Composition
- Samples: `N = 70,000` grayscale images, `28×28` pixels.
- Split: `60,000` train, `10,000` test (fixed test set).
- Channels: single-channel (grayscale), values typically in `[0, 255]` or `[0.0, 1.0]` after scaling.
- Task: single-label classification of digits `0–9`.

### 1.2 Labels
- Classes: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`.
- Labels are integers `0–9` matching the digit.

### 1.3 Access and Format
- Commonly loaded via deep learning libraries (e.g., PyTorch `torchvision.datasets.MNIST`).
- Two files per split: image tensors and label vectors; loaders handle parsing.

---

## 2) Preprocessing and Normalization

### 2.1 Standard Normalization (common statistics)
- Mean: `μ = 0.1307`
- Std:  `σ = 0.3081`
- Normalize each pixel `x` (0–1 float) as `x' = (x - μ) / σ`.

### 2.2 Recommended Transforms
- **ToTensor** (scales to `[0,1]`).
- **Normalize(mean=0.1307, std=0.3081)**.
- Optional light augmentations:
  - **RandomAffine** (small rotations ±10°, minor translations).
  - **RandomErasing** (rarely needed; use conservatively).
  - Avoid heavy color transforms (dataset is grayscale).

---

## 3) Evaluation Protocol

### 3.1 Metrics
- **Top-1 accuracy** on the 10k test set.
- Track **validation accuracy** if carving a validation split from train (e.g., 55k/5k).

### 3.2 Logging
- Plot curves for: train loss, train accuracy, validation/test accuracy, learning rate.
- Optionally monitor gradient norms and weight norms for debugging.

### 3.3 Reproducibility
- Fix seeds, log library versions and hardware, save checkpoints and configs.
- Beware nondeterminism in certain GPU kernels; document flags if determinism is required.

---

## 4) Baseline Models

### 4.1 Linear Classifier (Softmax Regression)
- Input: `784` (flattened 28×28).
- Output: `10` logits; softmax cross-entropy loss.
- Purpose: sanity check data pipeline and optimization loop.

### 4.2 MLP (1–2 Hidden Layers)
- Example: `784 → 300 → 100 → 10`, ReLU activations.
- Good stepping stone to CNNs; test regularization (dropout, weight decay).

### 4.3 LeNet-style CNN
- Classic architecture for small grayscale images:
  - `Conv(1→6, 5×5)` → ReLU → Avg/MaxPool
  - `Conv(6→16, 5×5)` → ReLU → Pool
  - Flatten → `FC(16×4×4 → 120)` → ReLU → `FC(120→84)` → ReLU → `FC(84→10)`

### 4.4 Modern Small CNN
- Two or three `Conv-BN-ReLU` blocks with 3×3 kernels, pooling/stride, then `FC→10`.
- Often reaches >99% with proper training.

---

## 5) Optimization and Schedules

- **Optimizers:**  
  - SGD + Momentum (0.9), weight decay `1e-4`.  
  - AdamW (`lr≈1e-3`, weight decay `1e-4`) as a robust default.
- **Learning rates:**  
  - Linear/MLP with SGD: `lr ∈ [1e-2, 1e-1]`.  
  - CNN with SGD: `lr ∈ [1e-2, 5e-2]`.  
  - AdamW: `lr ∈ [1e-3, 3e-3]`.
- **Schedules:**  
  - Step decay at a few milestones for short runs.  
  - Cosine annealing with short warmup (e.g., 1–3 epochs) for longer runs.

---

## 6) Data Loading and Transforms (PyTorch)

### 6.1 Basic Pipeline
'''python
import torch
import torchvision as tv
import torchvision.transforms as T

mean, std = (0.1307,), (0.3081,)

train_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

train_set = tv.datasets.MNIST(root="./data", train=True,  download=True, transform=train_tf)
test_set  = tv.datasets.MNIST(root="./data", train=False, download=True, transform=test_tf)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
'''

### 6.2 With Light Augmentation (Affine)
'''python
train_tf_aug = T.Compose([
    T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    T.ToTensor(),
    T.Normalize(mean, std),
])
'''

---

## 7) Minimal Models and Training Loops

### 7.1 Linear Classifier
'''python
import torch
import torch.nn as nn
import torch.optim as optim

class LinearMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LinearMNIST().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

def train_epoch(loader):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total
'''

### 7.2 LeNet-style CNN
'''python
import torch
import torch.nn as nn
import torch.optim as optim

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(inplace=True), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),            nn.ReLU(inplace=True), nn.AvgPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120), nn.ReLU(inplace=True),
            nn.Linear(120, 84),     nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

def train_one_epoch(loader):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

# Example training loop (5–10 epochs already achieves high accuracy)
for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(train_loader)
    te_loss, te_acc = eval_epoch(test_loader)
    print(f"epoch {epoch+1:02d}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  test_loss={te_loss:.4f}  test_acc={te_acc:.4f}")
'''

---

## 8) Regularization and Tricks

- **Weight decay**: modest values (e.g., `1e-4`) help CNNs avoid overfitting.
- **Dropout**: less critical on MNIST but useful in MLPs (e.g., `p=0.2–0.5`).
- **Label smoothing**: mild smoothing (`ε=0.05–0.1`) can improve calibration.
- **Early stopping**: stop on validation; MNIST converges quickly.
- **BatchNorm/LayerNorm**: generally not required for small models but can stabilize deeper variants.

---

## 9) Common Pitfalls

- **Overfitting with large models**: MNIST is small; strong models can memorize. Use weight decay and early stopping.
- **Normalization mismatch**: ensure same mean/std in train and test transforms.
- **Inadequate shuffling**: shuffle training data each epoch to avoid learning artifacts.
- **Data leakage**: do not tune on test; carve out a validation subset if needed.

---

## 10) Stronger Baselines and Extensions

- **Deeper CNNs** with 3×3 conv stacks and stride-based downsampling can surpass 99.2–99.4% with careful tuning.
- **EMA of weights**: keep a moving average of parameters for evaluation stability.
- **SWA/SAM**: Stochastic Weight Averaging or Sharpness-Aware Minimization can yield small but consistent gains.
- **Hyperparameter search**: quick grid or Bayesian search on `lr`, weight decay, and affine magnitude.

---

## 11) Plotly Illustration (example)

"""""js_plotly
{
  "data": [
    { "x": ["Linear", "MLP", "LeNet", "Small CNN (3x3)"], "y": [92, 97, 99, 99.3], "type": "bar" }
  ],
  "layout": { "title": "Illustrative Test Accuracy (%) on MNIST" }
}
"""""

---

## 12) Variants and Related Datasets

- **Fashion-MNIST**: same format as MNIST but clothing items; more challenging; good for testing robustness beyond digits.
- **EMNIST**: extended set including letters; multiple splits (Balanced, ByClass, Letters).
- **KMNIST**: Kuzushiji characters; different distribution and difficulty.

These drop in as replacements for MNIST loaders with minor code changes.

---

## 13) Checklist Before Training

- Data downloaded; transforms defined with correct normalization.
- Model chosen (Linear/MLP/CNN) and appropriate optimizer configured.
- Learning-rate schedule set (fixed or cosine/step with optional warmup).
- Logging and checkpointing enabled; seed fixed.
- Evaluation script confirms test metrics and confusion matrix if needed.

---
## 14) Optimizer Comparison on MNIST — 930 Steps

This section visualizes **validation accuracy vs. training step (0–930)** for several optimizers on MNIST with a SimpleCNN.  
Numbers below are illustrative placeholders—replace with your logged metrics to reflect your runs.

"""""js_plotly
{
  "data": [
    {
      "name": "SGD + Momentum",
      "mode": "lines",
      "x": [0,30,60,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,630,660,690,720,750,780,810,840,870,900,930],
      "y": [10.0,25.0,50.0,68.0,80.0,87.0,89.5,90.2,90.9,91.6,92.1,92.5,92.8,93.0,93.2,93.35,93.5,93.6,93.7,93.8,93.9,93.95,94.0,94.02,94.04,94.06,94.08,94.09,94.10,94.12,94.13,94.15]
    },
    {
      "name": "Nesterov Momentum",
      "mode": "lines",
      "x": [0,30,60,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,630,660,690,720,750,780,810,840,870,900,930],
      "y": [10.0,30.0,55.0,72.0,83.0,88.0,90.0,91.0,91.7,92.2,92.6,92.9,93.1,93.3,93.5,93.6,93.7,93.8,93.9,94.0,94.05,94.10,94.15,94.20,94.22,94.25,94.28,94.30,94.32,94.35,94.38,94.40]
    },
    {
      "name": "AdamW",
      "mode": "lines",
      "x": [0,30,60,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,630,660,690,720,750,780,810,840,870,900,930],
      "y": [10.0,35.0,60.0,75.0,85.0,89.0,90.0,90.8,91.5,92.0,92.4,92.7,93.0,93.2,93.4,93.55,93.7,93.8,93.9,94.0,94.05,94.08,94.10,94.12,94.14,94.16,94.18,94.19,94.20,94.22,94.23,94.25]
    },
    {
      "name": "RMSProp",
      "mode": "lines",
      "x": [0,30,60,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,630,660,690,720,750,780,810,840,870,900,930],
      "y": [10.0,28.0,52.0,70.0,82.0,88.0,89.8,90.6,91.3,91.9,92.3,92.6,92.9,93.1,93.3,93.45,93.6,93.7,93.8,93.9,93.98,94.03,94.07,94.10,94.12,94.14,94.16,94.18,94.19,94.20,94.22,94.23]
    }
  ],
  "layout": {
    "title": "MNIST: Validation Accuracy vs. Training Step (0–930)",
    "xaxis": { "title": "Training Step" },
    "yaxis": { "title": "Accuracy (%)", "range": [10, 96] },
    "legend": { "x": 0.02, "y": 0.02, "xanchor": "left", "yanchor": "bottom" }
  }
}
"""""

---

## 15) Summary

MNIST remains a compact, reliable benchmark for building and validating end-to-end training pipelines.  
It is ideal for teaching gradient descent, SGD vs. AdamW, regularization, and basic CNN design.  
While state-of-the-art accuracy is no longer meaningful on MNIST, disciplined experimentation on this dataset translates directly into solid practices for larger and more complex vision tasks.

---
