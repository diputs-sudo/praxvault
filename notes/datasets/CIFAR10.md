# CIFAR-10

**CIFAR-10** is a benchmark image classification dataset of small natural images.  
It contains **60,000 color images** at **32×32** resolution, spread across **10 classes**, with **50,000** training images and **10,000** test images.  
CIFAR-10 is widely used for prototyping architectures, testing augmentation/regularization, and comparing optimization schedules.

---

## 1) Dataset Overview

### 1.1 Composition
- Samples: `N = 60,000` RGB images, `32×32` pixels.
- Split: `50,000` train, `10,000` test (fixed test set).
- Channels: 3 (RGB), 8-bit per channel.
- Task: single-label, 10-way classification.

### 1.2 Classes
- `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.

### 1.3 Format and Access
- Commonly accessed via deep learning libraries (e.g., PyTorch `torchvision.datasets.CIFAR10`).
- Labels are integers `0–9` mapping to the classes above.

---

## 2) Data Preprocessing

### 2.1 Normalization (common statistics)
- Per-channel means: `μ = [0.4914, 0.4822, 0.4465]`
- Per-channel stds:  `σ = [0.2470, 0.2435, 0.2616]`
- Normalize each image `x` as `x' = (x - μ) / σ` channelwise.

### 2.2 Standard Augmentations (baseline)
- **RandomCrop(32, padding=4)**
- **RandomHorizontalFlip(p=0.5)**
- Optional: **Cutout** (e.g., 1–2 holes of size 16), **ColorJitter**.

### 2.3 Stronger Augmentations
- **AutoAugment/RandAugment**, **MixUp**, **CutMix**.
- **RandomErasing** after normalization.
- Note: stronger policies often require slightly longer training or tuned regularization.

---

## 3) Evaluation Protocol

### 3.1 Metric
- Report **Top-1 accuracy** on the 10k test set.
- Optionally, track **Top-1 on a validation split** carved from train (e.g., 45k/5k).

### 3.2 Logging
- Keep curves for: training loss, training accuracy, validation accuracy, learning rate, weight decay, gradient norms.

### 3.3 Reproducibility
- Fix random seeds, enable deterministic backends when possible, log library versions and GPU info.
- Save checkpoints, config files (optimizer, schedule, augmentations), and code commit hashes.

---

## 4) Baseline Models and Recipes

### 4.1 CNN Baselines
- **ResNet-18 (CIFAR variant)**: replace initial conv with 3×3, stride 1; remove first max-pool.
- **WideResNet-28-10**: strong baseline with moderate compute.
- **PreAct ResNet**, **DenseNet**, **ResNeXt**, **VGG** (for historical comparison).

### 4.2 Optimization and Schedules (typical)
- **SGD + Momentum (0.9)**; weight decay `1e-4` (decoupled when supported).
- Learning rate:  
  - ResNet-18: start `η ∈ [0.1, 0.2]` with batch 128; scale with batch size.  
  - WRN-28-10: start `η ∈ [0.1, 0.2]`.
- Schedules: **cosine annealing** with **warmup** (5–10 epochs) or **step decay** at `[60, 120, 160]` epochs for a 200-epoch run.
- **AdamW**: LR `1e-3` baseline, warmup + cosine; often competitive in time-to-accuracy.

### 4.3 Regularization
- Label smoothing (e.g., `ε = 0.1`).
- Strong augmentations (RandAugment, CutMix/MixUp).
- Stochastic depth (for deeper nets), Dropout (model-dependent).

---

## 5) Practical Training Recipes

### 5.1 “Classic” 200-Epoch SGD Recipe (ResNet-18)
- Optimizer: SGD, momentum `0.9`, weight decay `1e-4`.
- LR: `0.1` (batch 128), step decay at epochs `60/120/160` by ×0.2, or cosine with 5-epoch warmup.
- Aug: RandomCrop(32,4) + Flip; optionally Cutout(16).
- Expectation: solid test accuracy with modest compute.

### 5.2 “Modern” 300-Epoch AdamW Recipe (WRN-28-10)
- Optimizer: AdamW, `β1=0.9, β2=0.999`, weight decay `5e-4`.
- LR: `3e-4` to `1e-3`, 5–10 epoch warmup, cosine decay to near zero.
- Aug: RandAugment, MixUp α=0.2, CutMix α=1.0, RandomErasing p=0.25.
- Gradient clipping: optional (global norm 1.0) for stability.

### 5.3 Large-Batch Notes
- Use **linear LR scaling** and **warmup**.  
- Consider **LARS/LAMB** for very large effective batch sizes; increase total training steps.

---

## 6) Data Loading and Transforms (PyTorch)

### 6.1 Basic Pipeline
'''python
import torch
import torchvision as tv
import torchvision.transforms as T

mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std),
])

test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

train_set = tv.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
test_set  = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
'''

### 6.2 RandAugment / CutMix / MixUp Sketch
'''python
# RandAugment from torchvision (>=0.13)
from torchvision.transforms.autoaugment import RandAugment

train_tf_strong = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize(mean, std),
])
# MixUp/CutMix can be applied inside the training loop by mixing images/labels on the fly.
'''

---

## 7) Minimal Training Loop (PyTorch, SGD + Momentum)

'''python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

device = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-style ResNet18: last layer adjusted for 10 classes
model = resnet18(num_classes=10)
# Optional: modify first conv for CIFAR (3x3, stride 1):
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
# Cosine schedule with warmup
total_epochs = 200
warmup_epochs = 5
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)

def adjust_lr(epoch):
    if epoch < warmup_epochs:
        # linear warmup from 0 to base lr
        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * (epoch + 1) / warmup_epochs

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
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

for epoch in range(total_epochs):
    adjust_lr(epoch)
    train_loss, train_acc = train_one_epoch(train_loader)
    val_loss, val_acc = evaluate(test_loader)
    if epoch >= warmup_epochs:
        scheduler.step()
    print(f"epoch {epoch+1:03d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
'''

---

## 8) Strong Augmentations in the Loop (MixUp / CutMix)

### 8.1 MixUp Helper
'''python
import torch

def mixup(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y.float(), torch.ones_like(y, dtype=torch.float)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    indices = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[indices]
    y_onehot = torch.nn.functional.one_hot(y, num_classes=10).float()
    y_mix = lam * y_onehot + (1 - lam) * y_onehot[indices]
    return x_mix, y_mix, torch.tensor(lam)
'''

### 8.2 Loss with Soft Targets
'''python
def soft_ce(pred, target):
    # target is one-hot or soft distribution
    return -(target * torch.log_softmax(pred, dim=1)).sum(dim=1).mean()
'''

---

## 9) Common Pitfalls and Checks

- **Data leakage**: never use the test set for tuning; if you need a validation set, split from train.  
- **Normalization mismatch**: ensure identical mean/std in train and test transforms.  
- **Aug order**: apply geometric/photometric transforms before `ToTensor()` and normalization; Cutout after normalization is fine.  
- **Learning-rate scaling**: adjust LR with batch size; use warmup for large batches.  
- **Weight decay exclusions**: for AdamW, exclude biases and normalization parameters from decay.  
- **Under/over-regularization**: tune label smoothing, weight decay, and augmentation intensity together.

---

## 10) Plotly Illustration (example)

"""""js_plotly
{
  "data": [
    {
      "x": ["Baseline Aug", "RandAugment", "RandAug+MixUp", "RandAug+MixUp+CutMix"],
      "y": [90, 92, 93, 94],
      "type": "bar"
    }
  ],
  "layout": { "title": "Illustrative Test Accuracy (%) Progression" }
}
"""""

---

## 11) Advanced Topics

- **Knowledge Distillation**: train a student network on softened teacher logits to reach higher accuracy with smaller models.  
- **Sharpness-Aware Minimization (SAM)**: encourages flatter minima; pairs with SGD/AdamW.  
- **Exponential Moving Average (EMA)** of weights: maintain a shadow model for evaluation.  
- **Curriculum and Data Curation**: vary difficulty, filter noise; can improve final generalization.  
- **Hyperparameter Search**: small random or Bayesian search on LR, weight decay, and augmentation magnitude is highly cost-effective on CIFAR-10.

---

## 12) Checklist Before Training

- Data downloaded and integrity verified.  
- Transforms defined with correct normalization constants.  
- Model adjusted for CIFAR input (first conv, pooling) if using ImageNet-style backbones.  
- Optimizer, LR schedule, and weight decay configured; warmup set.  
- Augmentation policy chosen (baseline or strong).  
- Logging, checkpoints, and evaluation scripts ready; seeds fixed.

---

## 13) Summary

CIFAR-10 is a compact, standardized playground for supervised image classification.  
Despite its small size, it remains valuable for testing optimizers, schedules, normalization layers, and augmentation strategies.  
Well-tuned recipes with either SGD+Momentum or AdamW, combined with modern augmentations and consistent evaluation protocol, provide strong baselines and reproducible comparisons.

---
