# ImageNet

**ImageNet** is a large-scale image dataset widely used for training and evaluating visual recognition models.  
The most common benchmark subset is **ImageNet-1k (ILSVRC 2012)** with roughly **1.28 million** training images, **50,000** validation images, and **1,000 classes**.  
It has been the pivotal benchmark for modern deep learning in computer vision, driving advances from AlexNet and ResNet to ViT and ConvNeXt.

---

## 1) Dataset Overview

### 1.1 Composition
- **ImageNet-1k (ILSVRC12)**: ~**1,281,167** train, **50,000** val (50 per class), **1,000** categories.
- Images are **RGB**, **variable resolution** (typically resized to 224×224 or higher for training/eval).
- Labels: integer indices `0–999` mapped to WordNet synsets (class names).

### 1.2 Common Splits and Usage
- **Train**: ~1.28M images.
- **Validation**: 50k images (standard for reporting Top-1 and Top-5).
- **Test**: historically server-evaluated; most open-source work reports on the **validation** set.

### 1.3 Access & Storage
- Usually organized as `train/<class_name>/*.JPEG` and `val/<class_name>/*.JPEG`.  
- Many frameworks offer helpers (e.g., PyTorch `ImageFolder`) to index images by directory.

---

## 2) Preprocessing and Normalization

### 2.1 Normalization (canonical)
- Mean `μ = [0.485, 0.456, 0.406]`
- Std  `σ = [0.229, 0.224, 0.225]`
- Normalize channelwise: `x' = (x - μ) / σ` after scaling pixel values to `[0,1]`.

### 2.2 Train-Time Resizing & Cropping
- **RandomResizedCrop(224)** from the original image, with scale range typically `[0.08, 1.0]` and aspect ratio `[3/4, 4/3]`.
- **RandomHorizontalFlip(p=0.5)`**.

### 2.3 Eval-Time Resizing & Cropping
- Resize shorter side to **256** (or 256/0.875≈**292** for 224-center-crop policies), then **CenterCrop(224)**.
- For high-resolution models (e.g., 384), resize and center-crop accordingly.

---

## 3) Metrics and Reporting

- **Top-1 accuracy**: fraction of examples where the predicted class is the ground truth.
- **Top-5 accuracy**: fraction where the ground truth is within the five highest-scoring classes.
- Standard to report both on **validation**, sometimes with single-crop and 10-crop variants (the latter less common today).

---

## 4) Baseline Architectures

- **Convolutional Networks**: AlexNet (historical), VGG, **ResNet-50/101**, ResNeXt, **RegNet**, **EfficientNet**, **ConvNeXt**.
- **Transformers**: **ViT**, **DeiT**, Swin Transformer.
- **Mobile/Edge**: MobileNetV2/V3, ShuffleNetV2, EfficientNet-Lite.
- Typical image size: **224×224** (many modern models also support **256/288/320/384**).

---

## 5) Training Recipes (Modern, Strong Defaults)

- **Optimizer**:  
  - **SGD + Momentum (0.9)**, weight decay **1e-4** (decoupled when supported).  
  - Or **AdamW** with decoupled weight decay **(e.g., 2e-5–1e-4)**.
- **Learning Rate**:  
  - Linear LR scaling with total batch size `B`: `lr = lr_base * (B / 256)`.  
  - Example: `lr_base = 0.1` for SGD (ResNet-50 at B=256).  
  - AdamW baselines often start at `3e-4–1e-3` with warmup.
- **Schedules**:  
  - **Cosine annealing** with **warmup** (e.g., 5–10 epochs).  
  - Step decay (e.g., at epochs 30/60/90 for 100-epoch runs) remains viable, but cosine is now common.
- **Regularization & Stabilizers**:  
  - **Label smoothing** (e.g., `ε = 0.1`).  
  - **MixUp** (α≈0.2) and **CutMix** (α≈1.0).  
  - **RandomErasing** (p≈0.25).  
  - **Stochastic Depth** (for deep nets), **EMA** of weights (optional).
- **Normalization**:  
  - **BatchNorm** for CNNs, **LayerNorm** for Transformers.
- **Precision**:  
  - **AMP (mixed precision)** widely used for speed and memory savings; use loss scaling.

---

## 6) Data Loading & Transforms (PyTorch)

### 6.1 Basic Training/Eval Pipeline
'''python
import torch
import torchvision as tv
import torchvision.transforms as T

# Normalization constants
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std),
])

# Resize shorter side to 256, then center-crop 224
val_tf = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean, std),
])

train_set = tv.datasets.ImageFolder("/path/imagenet/train", transform=train_tf)
val_set   = tv.datasets.ImageFolder("/path/imagenet/val",   transform=val_tf)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=256, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True
)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=256, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True
)
'''

### 6.2 Strong Augmentations (Sketch)
'''python
from torchvision.transforms import autoaugment as A

train_tf_strong = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
    T.RandomHorizontalFlip(),
    A.RandAugment(num_ops=2, magnitude=9),  # or AutoAugment(ImageNetPolicy)
    T.ToTensor(),
    T.Normalize(mean, std),
    T.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
])
# MixUp/CutMix typically applied inside the training loop when forming mini-batches.
'''

---

## 7) Minimal Training Loop (ResNet-50, SGD + Cosine)

'''python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet50(num_classes=1000).to(device)

# Optimizer & schedule
base_lr = 0.1
momentum = 0.9
weight_decay = 1e-4
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

epochs = 100
warmup_epochs = 5
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def adjust_lr(epoch):
    if epoch < warmup_epochs:
        scale = (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * scale

def train_one_epoch(loader):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

for epoch in range(epochs):
    adjust_lr(epoch)
    tr_loss, tr_acc = train_one_epoch(train_loader)
    va_loss, va_acc = evaluate(val_loader)
    if epoch >= warmup_epochs:
        scheduler.step()
    print(f"epoch {epoch+1:03d}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}")
'''

---

## 8) Distributed & Large-Batch Training

- **DDP (DistributedDataParallel)** with multiple GPUs/nodes is standard for throughput.  
- **Linear LR scaling** with global batch size; always use **warmup**.  
- **Gradient accumulation** to emulate larger batches under memory constraints.  
- Ensure **DistributedSampler** in PyTorch to avoid sample duplication per epoch.

---

## 9) Advanced Regularization & Training Tricks

- **MixUp/CutMix**: algebraic mixing of inputs/labels; improves robustness and calibration.  
- **Label smoothing**: mitigates overconfidence; improves top-1/top-5 marginally.  
- **EMA** of weights: track a shadow model via exponential moving average for evaluation stability.  
- **Stochastic Depth / DropPath**: skip residual paths randomly in deep networks.  
- **SWA / SAM**: Stochastic Weight Averaging or Sharpness-Aware Minimization for flatter minima.

---

## 10) Transformers on ImageNet

- **ViT/DeiT/Swins** typically use **AdamW**, **LayerNorm**, strong aug (RandAugment), and **longer warmup** (e.g., 5–10 epochs).  
- Resolution scaling (224→384) often yields accuracy gains with higher compute.  
- Keep **weight decay exclusions** (biases and norms) in AdamW.

---

## 11) Inference and Evaluation Details

- **Single-crop 224** is standard for comparison.  
- Multi-crop or **test-time augmentation** can squeeze extra accuracy but increases latency.  
- **Throughput/latency** trade-offs: prefer channels-last, AMP, fused ops; profile with representative batch sizes.

---

## 12) Common Pitfalls

- **Normalization mismatch**: use identical mean/std for train and eval.  
- **Resize policy mismatch**: ensure eval uses the agreed 256→224 center-crop (or declared alternative).  
- **Data loading bottlenecks**: raise `num_workers`, enable `persistent_workers`, use JPEG decoders with parallel I/O.  
- **Over-regularization**: strong RandAug + heavy MixUp/CutMix may require longer training or tuned LR.  
- **Weight decay exclusions**: when using AdamW, exclude biases and normalization scales from decay.

---

## 13) Plotly Illustration (example)

The bar chart below (illustrative numbers) compares **Top-1 accuracy** for representative families at 224×224 single-crop under competitive recipes.

"""""js_plotly
{
  "data": [
    { "name": "ResNet-50 (SGD)",   "type": "bar", "x": ["Top-1"], "y": [76.5] },
    { "name": "RegNetY-8GF (SGD)", "type": "bar", "x": ["Top-1"], "y": [80.0] },
    { "name": "EfficientNet-B0",   "type": "bar", "x": ["Top-1"], "y": [77.5] },
    { "name": "ConvNeXt-T (AdamW)","type": "bar", "x": ["Top-1"], "y": [82.0] },
    { "name": "DeiT-S (AdamW)",    "type": "bar", "x": ["Top-1"], "y": [79.8] },
    { "name": "ViT-B/16 (AdamW)",  "type": "bar", "x": ["Top-1"], "y": [81.0] }
  ],
  "layout": { "title": "Illustrative Single-Crop Top-1 @ 224×224 (not SOTA)" }
}
"""""

---

## 14) Practical Recipes (Concise)

- **ResNet-50 (CNN)**: SGD+Momentum (0.9), wd=1e-4, bs=256, lr=0.1 (linear scale), cosine w/ 5–10 epoch warmup, label smoothing 0.1, RandAugment light, MixUp 0.2, CutMix 1.0, RandomErasing 0.25.  
- **ConvNeXt/RegNet**: SGD or AdamW; favor cosine + warmup; use stronger aug and sometimes EMA.  
- **DeiT/ViT (Transformer)**: AdamW (`lr≈3e-4` at bs=1024 equivalent), wd=0.05–0.1, 5–10 epoch warmup, cosine, strong RandAugment, MixUp/CutMix, stochastic depth.

---

## 15) Checklist Before Training

- Directory structure verified (`train/`, `val/` per class).  
- Transforms and normalization consistent; eval policy fixed (e.g., 256→224 center-crop).  
- Optimizer, LR, weight decay, schedule, warmup configured; batch size and LR scaling set.  
- AMP/mixed precision enabled; gradient clipping for unstable models.  
- DDP set up (if multi-GPU): samplers, seeds, deterministic flags as needed.  
- Logging (loss, lr, top-1/top-5), checkpointing, and resume logic in place.

---

## 16) Summary

ImageNet remains the central benchmark for supervised image recognition, enabling rigorous comparisons of architectures, augmentations, and optimization strategies.  
Strong modern recipes pair **cosine schedules + warmup** with **label smoothing**, **MixUp/CutMix**, and either **SGD+Momentum** or **AdamW**, using **AMP** and distributed training for efficiency.  
Although downstream transfer and larger datasets are increasingly important, mastering ImageNet training continues to provide durable intuition and reliable baselines for vision research and deployment.

---
