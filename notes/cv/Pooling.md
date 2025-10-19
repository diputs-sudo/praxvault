# Pooling

**Pooling** reduces spatial (and sometimes temporal) resolution by aggregating local neighborhoods.  
It provides limited **translation invariance**, reduces memory/compute, increases receptive field, and can control overfitting.  
Common forms include **Max Pooling**, **Average Pooling**, **Global Pooling**, and **Adaptive Pooling**; modern variants add **anti-aliasing**, **stochastic**, **attention-based**, and **region**-aware pooling.

---

## 1) Motivation and Effects

- **Dimensionality reduction:** downsample `H×W` feature maps to smaller grids to lower FLOPs and memory.  
- **Invariance vs. equivariance:** pooling aggregates over local shifts, yielding limited **translation invariance** (vs. convolutions’ translation **equivariance**).  
- **Receptive field growth:** each pooling step increases the effective receptive field.  
- **Regularization:** by discarding fine details, pooling can reduce overfitting.

---

## 2) Definitions and Notation

Let input feature map `X ∈ ℝ^{C×H×W}`. For each channel `c`, pooling applies a window `K_h×K_w` with stride `S` (and optional padding `P`) to produce `Y`.

### 2.1 Max Pooling
- Elementwise maximum over the window:  
  `Y[c, i, j] = max_{(u,v) ∈ window(i,j)} X[c, u, v]`  
- Captures the strongest activation; robust to small spatial shifts.

### 2.2 Average Pooling
- Mean over the window:  
  `Y[c, i, j] = (1/|W|) Σ_{(u,v) ∈ window(i,j)} X[c, u, v]`  
- Smooths features; reduces noise; may blur boundaries.

### 2.3 Global Pooling
- **Global Average Pool (GAP):** `Y[c] = (1/(H·W)) Σ_{u,v} X[c,u,v]`  
- **Global Max Pool (GMP):** `Y[c] = max_{u,v} X[c,u,v]`  
- Often used to replace large fully connected layers; parameter-free.

### 2.4 Adaptive Pooling
- Specify **output size** (e.g., `7×7` or `1×1`); framework computes kernel/stride to reach it—handy for variable-sized inputs.

---

## 3) Implementation Equivalences

- **Average Pool ≡ Depthwise Conv** with a uniform kernel and stride `S` (without learnable params).  
- **Max Pool ≡ L_p Pool with p→∞**.  
- **Strided Convolution as pooling:** learnable low-pass + downsample; widely used in modern CNNs instead of explicit pooling.

---

## 4) Design Choices

### 4.1 Kernel, Stride, Padding
- **Kernel** (`K=2,3` most common). Larger K increases invariance but may erase details.  
- **Stride** (`S=2` halves resolution).  
- **Padding** matters near borders; for average, choose whether to **count/include padding**.

### 4.2 Where to Pool
- Early layers: aggressive pooling risks information loss.  
- Mid/late layers: more semantic features—pooling is safer.  
- Many modern nets reduce use of max-pool, preferring **stride-2 conv** blocks.

### 4.3 Channels
- Pooling is typically **per-channel** (no cross-channel mixing). Cross-channel reduction is handled by convolutions or attention.

---

## 5) Variants and Extensions

### 5.1 L_p Pooling
- `Y = ( (1/|W|) Σ |X|^p )^{1/p}`.  
- `p=1` average, `p→∞` max; interpolate behavior by tuning `p`.

### 5.2 Stochastic Pooling
- Sample from the window with probability proportional to activations.  
- Adds regularization; less common in modern practice.

### 5.3 Anti-Aliasing Pooling
- Apply **blur/low-pass** before downsampling (e.g., fixed Gaussian).  
- Reduces aliasing artifacts and sensitivity to small translations.

### 5.4 Spatial Pyramid Pooling (SPP)
- Multi-level pooling to fixed bins (e.g., `1×1, 2×2, 4×4`) then concatenate.  
- Yields fixed-length descriptors from variable-sized inputs.

### 5.5 ROI Pooling / ROI Align (Detection)
- Pool features from arbitrary bounding boxes to fixed size.  
- **ROI Align** uses bilinear interpolation (no quantization) for higher accuracy.

### 5.6 Attention/Token Pooling
- Replace pooling with **attention-based** reduction (e.g., CLS token, pooling by learned queries).  
- Common in Transformers; trades fixed aggregation for learned, context-aware summarization.

---

## 6) Global Pooling vs. Fully Connected Heads

- **GAP** dramatically reduces parameters vs. FC layers.  
- Encourages localization behavior (Class Activation Maps).  
- Cons: loses spatial layout; mitigated by attention or multi-head pooling (e.g., GeM/Lp or multi-bin GAP).

---

## 7) Practical Recipes

- **Classic CNN blocks:** `Conv → BN → ReLU → MaxPool(2×2, S=2)` early; later switch to stride-2 convs.  
- **Modern vision (ResNet-like):** prefer **stride-2 conv** in downsampling blocks; **GAP** before classifier.  
- **Robustness to shifts:** consider **anti-aliased** pooling or blur+downsample.  
- **Low data regimes:** max-pool can act as strong regularizer; average pool smoother but may underfit edges.  
- **Variable input sizes:** **adaptive pooling** or SPP to fixed features.

---

## 8) Pitfalls and Diagnostics

- **Over-aggressive downsampling:** accuracy drop from lost detail → reduce stride or delay pooling.  
- **Aliasing artifacts:** use anti-aliased pooling or blur.  
- **Padding bias:** average pooling with zero-padding reduces border values; consider “count include/exclude pad” setting.  
- **Channel reduction confusion:** pooling is spatial; for channel squeeze use `1×1` conv or attention.

---

## 9) PyTorch Examples

### 9.1 Basic Layers
'''python
import torch
import torch.nn as nn

x = torch.randn(8, 64, 32, 32)  # B,C,H,W

maxpool = nn.MaxPool2d(kernel_size=2, stride=2)      # 32->16
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
gap     = nn.AdaptiveAvgPool2d((1,1))                # H,W -> 1,1
gmp     = nn.AdaptiveMaxPool2d((1,1))
adap7   = nn.AdaptiveAvgPool2d((7,7))                # resize features to 7x7

y_max = maxpool(x)     # [8,64,16,16]
y_avg = avgpool(x)     # [8,64,16,16]
y_gap = gap(x).squeeze(-1).squeeze(-1)  # [8,64]
'''

### 9.2 Anti-Aliased Average Pool (blur pool sketch)
'''python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurPool(nn.Module):
    def __init__(self, channels, stride=2, kernel=[1,2,1]):
        super().__init__()
        filt = torch.tensor(kernel, dtype=torch.float32)
        filt = (filt[:,None] * filt[None,:])
        filt = filt / filt.sum()
        self.register_buffer("filt", filt[None, None, :, :].repeat(channels, 1, 1, 1))
        self.stride = stride
        self.groups = channels
        self.pad = (len(kernel)//2,)*4  # symmetric
    def forward(self, x):
        x = F.pad(x, self.pad, mode="reflect")
        return F.conv2d(x, self.filt, stride=self.stride, groups=self.groups)

# usage
x = torch.randn(8, 64, 32, 32)
bp = BlurPool(64, stride=2)
y = bp(x)  # [8,64,16,16]
'''

### 9.3 Replace MaxPool with Strided Conv
'''python
import torch.nn as nn

block = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # downsample
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
)
'''

### 9.4 Adaptive Pool Head (GAP → Classifier)
'''python
import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_ch, num_classes)
    def forward(self, x):
        x = self.pool(x).flatten(1)  # B,C,1,1 -> B,C
        return self.fc(x)

# Example: last feature map = [B, 512, 7, 7]
'''

---

## 10) Worked Example: Mini CNN with Pooling Variants

### 10.1 MaxPool Model
'''python
import torch
import torch.nn as nn

class MiniCNN_MaxPool(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 32->16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 16->8
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.head(self.net(x))
'''

### 10.2 Strided-Conv Model (no explicit pooling)
'''python
import torch
import torch.nn as nn

class MiniCNN_Strided(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 32->16
            nn.Conv2d(64,128,3, stride=2, padding=1), nn.ReLU(inplace=True),   # 16->8
            nn.Conv2d(128,256,3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.head(self.net(x))
'''

---

## 11) Plotly Illustration (example)

"""""js_plotly
{
  "data": [
    { "x": ["MaxPool", "AvgPool", "Anti-aliased Avg", "Strided Conv"], "y": [90, 89, 91, 90], "type": "bar" }
  ],
  "layout": { "title": "Illustrative Accuracy (%) with Different Downsampling" }
}
"""""

---

## 12) Special Topics

- **GeM (Generalized Mean Pooling):** learnable L_p pooling (commonly in retrieval).  
- **Channel-wise pooling in attention:** e.g., SE blocks use global pooling to compute channel gates.  
- **Temporal pooling (videos):** 1D pooling over time or attention pooling.  
- **Graph pooling:** cluster/Top-K pooling; coarsen graphs analogous to spatial pooling.

---

## 13) Checklist

- Choose **downsampling strategy**: max/avg vs. stride-2 conv; consider anti-aliasing.  
- Set kernel/stride to match target resolution schedule.  
- Prefer **GAP** over large FC layers for compact heads.  
- For variable input sizes, use **adaptive pooling** or **SPP**.  
- Validate that early pooling is not over-aggressive; visualize feature maps.  
- For detection/segmentation, ensure region-aware pooling (ROI Align) to keep alignment.

---

## 14) Summary

Pooling aggregates local neighborhoods to reduce resolution, increase invariance, and control compute.  
Max and average pooling remain useful, but many modern CNNs rely on strided convolutions (often with blur) and global average pooling at the head.  
Selecting the right downsampling schedule—and when necessary, anti-aliased or attention-based pooling—balances detail preservation with robustness and efficiency.

---
