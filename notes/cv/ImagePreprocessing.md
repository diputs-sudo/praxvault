# Image Preprocessing

**Image preprocessing** converts raw images into consistent, model-ready tensors.  
It standardizes geometry, color/illumination, and dynamic range; reduces noise; preserves salient content; and ensures reproducible training/evaluation.  
Preprocessing differs from **data augmentation**: preprocessing aims for deterministic normalization and cleanliness, while augmentation introduces controlled randomness to improve generalization.

---

## 1) Goals and Principles

- **Consistency:** uniform size, color space, and numeric ranges across the dataset.
- **Signal preservation:** denoise and normalize without destroying discriminative content.
- **Reproducibility:** deterministic transforms for evaluation; logged parameters and seeds.
- **Separation of concerns:** preprocessing (deterministic) vs. augmentation (stochastic).
- **No leakage:** compute dataset statistics on the training split only.

---

## 2) Geometry: Size, Aspect Ratio, and Framing

### 2.1 Resizing Strategies
- **Isotropic resize + crop:** resize shortest side to `S` then center/RandomCrop to `H×W`.
- **Direct resize:** scale to `H×W` ignoring aspect ratio (simple but may distort).
- **Letterboxing (pad to aspect):** pad with a value (0/mean/reflect) to target aspect, then resize.
- **Multi-scale (for detection):** randomly vary `S` within a range during training; fix at eval.

### 2.2 Cropping and Padding
- **Center/Random crop:** classification standard (e.g., 224×224 from larger side).
- **Five/Ten-crop (eval):** center + corners (and flips) for robust evaluation.
- **Reflect/replicate/constant pad:** choose padding consistent with normalization.

### 2.3 Alignment for Structured Tasks
- **Detection/Segmentation/Keypoints:** apply identical geometric transforms to images and labels (boxes/masks/points); preserve alignment and interpolation modes (nearest for masks).

---

## 3) Color and Pixel Value Domains

### 3.1 Color Spaces
- **RGB** (default for deep learning).  
- **BGR** (OpenCV default)—convert to RGB if mixing libraries.  
- **Grayscale** for single-channel tasks; stack to 3 channels if model expects RGB.  
- **YCbCr/HSV/Lab** for illumination/color manipulation; convert back to RGB before training if model expects RGB.

### 3.2 Dynamic Range and Dtypes
- **8-bit:** `[0, 255]` `uint8` → cast to `float32` then scale to `[0,1]`.  
- **Floating point tensors:** `[0,1]` or standardized `N(0,1)` after normalization.  
- **16-bit/HDR/RAW:** linearize with camera response or demosaicing; tone-map carefully.

### 3.3 Gamma and Linearization
- Most sRGB images are gamma-compressed (~2.2). For physically meaningful ops (blending, convolution), consider converting to **linear** space: `x_linear = x_srgb^γ` with `γ ≈ 2.2`; then re-encode to sRGB after processing if needed.

---

## 4) Normalization and Standardization

### 4.1 Per-Channel Standardization
- Compute **training-set** mean `μ_c` and std `σ_c` per channel; normalize:  
  `x′ = (x - μ) / σ`.
- Benefits: centers data, stabilizes optimization.

### 4.2 Common Reference Statistics (when computing your own is impractical)
- **ImageNet (RGB):** `μ = [0.485, 0.456, 0.406]`, `σ = [0.229, 0.224, 0.225]`.  
- **CIFAR-10 (RGB):** `μ = [0.4914, 0.4822, 0.4465]`, `σ = [0.2470, 0.2435, 0.2616]`.  
Use dataset-specific stats when possible for best results.

### 4.3 Alternative Normalizations
- **Min–max scaling** to `[0,1]` or `[-1,1]`.  
- **Unit-norm per image** (rare for vision classification).  
- **Channel-wise zero-centering only** when `σ` is unstable.

---

## 5) Illumination and Contrast

### 5.1 Histogram Equalization
- Spreads intensities to use full dynamic range; can over-amplify noise.

### 5.2 CLAHE (Contrast Limited Adaptive Hist. Equalization)
- Local equalization with clip limit; robust for varied lighting; safer than global HE.

### 5.3 Gamma Correction
- `x′ = x^γ` (on `[0,1]`); `γ < 1` brightens, `γ > 1` darkens; apply before normalization.

---

## 6) Denoising and Sharpening

### 6.1 Denoising
- **Gaussian blur:** removes Gaussian noise; may blur edges.  
- **Median filter:** robust to salt-and-pepper noise; preserves edges.  
- **Bilateral filter:** edge-preserving smoothing (uses spatial and intensity distance).  
- **Non-Local Means:** stronger denoising; heavier compute.

### 6.2 Sharpening
- **Unsharp masking:** enhance edges by adding high-frequency residues.  
- Use lightly; aggressive sharpening harms generalization.

---

## 7) Compression Artifacts and Color Management

- **JPEG artifacts:** blockiness and ringing; mild blur or deblocking can help.  
- **PNG/TIFF:** lossless; heavier I/O.  
- **Color profiles (ICC):** convert to standard sRGB for consistency.  
- **Bit depth:** preserve 16-bit pipelines for medical/HDR tasks until just before batching.

---

## 8) Preprocessing vs. Augmentation

- **Preprocessing (deterministic, applied to train/val/test):** resize, color conversion, normalization.  
- **Augmentation (stochastic, train only):** flips, random crops, color jitter, MixUp/CutMix, RandAugment, geometric perturbations.
- Keep augmentation intensity aligned with task difficulty and model capacity.

---

## 9) Ordering of Operations (Recommended)

1. **Decode** (read file → array, handle color profile).  
2. **Color convert** to RGB (if needed).  
3. **Resize/Pad/Crop** (geometry unification).  
4. **Illumination/contrast** (gamma/CLAHE) if required by domain.  
5. **Denoise** (light touch if at all).  
6. **ToTensor** and **scale to [0,1]** (float32).  
7. **Normalize** with dataset `μ, σ`.  
8. **Augment** (train only) before normalization if augment needs raw intensity; after if operating on standardized tensors—be consistent.  
9. **Batch** and move to device.

---

## 10) Computing Dataset Statistics (Train Split Only)

'''python
# Compute per-channel mean and std over the TRAIN set (PyTorch)
import torch
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T

# 1) Build a temporary pipeline WITHOUT normalization
base_tf = T.Compose([T.Resize((224,224)), T.ToTensor()])  # adapt size as needed
train = tv.datasets.ImageFolder("path/to/train", transform=base_tf)
loader = DataLoader(train, batch_size=64, shuffle=False, num_workers=4)

n = 0
mean = 0.0
M2 = 0.0  # for streaming variance (per-channel)
for x, _ in loader:
    # x: [B, C, H, W] in [0,1]
    B = x.size(0)
    n += B * x.size(2) * x.size(3)
    # reshape to [B, C, H*W]
    x_ = x.view(x.size(0), x.size(1), -1)
    mean_batch = x_.mean(dim=(0,2))
    var_batch  = x_.var(dim=(0,2), unbiased=False)
    # combine means/vars across batches (channelwise)
    if isinstance(mean, float):
        mean = mean_batch
        M2 = var_batch
    else:
        # parallel variance merge
        delta = mean_batch - mean
        mean = mean + delta * (B * x.size(2) * x.size(3) / n)
        M2 = M2 + var_batch  # approximation acceptable for large n

std = (M2.clamp(min=1e-12)).sqrt()
print("mean=", mean.tolist(), "std=", std.tolist())
'''

---

## 11) PyTorch Pipelines

### 11.1 Classification (Deterministic Eval)
'''python
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

eval_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
'''

### 11.2 Classification (Training with Augmentation)
'''python
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
'''

### 11.3 Segmentation (Preserve Masks)
'''python
from torchvision.transforms import functional as F
import numpy as np
import random, torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class SegPairTransform:
    def __init__(self, size=(512,512)):
        self.size = size
    def __call__(self, img, mask):
        # same random crop/flip to both
        if random.random() < 0.5:
            img  = F.hflip(img)
            mask = F.hflip(mask)
        img  = F.resize(img,  self.size, interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        img  = F.to_tensor(img)
        img  = F.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        # mask to Long tensor of class ids
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask
'''

---

## 12) OpenCV Examples (I/O, Color, Denoise)

'''python
import cv2
import numpy as np

img_bgr = cv2.imread("img.jpg", cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Resize with letterbox to 224x224
h, w = img_rgb.shape[:2]
scale = 224 / max(h, w)
nh, nw = int(h * scale), int(w * scale)
resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
canvas = np.full((224,224,3), 128, dtype=np.uint8)
y0 = (224 - nh)//2; x0 = (224 - nw)//2
canvas[y0:y0+nh, x0:x0+nw] = resized

# CLAHE on L channel in Lab
lab = cv2.cvtColor(canvas, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l2 = clahe.apply(l)
lab2 = cv2.merge([l2, a, b])
enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

# Bilateral filter (edge-preserving denoise)
smoothed = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)
'''

---

## 13) Domain-Specific Notes

- **Medical Imaging (DICOM, 16-bit):** preserve bit depth; window/level mapping; z-score within body region; avoid lossy compression.  
- **Remote Sensing:** radiometric calibration; pan-sharpening; georeferencing; multi-spectral normalization per band.  
- **OCR/Document:** binarization (Otsu), morphology; deskew; illumination correction; preserve resolution.  
- **Low-Light:** denoise, deblur; raw-to-RGB pipelines; noise-aware training.

---

## 14) Performance and Engineering

- **Prefetch and parallel loaders** (`num_workers`, pinned memory).  
- **Cache decoded/augmented samples** when feasible.  
- **Mixed precision** transforms on GPU when supported (e.g., Kornia).  
- **Avoid redundant conversions** (e.g., BGR↔RGB ping-pong).  
- **Profile I/O vs. compute**; increase batch size only if input pipeline keeps up.

---

## 15) Quality Control and Diagnostics

- Visualize random batches after preprocessing to confirm: shape, color space, normalization, mask alignment.  
- Check histograms before/after normalization and CLAHE/gamma.  
- Verify per-split statistics; ensure test-time pipeline is deterministic and identical in geometry/normalization to validation.

---

## 16) Common Pitfalls

- **Leaking statistics:** computing `μ, σ` on full dataset including test.  
- **Mask interpolation errors:** using bilinear on label masks—use nearest neighbor.  
- **Order mistakes:** augment after normalization unintentionally changing distribution; or normalizing twice.  
- **Color confusion:** mixing RGB and BGR between libraries.  
- **Distortion:** direct resize that breaks aspect ratio when aspect matters (faces, OCR).

---

## 17) Minimal End-to-End Template (Classification)

'''python
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

train_tf = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std),
])

val_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean, std),
])

train = tv.datasets.ImageFolder("data/train", transform=train_tf)
val   = tv.datasets.ImageFolder("data/val",   transform=val_tf)
train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val,   batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
'''

---

## 18) Summary

Image preprocessing standardizes input geometry, color space, and numerical scaling; optionally enhances contrast and reduces noise while preserving task-relevant structure.  
A disciplined pipeline—deterministic for evaluation, stochastic only for training augmentation—prevents data leakage, stabilizes optimization, and improves reproducibility.  
Careful ordering, correct interpolation, dataset-specific statistics, and alignment across images and labels are the core of robust, high-performing vision systems.

---
