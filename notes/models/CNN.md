# Convolutional Neural Networks (CNN)

Convolutional Neural Networks learn hierarchical feature extractors with **convolutions**, **nonlinearities**, and **pooling**, enabling strong performance on images, audio spectrograms, and other grids.

---

## 1) Building Blocks

- **Convolution (2D)**  
  Sliding filters `K ∈ ℝ^{C_out×C_in×kH×kW}` over input `X ∈ ℝ^{N×C_in×H×W}`.  
  Output shape (with stride `s`, padding `p`, dilation `d`):  
  `H_out = ⌊(H + 2p − d·(kH−1) − 1)/s⌋ + 1`, same for width.

- **Nonlinearity**  
  ReLU/LeakyReLU/GELU: `y = max(0, x)` etc.

- **Pooling**  
  Max/avg reduces spatial size; adds invariance, lowers compute.

- **Normalization**  
  BatchNorm/LayerNorm stabilize and speed training.

- **Regularization**  
  Weight decay, dropout, data augmentation (flip, crop, color jitter, cutout, mixup/cutmix).

---

## 2) From-Scratch Convolution (reference)

'''python
import numpy as np

def conv2d_naive(x, w, b=None, stride=1, padding=0):
    """
    x: (N, C_in, H, W)
    w: (C_out, C_in, kH, kW)
    b: (C_out,) or None
    returns: (N, C_out, H_out, W_out)
    """
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    s = stride
    p = padding

    # pad
    x_pad = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode="constant")
    H_out = (H + 2*p - kH)//s + 1
    W_out = (W + 2*p - kW)//s + 1

    y = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    hs, ws = i*s, j*s
                    patch = x_pad[n, :, hs:hs+kH, ws:ws+kW]  # (C_in,kH,kW)
                    y[n, co, i, j] = np.sum(patch * w[co])
            if b is not None:
                y[n, co] += b[co]
    return y
'''

---

## 3) Minimal CNN in PyTorch (classification)

'''python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2)  # halves H,W
        self.head  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),  # adjust if input size differs
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # /2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # /4
        x = self.head(x)
        return x

# Example input: CIFAR-10 (N,3,32,32) -> after two pools -> (N,64,8,8)
'''

---

## 4) Training Loop (PyTorch)

'''python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data pipeline with standard augmentation for small images
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
test_tf = transforms.Compose([transforms.ToTensor()])

train_ds = datasets.CIFAR10(root="./data", train=True, transform=train_tf, download=True)
test_ds  = datasets.CIFAR10(root="./data", train=False, transform=test_tf, download=True)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

model = SmallCNN(in_ch=3, num_classes=10).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
criterion = nn.CrossEntropyLoss()

def eval_loop(dl):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum/total, correct/total

for epoch in range(100):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
    sched.step()
    val_loss, val_acc = eval_loop(test_dl)
    print(f"epoch {epoch+1:03d}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
'''

---

## 5) Deeper Block: Conv → Norm → Activation → Pool

'''python
import torch.nn as nn

def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class DeeperCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = conv_block(3, 64)
        self.stage1 = nn.Sequential(conv_block(64, 64), nn.MaxPool2d(2))  # /2
        self.stage2 = nn.Sequential(conv_block(64, 128), nn.MaxPool2d(2)) # /4
        self.stage3 = nn.Sequential(conv_block(128, 256), nn.MaxPool2d(2))# /8
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x
'''

---

## 6) Data Augmentation Recipes

'''python
from torchvision import transforms

aug_strong = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor()
])
'''

---

## 7) Transfer Learning (ImageNet backbone → new task)

'''python
import torch
import torch.nn as nn
from torchvision import models

def build_transfer_model(num_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False  # freeze backbone
    # replace classifier
    in_feat = m.fc.in_features
    m.fc = nn.Linear(in_feat, num_classes)
    return m

# Then fine-tune head (and optionally unfreeze later with smaller LR).
'''

---

## 8) Keras Alternative (compact)

'''python
import tensorflow as tf
from tensorflow.keras import layers, models

def keras_cnn(input_shape=(32,32,3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# model = keras_cnn()
# model.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_val, y_val))
'''

---

## 9) Evaluation & Inference

'''python
import torch
import numpy as np

def predict_logits(dl, model, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            ys.append(y.numpy())
            ps.append(logits.cpu().numpy())
    y_true = np.concatenate(ys)
    y_logits = np.concatenate(ps)
    y_pred = y_logits.argmax(1)
    acc = (y_pred == y_true).mean()
    return acc, y_pred, y_logits
'''

---

## 10) Troubleshooting

- Training loss flat → raise learning rate briefly, verify normalization/augmentations, check label order.  
- Overfitting → stronger augmentation, dropout, weight decay, early stopping, MixUp/CutMix.  
- Vanishing/Exploding → use BatchNorm, ResNet-style skip connections, careful init.  
- GPU OOM → lower batch size, use `torch.cuda.amp.autocast()` for mixed precision.

---

## 11) Mixed Precision (speed + memory)

'''python
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

for epoch in range(50):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
'''

---

## 12) Receptive Field & Downsampling (illustrative)

"""""js_plotly
{
  "data": [
    {"x": [1,2,3,4], "y": [3,7,15,31], "mode": "lines+markers", "name": "Receptive Field"},
    {"x": [1,2,3,4], "y": [32,16,8,4], "mode": "lines+markers", "name": "Feature Map Size"}
  ],
  "layout": { "title": "Receptive Field vs. Spatial Resolution (Example)", "xaxis": {"title": "Conv/Pool Stage"}, "yaxis": {"title": "Size"} }
}
"""""

---

## 13) Summary

Stacks of **Conv → Norm → Activation → Pool** build feature hierarchies.  
Train with SGD/momentum + cosine schedule, regularize with augmentation and weight decay, and consider transfer learning for small datasets.  
Use mixed precision for efficiency and monitor validation accuracy to guide architecture/schedule tweaks.
