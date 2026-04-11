"""
Non-Invasive Prenatal Cancer Detection — ResNet-50 Training Script
===================================================================
Run with:
    python train.py

Dataset folder structure expected:
    train/benign/      train/malignant/      train/normal/
    validation/benign/ validation/malignant/ validation/normal/
    test/benign/       test/malignant/       test/normal/

Class Imbalance is handled via:
  1. WeightedRandomSampler  — over-samples minority classes per batch
  2. Weighted CrossEntropyLoss — penalises errors on rare classes more
  3. Label smoothing (0.1)  — prevents over-confident predictions
  4. Heavy augmentation     — improves generalisation
  5. OneCycleLR scheduler   — fast convergence
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import time

# ═══════════════════════════════════════════════════════════════════
#  CONFIG  — edit these paths if needed
# ═══════════════════════════════════════════════════════════════════
TRAIN_DIR   = "./train"
VAL_DIR     = "./validation"
TEST_DIR    = "./test"
SAVE_PATH   = "./Resnet_fineTuning.pth"
NUM_CLASSES = 3
BATCH_SIZE  = 32
EPOCHS      = 40
LR          = 3e-4
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════════
#  TRANSFORMS
# ═══════════════════════════════════════════════════════════════════
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ═══════════════════════════════════════════════════════════════════
#  DATASETS
# ═══════════════════════════════════════════════════════════════════
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
test_ds  = datasets.ImageFolder(TEST_DIR,  transform=val_tf)

CLASS_NAMES = train_ds.classes
print(f"\nClasses       : {CLASS_NAMES}")
print(f"class_to_idx  : {train_ds.class_to_idx}")

# ─── Count samples per class ────────────────────────────────────────
labels_train  = [s[1] for s in train_ds.samples]
class_counts  = Counter(labels_train)
print(f"\nClass distribution (train): {dict(class_counts)}")

# ═══════════════════════════════════════════════════════════════════
#  CLASS IMBALANCE HANDLING
# ═══════════════════════════════════════════════════════════════════

# 1. WeightedRandomSampler — inverse-frequency sampling
inv_freq   = {cls: 1.0 / cnt for cls, cnt in class_counts.items()}
sample_wts = [inv_freq[l] for l in labels_train]
sampler    = WeightedRandomSampler(
    weights=sample_wts,
    num_samples=len(sample_wts),
    replacement=True,
)

# 2. Loss weights - REMOVED!
# Do not use class weights in CrossEntropyLoss when using WeightedRandomSampler!
# Doing so double-compensates for imbalance and destroys accuracy.

# ─── DataLoaders ────────────────────────────────────────────────────
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

# ═══════════════════════════════════════════════════════════════════
#  MODEL — ResNet-50 with selective unfreezing
# ═══════════════════════════════════════════════════════════════════
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze backbone; unfreeze layer3, layer4, fc only
for name, param in model.named_parameters():
    param.requires_grad = any(k in name for k in ("layer3", "layer4", "fc"))

model.fc = nn.Sequential(
    nn.BatchNorm1d(model.fc.in_features),
    nn.Dropout(0.45),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(512, NUM_CLASSES),
)
model = model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {trainable:,}")

# ═══════════════════════════════════════════════════════════════════
#  TRAINING SETUP
# ═══════════════════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4,
)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS, pct_start=0.1,
)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}


# ═══════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    with torch.set_grad_enabled(train):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, lbls)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == lbls).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


best_val_acc = 0.0
patience, no_improve = 7, 0

print("\n" + "═" * 65)
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader,   train=False)
    elapsed = time.time() - t0

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)

    flag = ""
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        no_improve   = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model": model,  # Save the full model architecture to avoid mismatch
            "class_to_idx": train_ds.class_to_idx,
            "class_names": CLASS_NAMES,
            "best_val_acc": best_val_acc,
            "history": history,
        }, SAVE_PATH)
        flag = "  ✓ SAVED"
    else:
        no_improve += 1

    print(f"[{epoch:02d}/{EPOCHS}] {elapsed:.1f}s | "
          f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
          f"Val {va_loss:.4f}/{va_acc:.4f}{flag}")

    if no_improve >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

# ═══════════════════════════════════════════════════════════════════
#  TEST EVALUATION
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("Loading best checkpoint for test evaluation…")
ckpt = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbls.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix — Test Set")
plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Confusion matrix saved → confusion_matrix.png")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"],   label="Val")
axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(history["train_acc"], label="Train")
axes[1].plot(history["val_acc"],   label="Val")
axes[1].set_title("Accuracy"); axes[1].legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("Training curves saved → training_curves.png")

print(f"\nBest Val Accuracy : {best_val_acc:.4f}")
print("Done ✓")
