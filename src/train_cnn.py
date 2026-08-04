import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from collections import Counter
from pathlib import Path

from dataset import HistoDataset, CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_ROOT = Path("data/processed")
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4

train_ds = HistoDataset(DATA_ROOT / "train", train=True)
val_ds = HistoDataset(DATA_ROOT / "val", train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# --- Handle class imbalance (Malignant = 2x Normal/Benign, from Part 3) ---
label_counts = Counter(label for _, label in train_ds.samples)
total = sum(label_counts.values())
class_weights = torch.tensor(
    [total / label_counts[i] for i in range(len(CLASSES))], dtype=torch.float32
).to(DEVICE)
print(f"Class weights: {dict(zip(CLASSES, class_weights.tolist()))}")

# --- Model: ResNet18, pretrained, replace final layer for our 3 classes ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers except the final block and the new classifier head --
# this is "fine-tuning" rather than "training from scratch": we keep
# ImageNet's low-level edge/texture knowledge, and only adapt the
# higher-level layers to histopathology-specific patterns.
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=LR
)

def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)

    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "models/resnet18_best.pt")
        print(f"  -> Saved new best model (val_acc={val_acc:.4f})")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")