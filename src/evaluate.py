import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from dataset import HistoDataset, CLASSES

DEVICE = torch.device("cpu")
DATA_ROOT = Path("data/processed")
BATCH_SIZE = 16

test_ds = HistoDataset(DATA_ROOT / "test", train=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

def load_resnet():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("models/resnet18_best.pt", map_location=DEVICE))
    return model.to(DEVICE).eval()

def load_vit():
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.hidden_dim, len(CLASSES))
    model.load_state_dict(torch.load("models/vit_b16_best.pt", map_location=DEVICE))
    return model.to(DEVICE).eval()

def evaluate(model, name):
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    print(f"\n{'='*60}\n{name} — Test Set Results\n{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(f"notebooks/{name.lower()}_confusion_matrix.png", dpi=100)
    plt.close()

    # ROC-AUC (one-vs-rest, per class, since this is multiclass)
    labels_bin = label_binarize(all_labels, classes=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = {}
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[cls] = roc_auc
        ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} — ROC Curves (Test Set, One-vs-Rest)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"notebooks/{name.lower()}_roc_curve.png", dpi=100)
    plt.close()

    acc = (all_preds == all_labels).mean()
    print(f"\nOverall test accuracy: {acc:.4f}")
    print(f"Per-class AUC: {aucs}")

    return {"acc": acc, "cm": cm, "aucs": aucs, "labels": all_labels, "preds": all_preds}

resnet_results = evaluate(load_resnet(), "ResNet18")
vit_results = evaluate(load_vit(), "ViT-B16")