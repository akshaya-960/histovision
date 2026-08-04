import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import HistoDataset, CLASSES

DEVICE = torch.device("cpu")
DATA_ROOT = Path("data/processed")
test_ds = HistoDataset(DATA_ROOT / "test", train=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)


def load_resnet():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load("models/resnet18_best.pt", map_location=DEVICE))
    return m.to(DEVICE).eval()


def load_vit():
    m = models.vit_b_16(weights=None)
    m.heads.head = nn.Linear(m.hidden_dim, len(CLASSES))
    m.load_state_dict(torch.load("models/vit_b16_best.pt", map_location=DEVICE))
    return m.to(DEVICE).eval()


@torch.no_grad()
def get_probs_labels(model):
    all_probs, all_labels = [], []
    for images, labels in test_loader:
        probs = torch.softmax(model(images.to(DEVICE)), dim=1)
        all_probs.append(probs.numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)

resnet_probs, labels = get_probs_labels(load_resnet())
vit_probs, _ = get_probs_labels(load_vit())

Path("models").mkdir(exist_ok=True)
np.savez("models/test_predictions.npz",
         resnet_probs=resnet_probs, vit_probs=vit_probs, labels=labels)
print("Saved test set probabilities to models/test_predictions.npz")