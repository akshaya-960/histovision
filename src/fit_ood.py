import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import HistoDataset
from ood_detector import MahalanobisOOD

DEVICE = torch.device("cpu")
DATA_ROOT = Path("data/processed")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("models/resnet18_best.pt", map_location=DEVICE))
model.fc = nn.Identity()
model.to(DEVICE).eval()

train_ds = HistoDataset(DATA_ROOT / "train", train=False)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)

val_ds = HistoDataset(DATA_ROOT / "val", train=False)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

ood = MahalanobisOOD(feature_extractor=model, device=DEVICE)
ood.fit(train_loader, calibration_loader=val_loader, percentile=99.0)

Path("models").mkdir(exist_ok=True)
ood.save("models/ood_stats.npz")
print("Saved OOD detector stats to models/ood_stats.npz")