import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import json
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
def get_confidences_correctness(model):
    confidences, correct = [], []
    for images, labels in test_loader:
        images = images.to(DEVICE)
        probs = torch.softmax(model(images), dim=1)
        conf, preds = probs.max(dim=1)
        confidences.extend(conf.numpy())
        correct.extend((preds.numpy() == labels.numpy()).astype(int))
    return np.array(confidences), np.array(correct)


def reliability_diagram(confidences, correct, n_bins=10):
    """
    Bins predictions by confidence, compares each bin's average confidence
    against its actual accuracy. A perfectly calibrated model has accuracy
    == confidence in every bin (points lying on the diagonal). Expected
    Calibration Error (ECE) is the count-weighted average gap between them.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(mask.sum())
        if count > 0:
            acc = float(correct[mask].mean())
            conf = float(confidences[mask].mean())
            ece += (count / n) * abs(acc - conf)
        else:
            acc, conf = 0.0, float((lo + hi) / 2)
        bin_accs.append(acc)
        bin_confs.append(conf)
        bin_counts.append(count)

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "ece": float(ece),
    }


results = {}
for name, loader_fn in [("resnet18", load_resnet), ("vit_b16", load_vit)]:
    model = loader_fn()
    conf, correct = get_confidences_correctness(model)
    results[name] = reliability_diagram(conf, correct)
    print(f"{name}: ECE = {results[name]['ece']:.4f}  (lower is better)")

Path("models").mkdir(exist_ok=True)
with open("models/calibration_data.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved calibration data to models/calibration_data.json")