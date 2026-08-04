import sys
sys.path.append("src")

import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import CLASSES, get_transforms
from gradcam import GradCAM, overlay_heatmap

DEVICE = torch.device("cpu")

# Rebuild the same ResNet18 architecture and load our fine-tuned weights
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("models/resnet18_best.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Target the last conv block -- this is where spatial info is richest
# right before global average pooling collapses it
gradcam = GradCAM(model, target_layer=model.layer4[-1])

transform = get_transforms(train=False)

# Run on one sample per class from the TEST set (untouched, unseen images)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, cls in enumerate(CLASSES):
    test_dir = Path("data/processed/test") / cls
    img_path = sorted(test_dir.glob("*.tif"))[0]

    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    cam, predicted_class = gradcam.generate(input_tensor)
    overlay = overlay_heatmap(img_resized, cam)

    axes[0, i].imshow(img_resized)
    axes[0, i].set_title(f"True: {cls} | {img_path.name}")
    axes[0, i].axis("off")

    axes[1, i].imshow(overlay)
    axes[1, i].set_title(f"Predicted: {CLASSES[predicted_class]}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("notebooks/gradcam_comparison.png", dpi=100)
print("Saved Grad-CAM comparison.")