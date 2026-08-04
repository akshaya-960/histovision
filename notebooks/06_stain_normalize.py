import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Reinhard color normalization: matches the mean/std of L*a*b* channels
# between a "source" image and a "target" (reference) image.
# This does NOT require a stain-separation model (unlike Macenko/Vahadane),
# making it the simplest correct starting point.

def reinhard_normalize(source_rgb, target_rgb):
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    result = np.zeros_like(source_lab)
    for i in range(3):  # L, a, b channels independently
        s_mean, s_std = source_lab[:, :, i].mean(), source_lab[:, :, i].std()
        t_mean, t_std = target_lab[:, :, i].mean(), target_lab[:, :, i].std()

        # Shift source distribution to match target's mean/std
        channel = (source_lab[:, :, i] - s_mean) * (t_std / (s_std + 1e-6)) + t_mean
        result[:, :, i] = channel

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)


TRAIN_DIR = Path("data/processed/train")

# Pick two DIFFERENT images to simulate the "different slide/scanner" problem
normal_files = sorted((TRAIN_DIR / "Normal").glob("*.tif"))
malignant_files = sorted((TRAIN_DIR / "Malignant").glob("*.tif"))

source_path = malignant_files[0]
target_path = normal_files[0]  # this becomes our normalization "reference"

source_bgr = cv2.imread(str(source_path))
target_bgr = cv2.imread(str(target_path))
source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

normalized = reinhard_normalize(source_rgb, target_rgb)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(source_rgb); axes[0].set_title(f"Source: {source_path.name}"); axes[0].axis("off")
axes[1].imshow(target_rgb); axes[1].set_title(f"Target/Reference: {target_path.name}"); axes[1].axis("off")
axes[2].imshow(normalized); axes[2].set_title("Source, Normalized to Target"); axes[2].axis("off")
plt.tight_layout()
plt.savefig("notebooks/stain_normalize_comparison.png", dpi=100)
print("Saved stain normalization comparison.")