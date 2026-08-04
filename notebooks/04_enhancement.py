import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SAMPLE_DIR = Path("data/processed/train/Normal")
img_path = sorted(SAMPLE_DIR.glob("*.tif"))[0]

img_bgr = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --- Plain (global) histogram equalization, applied on luminance only ---
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(img_lab)

l_eq = cv2.equalizeHist(l)
img_eq_lab = cv2.merge([l_eq, a, b])
img_eq = cv2.cvtColor(img_eq_lab, cv2.COLOR_LAB2RGB)

# --- CLAHE (Contrast Limited Adaptive Histogram Equalization), also on luminance ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)
img_clahe_lab = cv2.merge([l_clahe, a, b])
img_clahe = cv2.cvtColor(img_clahe_lab, cv2.COLOR_LAB2RGB)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(img_eq); axes[1].set_title("Global Histogram Equalization"); axes[1].axis("off")
axes[2].imshow(img_clahe); axes[2].set_title("CLAHE"); axes[2].axis("off")
plt.tight_layout()
plt.savefig("notebooks/enhancement_comparison.png", dpi=100)
print("Saved comparison image.")