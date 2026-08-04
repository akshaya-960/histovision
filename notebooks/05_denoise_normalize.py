import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SAMPLE_DIR = Path("data/processed/train/Normal")
img_path = sorted(SAMPLE_DIR.glob("*.tif"))[0]

img_bgr = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --- Gaussian Blur: fast, but blurs edges uniformly ---
gaussian = cv2.GaussianBlur(img_rgb, (5, 5), sigmaX=1.0)

# --- Median Filter: better at removing salt-and-pepper noise, preserves edges better than Gaussian ---
median = cv2.medianBlur(img_rgb, 5)

# --- Bilateral Filter: smooths flat regions while preserving edges (nucleus boundaries) ---
bilateral = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ["Original", "Gaussian Blur", "Median Filter", "Bilateral Filter"]
images = [img_rgb, gaussian, median, bilateral]
for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig("notebooks/denoise_comparison.png", dpi=100)
print("Saved denoise comparison.")