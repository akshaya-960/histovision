import cv2
import matplotlib.pyplot as plt
from pathlib import Path

SAMPLE_IMAGE = Path("data/processed/train/Normal") 
sample_files = sorted(SAMPLE_IMAGE.glob("*.tif"))
img_path = sample_files[0]

img_bgr = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Original (RGB)")
axes[0].axis("off")

for i, (channel, name) in enumerate(zip(cv2.split(img_rgb), ["Red", "Green", "Blue"])):
    axes[i+1].imshow(channel, cmap="gray")
    axes[i+1].set_title(f"{name} channel")
    axes[i+1].axis("off")

plt.tight_layout()
plt.savefig("notebooks/raw_sample_channels.png", dpi=100)
print(f"Saved visualization from: {img_path.name}")
print(f"Image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
print(f"Min/Max pixel values: {img_rgb.min()}/{img_rgb.max()}")