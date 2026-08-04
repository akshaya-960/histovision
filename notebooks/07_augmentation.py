import cv2
import albumentations as A
import matplotlib.pyplot as plt
from pathlib import Path

SAMPLE_DIR = Path("data/processed/train/Normal")
img_path = sorted(SAMPLE_DIR.glob("*.tif"))[0]

img_bgr = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Each transform isolated so we can see its individual effect
transforms = {
    "Original": A.Compose([]),
    "Rotation (90)": A.Compose([A.SafeRotate(limit=(90, 90), p=1.0)]),
    "Horizontal Flip": A.Compose([A.HorizontalFlip(p=1.0)]),
    "Brightness +": A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.3, 0.3), contrast_limit=0, p=1.0)]),
    "Contrast +": A.Compose([A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.4, 0.4), p=1.0)]),
    "Random Crop (80%)": A.Compose([A.RandomCrop(height=int(img_rgb.shape[0]*0.8), width=int(img_rgb.shape[1]*0.8), p=1.0)]),
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, (name, transform) in zip(axes, transforms.items()):
    augmented = transform(image=img_rgb)["image"]
    ax.imshow(augmented)
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.savefig("notebooks/augmentation_comparison.png", dpi=100)
print("Saved augmentation comparison.")