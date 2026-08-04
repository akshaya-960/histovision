import torch
import torchvision.models as models
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

TRAIN_DIR = Path("data/processed/train")
CLASSES = ["Normal", "Benign", "Malignant"]

# Load a pretrained ResNet18, remove the final classification layer
# so it outputs a 512-dim feature vector instead of class scores.
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # drop final fc layer
feature_extractor.eval()

preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
])

def extract_features(img_rgb):
    tensor = preprocess(img_rgb).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        features = feature_extractor(tensor)
    return features.flatten().numpy()  # 512-dim vector

# Extract features for a handful of images per class (fast demo — full set later)
all_features = []
all_labels = []

for cls in CLASSES:
    files = sorted((TRAIN_DIR / cls).glob("*.tif"))[:15]  # sample 15 per class for speed
    for f in files:
        img_bgr = cv2.imread(str(f))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        feat = extract_features(img_rgb)
        all_features.append(feat)
        all_labels.append(cls)

all_features = np.array(all_features)
print(f"Extracted features shape: {all_features.shape}")  # (n_samples, 512)

# --- PCA: linear dimensionality reduction, 512 dims -> 2 dims ---
pca = PCA(n_components=2)
features_pca = pca.fit_transform(all_features)

# --- t-SNE: non-linear, better at preserving local cluster structure ---
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
features_tsne = tsne.fit_transform(all_features)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = {"Normal": "green", "Benign": "orange", "Malignant": "red"}

for cls in CLASSES:
    idx = [i for i, l in enumerate(all_labels) if l == cls]
    axes[0].scatter(features_pca[idx, 0], features_pca[idx, 1], label=cls, c=colors[cls])
    axes[1].scatter(features_tsne[idx, 0], features_tsne[idx, 1], label=cls, c=colors[cls])

axes[0].set_title("PCA (2D projection of ResNet18 features)")
axes[1].set_title("t-SNE (2D projection of ResNet18 features)")
for ax in axes:
    ax.legend()
plt.tight_layout()
plt.savefig("notebooks/feature_projection_comparison.png", dpi=100)
print("Saved feature projection comparison.")