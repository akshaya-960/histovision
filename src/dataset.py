import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as T

CLASSES = ["Normal", "Benign", "Malignant"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(train: bool):
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=90),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # NO augmentation for val/test -- must reflect real, unmodified images
        return T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class HistoDataset(Dataset):
    def __init__(self, root_dir, train: bool):
        self.root_dir = Path(root_dir)
        self.transform = get_transforms(train)
        self.samples = []  # list of (filepath, label_idx)

        for cls in CLASSES:
            for f in sorted((self.root_dir / cls).glob("*.tif")):
                self.samples.append((f, CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        img_bgr = cv2.imread(str(filepath))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb)
        return img_tensor, label