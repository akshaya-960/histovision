import os
import shutil
import random
from pathlib import Path

random.seed(42)

SOURCE_DIR = Path("data/raw/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos")
OUTPUT_DIR = Path("data/processed")

# Merge rule: BACH's 4 original classes -> our 3 target classes
CLASS_MAP = {
    "Normal": "Normal",
    "Benign": "Benign",
    "InSitu": "Malignant",
    "Invasive": "Malignant",
}

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

def prepare_splits():
    # Collect (filepath, target_class) pairs, merging InSitu+Invasive
    by_target_class = {"Normal": [], "Benign": [], "Malignant": []}

    for src_class, target_class in CLASS_MAP.items():
        class_dir = SOURCE_DIR / src_class
        images = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in {".tif", ".tiff"}])
        by_target_class[target_class].extend(images)

    for target_class, images in by_target_class.items():
        print(f"{target_class}: {len(images)} images total")

    # Shuffle and split each class independently (stratified split)
    for target_class, images in by_target_class.items():
        random.shuffle(images)
        n = len(images)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            out_dir = OUTPUT_DIR / split_name / target_class
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, out_dir / img_path.name)
            print(f"  {target_class}/{split_name}: {len(split_images)} images")

if __name__ == "__main__":
    prepare_splits()