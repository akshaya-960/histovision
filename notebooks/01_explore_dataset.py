import os
from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/raw/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos")
def explore():
    print(f"Scanning: {DATA_DIR.resolve()}\n")
    
    for root, dirs, files in os.walk(DATA_DIR):
        depth = root.replace(str(DATA_DIR), "").count(os.sep)
        indent = "  " * depth
        print(f"{indent}{Path(root).name}/  ({len(files)} files)")

    print("\n--- Class counts ---")
    class_counts = {}
    image_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    for root, dirs, files in os.walk(DATA_DIR):
        images = [f for f in files if Path(f).suffix.lower() in image_exts]
        if images:
            class_name = Path(root).name
            class_counts[class_name] = class_counts.get(class_name, 0) + len(images)

    for cls, count in sorted(class_counts.items()):
        print(f"{cls}: {count} images")
    print(f"Total: {sum(class_counts.values())} images")

    print("\n--- Sample image check ---")
    sample_checked = False
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if Path(f).suffix.lower() in image_exts:
                img_path = Path(root) / f
                try:
                    with Image.open(img_path) as img:
                        print(f"{img_path.name}: size={img.size}, mode={img.mode}, format={img.format}")
                except Exception as e:
                    print(f"CORRUPTED: {img_path} — {e}")
                sample_checked = True
                break
        if sample_checked:
            break

if __name__ == "__main__":
    explore()