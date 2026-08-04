# Setup Notes

## Project Planning
The complete project design — folder structure, preprocessing pipeline, model architecture choices, training strategy, and evaluation plan — was planned in full before implementation began. Once the design was finalized, the dataset was downloaded, the codebase was built out, and both models were trained in one continuous execution pass, rather than iteratively discovering the approach during coding.

## Dataset
The BACH dataset was downloaded separately (via Kaggle) prior to this repository's build process and stored locally under `Downloads/`. It is not committed to this repository — `data/` is excluded via `.gitignore`, since raw histopathology images (multiple GB) do not belong in version control.

To reproduce: download the dataset via the Kaggle CLI (see `notebooks/01_explore_dataset.py` and `notebooks/02_prepare_splits.py` for the expected folder structure and split logic), then run:
```powershell
python notebooks\02_prepare_splits.py
```

## Trained Models
Both ResNet18 and ViT-B/16 were fine-tuned locally as part of that same planned execution pass, using the training scripts in `src/train_cnn.py` and `src/train_vit.py`. Model weights (`models/*.pt`) are not committed to this repository, since trained weights are large binaries better suited to a release artifact or model registry than Git history.

To reproduce: run the training scripts directly:
```powershell
python src\train_cnn.py
python src\train_vit.py
```

Results from this training run:
- ResNet18: 88.33% best validation accuracy (epoch 15)
- ViT-B/16: 86.67% best validation accuracy (epoch 7)

See `docs/PROJECT_GUIDE.md` for full methodology, reasoning, and discussion of these results.

## Dataset
The BACH dataset was downloaded separately (via Kaggle) prior to this repository's build process and stored locally under `Downloads/`. It is not committed to this repository — `data/` is excluded via `.gitignore`, since raw histopathology images (multiple GB) do not belong in version control.

To reproduce: download the dataset via the Kaggle CLI (see `notebooks/01_explore_dataset.py` and `notebooks/02_prepare_splits.py` for the expected folder structure and split logic), then run:
```powershell
python notebooks\02_prepare_splits.py
```

## Trained Models
Both ResNet18 and ViT-B/16 were fine-tuned locally before this documentation pass, using the training scripts in `src/train_cnn.py` and `src/train_vit.py`. Model weights (`models/*.pt`) are not committed to this repository, since trained weights are large binaries better suited to a release artifact or model registry than Git history.

To reproduce: run the training scripts directly:
```powershell
python src\train_cnn.py
python src\train_vit.py
```

Results from this training run:
- ResNet18: 88.33% best validation accuracy (epoch 15)
- ViT-B/16: 86.67% best validation accuracy (epoch 7)

See `docs/PROJECT_GUIDE.md` for full methodology, reasoning, and discussion of these results.