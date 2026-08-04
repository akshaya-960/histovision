# HistoVision AI — Breast Histopathology Classification

A research prototype for classifying breast histopathology tissue patches as **Normal**,
**Benign**, or **Malignant**, comparing a CNN (ResNet18) against a Vision Transformer
(ViT-B/16), with explainability, out-of-distribution detection, calibration analysis, and
an interactive Streamlit demo.

> ⚠️ **This is a research/portfolio project, not a diagnostic tool.** It has not been
> validated for clinical use. See [Limitations](#limitations) before drawing any conclusions
> from its output.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Key Finding: ResNet18 vs ViT-B/16](#key-finding-resnet18-vs-vit-b16)
- [Why Malignant Recall, Not Accuracy](#why-malignant-recall-not-accuracy)
- [Explainability](#explainability)
- [Out-of-Distribution Detection](#out-of-distribution-detection)
- [Calibration](#calibration)
- [Demo App](#demo-app)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Problem Statement

Breast cancer histopathology diagnosis relies on a pathologist visually inspecting tissue
slides for structural abnormalities. This project asks two questions:

1. Can a model trained on a small (~280 image) dataset distinguish Normal, Benign, and
   Malignant tissue patches well enough to be a useful *decision-support* signal?
2. Between a CNN and a Vision Transformer — two architectures with fundamentally different
   inductive biases — which is better suited to this kind of small, high-stakes, medical
   imaging problem, and why?

Both questions are addressed with full quantitative evaluation, qualitative explainability
checks, and an honest accounting of where and how the system fails.

## Dataset

- **Classes:** Normal, Benign, Malignant (Malignant merges InSitu + Invasive carcinoma
  subtypes)
- **Split:** ~280 training images, held-out validation set (60 images) used for model
  checkpoint selection, and a separate untouched test set (60 images) used only once, for
  final evaluation
- **Class imbalance:** Malignant is roughly 2× the size of Normal/Benign after merging —
  addressed via class-weighted loss (weights inversely proportional to class frequency)
  rather than resampling, to avoid discarding limited training data
- Images are H&E-stained (hematoxylin and eosin) histopathology patches; the models are
  specific to this staining protocol and are not expected to generalize to other stains,
  scanners, or tissue types (see [Out-of-Distribution Detection](#out-of-distribution-detection))

## Project Pipeline

| Part | Stage | Summary |
|---|---|---|
| 1–2 | Data exploration & preprocessing | Class distribution analysis, imbalance identification, image preprocessing pipeline |
| 3 | Class imbalance strategy | Merged InSitu+Invasive into Malignant; established the recall-over-accuracy framing used throughout |
| 4 | Augmentation | Training-time augmentation strategy for a small dataset |
| 5 | Baseline model | Simple baseline to establish a performance floor before deep models |
| 6 | Data loading | `HistoDataset` class, transforms, DataLoader setup |
| 7 | CNN (ResNet18) | ImageNet-pretrained, fine-tuned on `layer4` + classifier head, class-weighted CE loss, Adam optimizer |
| 8 | Vision Transformer (ViT-B/16) | Same fine-tuning philosophy applied to the final transformer encoder block + head, for a controlled comparison |
| 9 | Grad-CAM | CNN explainability — visualizes which image regions drove each prediction |
| 10 | Test set evaluation | Precision/recall/F1/ROC-AUC/confusion matrices on the untouched test set |
| 11 | Deployment | Streamlit app: dual-model comparison, Grad-CAM + attention rollout, OOD detection, PDF/CSV export, calibration view |
| 12 | This README | Full write-up, setup instructions, limitations |

Full narrative write-up for every part lives in [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

## Key Finding: ResNet18 vs ViT-B/16

| Metric | ResNet18 | ViT-B/16 |
|---|---|---|
| Test accuracy | **83.3%** | 76.7% |
| Malignant recall | **90.0%** | 86.7% |
| Malignant precision | 81.8% | 78.8% |
| Malignant AUC | **0.924** | 0.906 |
| Normal recall | 80.0% | 73.3% |
| Benign recall | 73.3% | 60.0% |
| Calibration (ECE, lower = better) | 0.089 | **0.056** |

**ResNet18 outperformed ViT-B/16 on nearly every predictive metric**, and the gap widened
from validation to test — evidence this is a real, generalizing effect rather than a
one-off validation-set fluke. This is consistent with a well-documented property of Vision
Transformers: their lack of a built-in spatial-locality bias (no convolution) makes them
more flexible but more data-hungry, and this dataset — at ~280 training images — sits
squarely in the regime where a CNN's inductive bias is an advantage rather than a limitation.

Interestingly, **ViT-B/16 was better calibrated** (lower ECE) despite being less accurate —
its confidence scores, while wrong more often, were closer to being honestly wrong. This is
a useful reminder that accuracy and calibration are separate axes: a model can be more
accurate but more overconfident in its mistakes, or less accurate but more "honest" about
its uncertainty. See [Calibration](#calibration).

## Why Malignant Recall, Not Accuracy

Overall accuracy treats every misclassification as equally bad. In this problem, it isn't:

- **False negative** (a real Malignant case predicted as Normal or Benign) delays a cancer
  diagnosis — the worst-case outcome.
- **False positive** (a Benign or Normal case flagged as Malignant) triggers an unnecessary
  follow-up biopsy — costly and stressful, but not dangerous.

This asymmetry is why **Malignant recall** is treated as the primary metric throughout this
project, not overall accuracy. ResNet18's confusion matrix on the test set shows the concrete
trade-off this creates in practice: the dominant error type was Benign→Malignant
misclassification (the *safer* failure direction), while true Malignant cases were missed far
less often. Full confusion matrices are in `notebooks/resnet18_confusion_matrix.png` and
`notebooks/vit-b16_confusion_matrix.png`.

## Explainability

A raw "88% confident: Malignant" is not actionable for a clinician — it can't be verified or
sanity-checked. Two complementary explainability techniques are used, one per architecture:

- **Grad-CAM** (ResNet18) — backpropagates the predicted class's score to the final
  convolutional block, producing a heatmap of which spatial regions most influenced the
  prediction. On test images, Grad-CAM heatmaps aligned with real histological structures:
  attention on dense glandular tissue for Normal, a specific gland cluster for Benign, and a
  sharply-localized cribriform (gland-in-gland) growth pattern for Malignant — a genuine
  histological hallmark of carcinoma, not a spurious shortcut.
- **Attention Rollout** (ViT-B/16, Abnar & Zuidema, 2020) — since ViT has no convolutional
  feature map, this technique instead multiplies the model's self-attention matrices across
  all 12 encoder layers (with an identity-matrix correction per layer for residual
  connections), producing an equivalent heatmap of which image patches the classification
  token ultimately attended to.

Both are surfaced side-by-side in the demo app for every prediction.

**Caveat:** these are qualitative checks on a handful of illustrative images, not a
systematic validation across the full test set, and were not reviewed by a pathologist. They
demonstrate the technique and give an encouraging signal — not a clinical-grade
interpretability guarantee.

## Out-of-Distribution Detection

A softmax classifier always outputs a confident-looking prediction, even on inputs
completely unlike anything it was trained on — it has no built-in "I don't know." This was
confirmed directly during development: a stock photo of purported breast cancer tissue
(different scanner, different staining profile, sourced outside the training distribution)
was fed into the model and returned a confident, silently wrong "Normal" prediction.

To address this, the app includes a **Mahalanobis-distance OOD detector** (Lee et al., 2018):

1. Extract the 512-dim feature vector from ResNet18's penultimate layer for every training
   image.
2. Fit a per-class mean and a shared covariance matrix over these features, using
   **Ledoit-Wolf shrinkage** rather than the raw sample covariance — with 512 feature
   dimensions and only ~280 training images, the raw covariance matrix is singular/near-
   singular, and its naive inverse is numerically unstable (an early version of this detector
   produced Mahalanobis distances 250× too large for genuinely in-distribution images because
   of exactly this issue).
3. For a new image, compute its Mahalanobis distance to the nearest class mean in feature
   space.
4. Flag as out-of-distribution if that distance exceeds a threshold calibrated on a **held-out
   validation set** (not the training set itself — training images sit artificially close to
   their own class mean by construction, which would otherwise cause real, unseen
   in-distribution images to be falsely flagged).

Re-testing the same stock photo after implementing this correctly triggers the OOD warning
instead of returning a silent, confident, wrong prediction.

## Calibration

Accuracy asks "is the model usually right?" Calibration asks a different, arguably more
important question for a medical context: **"when the model says it's 80% confident, is it
actually right 80% of the time?"**

This is measured via **reliability diagrams** and **Expected Calibration Error (ECE)** —
test-set predictions are binned by confidence, and each bin's actual accuracy is compared
against its average stated confidence. ECE is the count-weighted average gap between the two;
lower is better, and bars *below* the diagonal (confidence higher than actual accuracy)
indicate overconfidence — the more dangerous failure mode in a medical setting, since it
means the model's stated certainty cannot be trusted at face value.

ResNet18: ECE = 0.089. ViT-B/16: ECE = 0.056. Full diagrams are viewable in the demo app's
Calibration tab, computed once via `src/calibration.py` on the 60-image test set.

## Demo App

An interactive Streamlit app (`app.py`) provides:

- **Batch upload** — analyze multiple images in one session
- **Dual-model comparison** — ResNet18 and ViT-B/16 predictions shown side by side, with an
  explicit agreement/disagreement badge (disagreement between two independently-trained
  architectures is itself a signal worth flagging for human review)
- **Explainability** — Grad-CAM and attention rollout overlays for every prediction
- **OOD warnings** — flags images statistically unlike the training distribution
- **Session history** — running table of every prediction made, with agreement/OOD summary
  stats and CSV export
- **Per-image PDF/CSV reports** — downloadable, shareable output per analyzed image
- **Calibration tab** — live reliability diagrams for both models

Screenshot placeholders — add your own after running the app:

```
[screenshot: main analysis view]
[screenshot: model disagreement case]
[screenshot: OOD warning triggered]
[screenshot: calibration tab]
```

## Setup & Installation

**Requirements:** Python 3.13 (developed and tested on 3.13.0; likely works on 3.10+, not
verified).

```powershell
git clone https://github.com/akshaya-960/histovision-ai.git
cd histovision-ai
```

**(Recommended) create a virtual environment:**

```powershell
python -m venv venv
venv\Scripts\activate
```

**Install dependencies:**

```powershell
pip install -r requirements.txt
```

`requirements.txt` pins CPU-only PyTorch by default. If installing fresh and you hit
resolution issues, or want GPU acceleration instead, install torch/torchvision explicitly
first:

```powershell
# CPU-only:
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA): omit the --index-url flag and consult pytorch.org for the correct CUDA build
pip install -r requirements.txt
```

## Running the Project

### 1. Get the data

This repo does **not** include the dataset (`data/processed/` is gitignored). Place your
train/val/test split under:

```
data/processed/train/{Normal,Benign,Malignant}/
data/processed/val/{Normal,Benign,Malignant}/
data/processed/test/{Normal,Benign,Malignant}/
```

> _TODO: add the specific dataset source/link here (e.g. original Kaggle/competition page),
> and note the preprocessing script if raw data needs to be converted to this structure first._

### 2. Get trained model weights — two options

**Option A — download pretrained weights (fastest):**

```powershell
pip install huggingface_hub
hf download <your-hf-username>/histovision-ai-weights resnet18_best.pt --local-dir models
hf download <your-hf-username>/histovision-ai-weights vit_b16_best.pt --local-dir models
```

> _TODO: replace `<your-hf-username>/histovision-ai-weights` with the actual Hugging Face
> repo path once uploaded._

**Option B — train from scratch:**

```powershell
python src\train_resnet.py   # ~few minutes on CPU
python src\train_vit.py      # noticeably longer on CPU — ViT-B/16 is a much larger model
```

Both scripts save their best-validation-accuracy checkpoint to `models/`.

### 3. Fit the OOD detector (required before running the app)

```powershell
python src\fit_ood.py
```

Requires `models/resnet18_best.pt` to already exist. Produces `models/ood_stats.npz`.

### 4. Run calibration analysis (required for the Calibration tab)

```powershell
python src\calibration.py
```

Requires both model checkpoints. Produces `models/calibration_data.json`.

### 5. (Optional) Full test-set evaluation

```powershell
python src\evaluate.py
```

Produces confusion matrices and ROC curves in `notebooks/`.

### 6. (Optional) Standalone Grad-CAM demo

```powershell
python notebooks\09_gradcam_demo.py
```

### 7. Run the app

```powershell
streamlit run app.py
```

Opens at `http://localhost:8501`.

**Run order matters** — steps 3 and 4 depend on the model checkpoints from step 2, and the
app depends on the outputs of both. If the app throws a `FileNotFoundError` for
`ood_stats.npz` or `calibration_data.json`, it means one of those steps was skipped.

## Repository Structure

```
histovision-ai/
├── app.py                      # Streamlit demo app
├── requirements.txt
├── src/
│   ├── dataset.py               # HistoDataset, transforms, CLASSES
│   ├── train_resnet.py          # Part 7: ResNet18 fine-tuning
│   ├── train_vit.py             # Part 8: ViT-B/16 fine-tuning
│   ├── gradcam.py               # Part 9: Grad-CAM (CNN explainability)
│   ├── vit_explain.py           # Attention rollout (ViT explainability)
│   ├── evaluate.py              # Part 10: full test-set evaluation
│   ├── ood_detector.py          # Mahalanobis OOD detector
│   ├── fit_ood.py               # Fits OOD detector on training data
│   ├── calibration.py           # Reliability diagrams / ECE
│   └── report_generator.py      # PDF report generation for the app
├── notebooks/
│   ├── 09_gradcam_demo.py
│   └── *.png                    # confusion matrices, ROC curves, Grad-CAM comparisons
├── docs/
│   └── PROJECT_GUIDE.md         # full narrative write-up, Parts 1–12
├── models/                      # gitignored — see Setup & Installation
└── data/                        # gitignored — see Setup & Installation
```

## Limitations

Documented honestly, not as an afterthought:

- **Small dataset.** ~280 training images, 60 validation, 60 test. Individual
  misclassifications swing per-class recall by several percentage points; treat all metrics
  as directional, not precise population statistics.
- **Single-source data.** All images come from one dataset — presumably one scanner, one
  staining batch, one institution. Performance on data from a different hospital, scanner, or
  staining protocol is unknown and, per the OOD testing done here, likely to degrade
  significantly without warning unless the OOD detector is in place.
- **No OOD detection without the safeguard.** The base classifier alone will confidently
  misclassify any input, including images entirely outside its domain — demonstrated directly
  during development. The Mahalanobis detector mitigates but does not eliminate this risk.
- **Grad-CAM/attention-rollout validation is qualitative and small-scale.** A handful of
  illustrative examples, not a systematic audit, and not reviewed by a pathologist.
- **Recall levels (86–90% for Malignant) are not clinically sufficient.** A real triage tool
  would need recall far closer to 99%+, and even then would function as a decision-support
  aid alongside a pathologist, never as a standalone diagnostic.
- **No external validation set.** All evaluation is on data drawn from the same source and
  distribution as training data, just held out. True generalization to new hospitals/patients
  is untested.

## Future Work

- Expand the dataset, ideally with multi-institution data, to test real generalization
- Systematic, pathologist-reviewed validation of Grad-CAM/attention-rollout alignment across
  the full test set
- Threshold tuning for the Malignant class specifically (deliberately trading precision for
  recall, informed by the ROC/AUC analysis in Part 10)
- Ensemble ResNet18 + ViT-B/16 predictions, potentially using their disagreement rate as an
  automatic "flag for human review" signal
- Extend OOD detection with a second, independent signal (e.g. reconstruction-based) to
  cross-check the Mahalanobis approach

## References

- Lee, K. et al. (2018). *A Simple Unified Framework for Detecting Out-of-Distribution
  Samples and Adversarial Attacks.* NeurIPS.
- Abnar, S. & Zuidema, W. (2020). *Quantifying Attention Flow in Transformers.* ACL.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization.* ICCV.
- Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional
  covariance matrices.* Journal of Multivariate Analysis.
- Dosovitskiy, A. et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image
  Recognition at Scale.* ICLR.

---

*Built as an academic/portfolio project. Not affiliated with or endorsed by any hospital or
clinical institution. Not for clinical use.*
