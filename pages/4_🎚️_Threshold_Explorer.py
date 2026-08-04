import streamlit as st
import numpy as np
from pathlib import Path
import sys
sys.path.append("src")
from styles import PAGE_STYLE
from constants import CLASSES

st.set_page_config(page_title="Threshold Explorer — HistoVision AI", layout="wide")
st.markdown(PAGE_STYLE, unsafe_allow_html=True)

st.markdown('<div class="main-header">Threshold Explorer</div>', unsafe_allow_html=True)
st.caption(
    "The default classifier picks whichever class has the highest probability. In a medical "
    "context, that's not always the right rule — missing a real cancer (false negative) is "
    "far worse than an unnecessary follow-up (false positive). Drag the threshold to see the "
    "trade-off directly, computed on the untouched test set."
)

if not Path("models/test_predictions.npz").exists():
    st.warning("Run `python src/precompute_test_predictions.py` once, then restart the app.")
    st.stop()

data = np.load("models/test_predictions.npz")
malignant_idx = CLASSES.index("Malignant")

model_choice = st.radio("Model", ["ResNet18", "ViT-B/16"], horizontal=True)
probs = data["resnet_probs"] if model_choice == "ResNet18" else data["vit_probs"]
labels = data["labels"]

threshold = st.slider(
    "Malignant decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    help="Predict Malignant whenever its probability exceeds this value, regardless of the other two classes."
)

malignant_probs = probs[:, malignant_idx]
true_malignant = (labels == malignant_idx)
pred_malignant = (malignant_probs >= threshold)

tp = int(np.sum(pred_malignant & true_malignant))
fn = int(np.sum(~pred_malignant & true_malignant))
fp = int(np.sum(pred_malignant & ~true_malignant))
tn = int(np.sum(~pred_malignant & ~true_malignant))

recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Malignant Recall", f"{recall*100:.1f}%", help="Of all real Malignant cases, how many did we catch?")
c2.metric("Malignant Precision", f"{precision*100:.1f}%", help="Of everything flagged Malignant, how many really are?")
c3.metric("False Negatives", fn, help="Real cancers missed — the costliest error type")
c4.metric("False Positives", fp, help="Unnecessary follow-ups triggered")

st.divider()
st.markdown("#### Confusion breakdown at this threshold")
st.table({
    "": ["Predicted Malignant", "Predicted Not-Malignant"],
    "Actually Malignant": [tp, fn],
    "Actually Not-Malignant": [fp, tn],
})

if threshold < 0.5:
    st.info(
        f"At threshold {threshold:.2f} (below default 0.5), the model flags more cases as "
        "Malignant — this raises recall (fewer missed cancers) at the cost of more false "
        "alarms. A common deliberate choice in cancer screening, where missing a case is far "
        "costlier than an unnecessary follow-up."
    )
elif threshold > 0.5:
    st.info(
        f"At threshold {threshold:.2f} (above default 0.5), the model is more conservative "
        "about flagging Malignant — fewer false alarms, but at the risk of missing more real "
        "cases. Generally the wrong direction to move for a cancer-screening use case."
    )