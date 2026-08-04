import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append("src")

from styles import PAGE_STYLE
from constants import CLASSES
from calibration_metrics import (
    compute_reliability, ece_mce, brier_score, nll,
    temperature_scale, find_best_temperature
)

st.set_page_config(page_title="Calibration — HistoVision AI", layout="wide")
st.markdown(PAGE_STYLE, unsafe_allow_html=True)

st.markdown('<div class="main-header">Calibration</div>', unsafe_allow_html=True)
st.caption("Is the model's stated confidence trustworthy, not just its accuracy?")

if not Path("models/test_predictions.npz").exists():
    st.warning("Run `python src/precompute_test_predictions.py` once, then restart the app.")
    st.stop()

data = np.load("models/test_predictions.npz")
labels = data["labels"]
n_classes = len(CLASSES)
model_data = {"ResNet18": data["resnet_probs"], "ViT-B/16": data["vit_probs"]}

tab1, tab2, tab3 = st.tabs(["Overview", "Per-Class", "Temperature Scaling"])

# ---------------------------------------------------------------- Overview
with tab1:
    st.write(
        "A perfectly calibrated model's bars sit on the diagonal: when it says '80% confident,' "
        "it should be right about 80% of the time. Bars below the diagonal mean overconfidence — "
        "the more dangerous failure mode in a medical context."
    )

    metrics_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, probs) in zip(axes, model_data.items()):
        confidences = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == labels).astype(float)

        bin_edges, bin_accs, bin_confs, bin_counts = compute_reliability(confidences, correct)
        ece, mce = ece_mce(bin_accs, bin_confs, bin_counts)
        brier = brier_score(probs, labels, n_classes)
        loss = nll(probs, labels)
        metrics_rows.append({"Model": name, "ECE": ece, "MCE": mce, "Brier Score": brier, "NLL": loss})

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color="#1a3a6b", edgecolor="#10182b", label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{name} (ECE={ece:.3f}, MCE={mce:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("#### Metrics")
    st.table({
        "Model": [r["Model"] for r in metrics_rows],
        "ECE ↓": [f"{r['ECE']:.3f}" for r in metrics_rows],
        "MCE ↓": [f"{r['MCE']:.3f}" for r in metrics_rows],
        "Brier Score ↓": [f"{r['Brier Score']:.3f}" for r in metrics_rows],
        "NLL ↓": [f"{r['NLL']:.3f}" for r in metrics_rows],
    })
    st.caption(
        "ECE: count-weighted average gap between confidence and accuracy across bins. "
        "MCE: the single worst bin's gap — the model's worst-case miscalibration. "
        "Brier Score: mean squared error between predicted probabilities and the true outcome, "
        "rewarding both correctness and well-calibrated confidence. NLL: average negative log-"
        "likelihood assigned to the correct class — heavily penalizes confident wrong answers. "
        "Lower is better for all four."
    )

    st.markdown("#### Confidence distribution")
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, (name, probs) in zip(axes2, model_data.items()):
        confidences = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct_mask = preds == labels
        ax.hist(confidences[correct_mask], bins=15, range=(0, 1), alpha=0.7,
                color="#1a3a6b", label="Correct", edgecolor="white")
        ax.hist(confidences[~correct_mask], bins=15, range=(0, 1), alpha=0.7,
                color="#c0392b", label="Incorrect", edgecolor="white")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    st.caption(
        "Incorrect predictions clustering at high confidence (right side of the red bars) is the "
        "pattern to watch for — it means the model is confidently wrong, which is worse than being "
        "uncertain and wrong."
    )

# --------------------------------------------------------------- Per-Class
with tab2:
    st.write(
        "Overall calibration can hide class-specific problems — a model can be well-calibrated "
        "on average while being systematically overconfident on the one class that matters most "
        "(Malignant). Each curve below treats one class as 'positive' and asks: of all cases where "
        "the model output a given probability for this class, how often was it actually that class?"
    )

    model_choice_pc = st.radio("Model", list(model_data.keys()), horizontal=True, key="pc_model")
    probs_pc = model_data[model_choice_pc]

    fig3, axes3 = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes3 = [axes3]
    for ax, cls_idx in zip(axes3, range(n_classes)):
        cls_confidences = probs_pc[:, cls_idx]
        cls_correct = (labels == cls_idx).astype(float)
        bin_edges, bin_accs, bin_confs, bin_counts = compute_reliability(cls_confidences, cls_correct)
        ece, _ = ece_mce(bin_accs, bin_confs, bin_counts)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color="#1a3a6b", edgecolor="#10182b")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel(f"P({CLASSES[cls_idx]})")
        ax.set_ylabel(f"Actually {CLASSES[cls_idx]}")
        ax.set_title(f"{CLASSES[cls_idx]} (ECE={ece:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    st.pyplot(fig3)

# ------------------------------------------------------- Temperature Scaling
with tab3:
    st.write(
        "Temperature scaling divides the pre-softmax logits by a single learned constant T before "
        "reapplying softmax. T > 1 softens (de-confidences) an overconfident model; T < 1 sharpens "
        "an underconfident one. It changes confidence, never changes which class wins, so accuracy "
        "is unaffected."
    )
    st.caption(
        "Note: T here is fit and evaluated on the same test set for demonstration. A rigorous "
        "pipeline fits T on a separate held-out calibration split so the reported metrics stay honest."
    )

    model_choice_ts = st.radio("Model", list(model_data.keys()), horizontal=True, key="ts_model")
    probs_ts = model_data[model_choice_ts]

    best_T, best_nll = find_best_temperature(probs_ts, labels)
    st.info(f"Optimal T for {model_choice_ts}: **{best_T:.2f}** (minimizes NLL to {best_nll:.3f})")

    T = st.slider("Temperature (T)", min_value=0.3, max_value=3.0, value=1.0, step=0.02)
    scaled_probs = temperature_scale(probs_ts, T)

    confidences_raw = probs_ts.max(axis=1)
    confidences_scaled = scaled_probs.max(axis=1)
    correct = (probs_ts.argmax(axis=1) == labels).astype(float)

    edges_r, accs_r, confs_r, counts_r = compute_reliability(confidences_raw, correct)
    ece_r, _ = ece_mce(accs_r, confs_r, counts_r)
    edges_s, accs_s, confs_s, counts_s = compute_reliability(confidences_scaled, correct)
    ece_s, _ = ece_mce(accs_s, confs_s, counts_s)

    c1, c2 = st.columns(2)
    c1.metric("ECE before (T=1.0)", f"{ece_r:.3f}")
    c2.metric(f"ECE after (T={T:.2f})", f"{ece_s:.3f}", delta=f"{ece_s - ece_r:+.3f}", delta_color="inverse")

    fig4, axes4 = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (title, edges, accs, ece_val) in zip(
        axes4, [("Before (T=1.0)", edges_r, accs_r, ece_r), (f"After (T={T:.2f})", edges_s, accs_s, ece_s)]
    ):
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, accs, width=0.08, alpha=0.7, color="#1a3a6b", edgecolor="#10182b")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title} — ECE={ece_val:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    st.pyplot(fig4)