import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import sys
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append("src")
from dataset import CLASSES, get_transforms
from gradcam import GradCAM, overlay_heatmap
from ood_detector import MahalanobisOOD
from vit_explain import ViTAttentionRollout, overlay_attention
from report_generator import generate_pdf_report

DEVICE = torch.device("cpu")

st.set_page_config(page_title="HistoVision AI", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .metric-card { background: #f8f9fb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 16px; text-align: center; }
    .agree-badge { background: #e7f7ed; color: #1a7f37; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .disagree-badge { background: #fdecea; color: #c92a2a; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resnet():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("models/resnet18_best.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_vit():
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.hidden_dim, len(CLASSES))
    model.load_state_dict(torch.load("models/vit_b16_best.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_ood_detector():
    feature_model = models.resnet18(weights=None)
    feature_model.fc = nn.Linear(feature_model.fc.in_features, len(CLASSES))
    feature_model.load_state_dict(torch.load("models/resnet18_best.pt", map_location=DEVICE))
    feature_model.fc = nn.Identity()
    feature_model.to(DEVICE).eval()
    ood = MahalanobisOOD(feature_extractor=feature_model, device=DEVICE)
    ood.load("models/ood_stats.npz")
    return ood


@st.cache_resource
def load_calibration_data():
    try:
        with open("models/calibration_data.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


resnet_model = load_resnet()
vit_model = load_vit()
gradcam = GradCAM(resnet_model, target_layer=resnet_model.layer4[-1])
@st.cache_resource
def load_vit_rollout(_vit_model):
    return ViTAttentionRollout(_vit_model)

vit_rollout = load_vit_rollout(vit_model)
ood_detector = load_ood_detector()
calibration_data = load_calibration_data()
transform = get_transforms(train=False)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### 🔬 HistoVision AI")
    st.caption("Breast histopathology classification research prototype")

    st.markdown("#### Test Set Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="metric-card"><b>ResNet18</b><br>83.3% acc<br>90.0% Malig. recall</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><b>ViT-B/16</b><br>76.7% acc<br>86.7% Malig. recall</div>', unsafe_allow_html=True)

    if calibration_data:
        st.markdown("#### Calibration (ECE, lower=better)")
        c3, c4 = st.columns(2)
        c3.metric("ResNet18", f"{calibration_data['resnet18']['ece']:.3f}")
        c4.metric("ViT-B16", f"{calibration_data['vit_b16']['ece']:.3f}")

    st.markdown("#### About")
    st.caption(
        "Classifies tissue patches as **Normal**, **Benign**, or **Malignant**, with "
        "Grad-CAM (ResNet18) and attention rollout (ViT) explainability, and "
        "Mahalanobis-distance out-of-distribution detection (Lee et al., 2018)."
    )

    st.markdown("#### Disclaimer")
    st.caption("⚠️ Research prototype trained on ~280 images. Not validated for clinical use.")

    if st.session_state.history:
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()

st.markdown('<div class="main-header">HistoVision AI — Breast Histopathology Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Compare ResNet18 and ViT-B/16 predictions side by side, with '
    'explainability, OOD detection, and downloadable reports.</div>',
    unsafe_allow_html=True
)

tab_analyze, tab_history, tab_calibration = st.tabs(["🔍 Analyze", "📋 History", "📐 Calibration"])


def analyze_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(img)
    img_resized = cv2.resize(img_rgb, (224, 224))
    input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    min_dist, is_ood, _ = ood_detector.score(input_tensor)

    resnet_out = resnet_model(input_tensor)
    resnet_probs = torch.softmax(resnet_out, dim=1)[0].detach().numpy()
    resnet_pred = int(np.argmax(resnet_probs))
    cam, _ = gradcam.generate(input_tensor, target_class=resnet_pred)
    resnet_overlay = overlay_heatmap(img_resized, cam)

    vit_out = vit_model(input_tensor)
    vit_probs = torch.softmax(vit_out, dim=1)[0].detach().numpy()
    vit_pred = int(np.argmax(vit_probs))
    attn_mask = vit_rollout.generate(input_tensor)
    vit_overlay = overlay_attention(img_resized, attn_mask)

    agree = resnet_pred == vit_pred

    return {
        "filename": uploaded_file.name,
        "img_resized": img_resized,
        "overlay": resnet_overlay,
        "vit_overlay": vit_overlay,
        "min_dist": min_dist,
        "is_ood": is_ood,
        "resnet_probs": resnet_probs,
        "resnet_pred": resnet_pred,
        "vit_probs": vit_probs,
        "vit_pred": vit_pred,
        "agree": agree,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


def render_result(r, key_prefix):
    st.markdown(f"#### {r['filename']}")

    if r["is_ood"]:
        st.error(
            f"⚠️ **Out-of-distribution warning** — distance={r['min_dist']:.1f}, "
            f"threshold={ood_detector.threshold:.1f}. Predictions below may be unreliable."
        )
    else:
        st.success(f"✓ In-distribution (distance={r['min_dist']:.1f}, threshold={ood_detector.threshold:.1f})")

    badge_class = "agree-badge" if r["agree"] else "disagree-badge"
    badge_text = "✓ Models agree" if r["agree"] else "⚠ Models disagree"
    st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
    if not r["agree"]:
        st.caption(
            "Two independently-trained architectures reached different conclusions here — "
            "cases like this are where automated triage should defer to a pathologist rather "
            "than trusting either model's confidence score."
        )

    st.write("")
    img_col1, img_col2, img_col3 = st.columns(3)
    with img_col1:
        st.image(r["img_resized"], caption="Original (224x224)", use_container_width=True)
    with img_col2:
        st.image(r["overlay"], caption="Grad-CAM (ResNet18)", use_container_width=True)
    with img_col3:
        st.image(r["vit_overlay"], caption="Attention Rollout (ViT-B/16)", use_container_width=True)

    pred_col1, pred_col2 = st.columns(2)
    with pred_col1:
        st.markdown(f"**ResNet18 → {CLASSES[r['resnet_pred']]}**")
        for cls, p in zip(CLASSES, r["resnet_probs"]):
            st.write(f"{cls}: {p*100:.1f}%")
            st.progress(float(p))
    with pred_col2:
        st.markdown(f"**ViT-B/16 → {CLASSES[r['vit_pred']]}**")
        for cls, p in zip(CLASSES, r["vit_probs"]):
            st.write(f"{cls}: {p*100:.1f}%")
            st.progress(float(p))

    if CLASSES[r["resnet_pred"]] == "Malignant" and r["resnet_probs"][r["resnet_pred"]] < 0.7:
        st.warning(
            "ResNet18 confidence is below 70% for a Malignant prediction. Low-confidence "
            "Malignant/Benign predictions should always be reviewed by a pathologist."
        )

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        pdf_buffer = generate_pdf_report(r, CLASSES)
        st.download_button(
            "📄 Download PDF report",
            data=pdf_buffer,
            file_name=f"{r['filename']}_report.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf",
        )
    with dl_col2:
        row = {
            "filename": r["filename"], "timestamp": r["timestamp"],
            "resnet_prediction": CLASSES[r["resnet_pred"]],
            "vit_prediction": CLASSES[r["vit_pred"]],
            "agree": r["agree"], "is_ood": r["is_ood"], "ood_distance": r["min_dist"],
            **{f"resnet_{c}": p for c, p in zip(CLASSES, r["resnet_probs"])},
            **{f"vit_{c}": p for c, p in zip(CLASSES, r["vit_probs"])},
        }
        csv_data = pd.DataFrame([row]).to_csv(index=False)
        st.download_button(
            "📊 Download CSV row",
            data=csv_data,
            file_name=f"{r['filename']}_data.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
        )


with tab_analyze:
    uploaded_files = st.file_uploader(
        "Upload histopathology image(s) (.tif, .png, .jpg)",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for idx, uploaded_file in enumerate(uploaded_files):
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                result = analyze_image(uploaded_file)

            render_result(result, key_prefix=f"img{idx}_{uploaded_file.name}")
            st.session_state.history.append({
                "Time": result["timestamp"], "File": result["filename"],
                "ResNet18": CLASSES[result["resnet_pred"]], "ViT-B16": CLASSES[result["vit_pred"]],
                "Agree": "✓" if result["agree"] else "✗", "OOD": "⚠️" if result["is_ood"] else "—",
            })
            st.divider()
    else:
        st.info("Upload one or more images to get predictions from both models.")

with tab_history:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Images analyzed", len(df))
        c2.metric("Model agreement rate", f"{(df['Agree'] == '✓').mean()*100:.0f}%")
        c3.metric("OOD flags", int((df["OOD"] == "⚠️").sum()))

        st.download_button(
            "📊 Download full session history (CSV)",
            data=df.to_csv(index=False),
            file_name="histovision_session_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No predictions yet this session. Analyze an image to build history.")

with tab_calibration:
    if calibration_data is None:
        st.warning("Calibration data not found. Run `python src/calibration.py` once, then restart the app.")
    else:
        st.markdown("#### Reliability Diagrams — Test Set")
        st.caption(
            "A perfectly calibrated model's bars sit on the diagonal: when it says "
            "'80% confident,' it should be right about 80% of the time. Bars above the "
            "diagonal mean the model is underconfident there; bars below mean overconfident — "
            "in a medical context, overconfidence is the more dangerous failure mode."
        )

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (name, label) in zip(axes, [("resnet18", "ResNet18"), ("vit_b16", "ViT-B/16")]):
            d = calibration_data[name]
            bin_centers = [(d["bin_edges"][i] + d["bin_edges"][i+1]) / 2 for i in range(len(d["bin_edges"]) - 1)]
            ax.bar(bin_centers, d["bin_accs"], width=0.08, alpha=0.7, edgecolor="black", label="Accuracy")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{label} (ECE={d['ece']:.3f})")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "ECE (Expected Calibration Error) is the count-weighted average gap between "
            "confidence and accuracy across bins — lower is better. Computed once on the "
            "60-image held-out test set; treat as directional given the small sample size."
        )

st.divider()
st.caption(
    "Trained on ~280 images, evaluated on a 60-image held-out test set. OOD detection uses "
    "Mahalanobis distance (Lee et al., 2018). ViT explainability uses attention rollout "
    "(Abnar & Zuidema, 2020). Session history is not persisted between app restarts. "
    "Not validated for clinical use."
)