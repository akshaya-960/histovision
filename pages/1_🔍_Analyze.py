import streamlit as st
import sys
sys.path.append("src")
import time
import numpy as np
import cv2
import torch
from PIL import Image
from datetime import datetime
import pandas as pd

from styles import PAGE_STYLE, init_session_state
from warmup import start_full_preload
from model_loaders import (
    DEVICE, load_resnet, load_vit, load_gradcam,
    load_vit_rollout, load_ood_detector, load_transform
)
from constants import CLASSES
from gradcam import overlay_heatmap
from vit_explain import overlay_attention
from report_generator import generate_pdf_report

st.set_page_config(page_title="Analyze — HistoVision AI", layout="wide")
st.markdown(PAGE_STYLE, unsafe_allow_html=True)
init_session_state()

LOADING_GRAPHIC = """
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:65vh;">
  <svg width="74" height="74" viewBox="0 0 74 74">
    <circle cx="37" cy="37" r="30" fill="none" stroke="#d9e0ea" stroke-width="5"/>
    <circle cx="37" cy="37" r="30" fill="none" stroke="#1a3a6b" stroke-width="5"
            stroke-linecap="round" stroke-dasharray="47 141">
      <animateTransform attributeName="transform" type="rotate"
                         from="0 37 37" to="360 37 37" dur="0.9s" repeatCount="indefinite"/>
    </circle>
  </svg>
  <div style="margin-top:22px; font-size:1.05rem; font-weight:700; color:#10182b;">
    Loading Analyzer
  </div>
  <div style="margin-top:4px; font-size:0.85rem; color:#4a5568;">
    Preparing ResNet18 and ViT-B/16 — this only happens once per session
  </div>
</div>
"""

placeholder = st.empty()
preload_thread = start_full_preload()

if preload_thread.is_alive():
    with placeholder.container():
        st.markdown(LOADING_GRAPHIC, unsafe_allow_html=True)
    time.sleep(0.4)
    st.rerun()

placeholder.empty()

resnet_model = load_resnet()
vit_model = load_vit()
gradcam = load_gradcam(resnet_model)
vit_rollout = load_vit_rollout(vit_model)
ood_detector = load_ood_detector(resnet_model)
transform = load_transform()

st.markdown('<div class="main-header">Analyze</div>', unsafe_allow_html=True)
st.caption("Upload histopathology image(s) for dual-model prediction, explainability, and OOD checks.")


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

    return {
        "filename": uploaded_file.name, "img_resized": img_resized,
        "overlay": resnet_overlay, "vit_overlay": vit_overlay,
        "min_dist": min_dist, "is_ood": is_ood,
        "resnet_probs": resnet_probs, "resnet_pred": resnet_pred,
        "vit_probs": vit_probs, "vit_pred": vit_pred,
        "agree": resnet_pred == vit_pred,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


def render_result(r, key_prefix):
    st.markdown(f"#### {r['filename']}")

    if r["is_ood"]:
        st.error(f"Out-of-distribution warning — distance={r['min_dist']:.1f}, threshold={ood_detector.threshold:.1f}.")
    else:
        st.success(f"In-distribution (distance={r['min_dist']:.1f}, threshold={ood_detector.threshold:.1f})")

    badge_class = "agree-badge" if r["agree"] else "disagree-badge"
    badge_text = "Models agree" if r["agree"] else "Models disagree"
    st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
    if not r["agree"]:
        st.caption("Two independently-trained architectures disagree — defer to a pathologist rather than trusting either score.")

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1: st.image(r["img_resized"], caption="Original", use_container_width=True)
    with c2: st.image(r["overlay"], caption="Grad-CAM (ResNet18)", use_container_width=True)
    with c3: st.image(r["vit_overlay"], caption="Attention Rollout (ViT-B/16)", use_container_width=True)

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"**ResNet18 → {CLASSES[r['resnet_pred']]}**")
        for cls, p in zip(CLASSES, r["resnet_probs"]):
            st.write(f"{cls}: {p*100:.1f}%")
            st.progress(float(p))
    with p2:
        st.markdown(f"**ViT-B/16 → {CLASSES[r['vit_pred']]}**")
        for cls, p in zip(CLASSES, r["vit_probs"]):
            st.write(f"{cls}: {p*100:.1f}%")
            st.progress(float(p))

    if CLASSES[r["resnet_pred"]] == "Malignant" and r["resnet_probs"][r["resnet_pred"]] < 0.7:
        st.warning("ResNet18 confidence is below 70% for Malignant — should be reviewed by a pathologist.")

    d1, d2 = st.columns(2)
    with d1:
        pdf_buffer = generate_pdf_report(r, CLASSES)
        st.download_button("Download PDF report", data=pdf_buffer, file_name=f"{r['filename']}_report.pdf",
                            mime="application/pdf", key=f"{key_prefix}_pdf")
    with d2:
        row = {
            "filename": r["filename"], "timestamp": r["timestamp"],
            "resnet_prediction": CLASSES[r["resnet_pred"]], "vit_prediction": CLASSES[r["vit_pred"]],
            "agree": r["agree"], "is_ood": r["is_ood"], "ood_distance": r["min_dist"],
            **{f"resnet_{c}": p for c, p in zip(CLASSES, r["resnet_probs"])},
            **{f"vit_{c}": p for c, p in zip(CLASSES, r["vit_probs"])},
        }
        st.download_button("Download CSV row", data=pd.DataFrame([row]).to_csv(index=False),
                            file_name=f"{r['filename']}_data.csv", mime="text/csv", key=f"{key_prefix}_csv")


uploaded_files = st.file_uploader(
    "Upload histopathology image(s) (.tif, .png, .jpg)",
    type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=True,
)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            result = analyze_image(uploaded_file)
        render_result(result, key_prefix=f"img{idx}_{uploaded_file.name}")
        st.session_state.history.append({
            "Time": result["timestamp"], "File": result["filename"],
            "ResNet18": CLASSES[result["resnet_pred"]], "ViT-B16": CLASSES[result["vit_pred"]],
            "Agree": "Yes" if result["agree"] else "No", "OOD": "Yes" if result["is_ood"] else "No",
        })
        st.divider()
else:
    st.info("Upload one or more images to get predictions from both models.")