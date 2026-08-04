import streamlit as st
import sys
sys.path.append("src")
from styles import PAGE_STYLE, load_calibration_data
from warmup import start_full_preload


st.set_page_config(page_title="HistoVision AI", layout="wide")
st.markdown(PAGE_STYLE, unsafe_allow_html=True)

start_full_preload()

st.markdown('<div class="main-header">HistoVision AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Breast histopathology classification research prototype — '
    'compare ResNet18 and ViT-B/16, with explainability, out-of-distribution detection, '
    'and calibration analysis.</div>',
    unsafe_allow_html=True
)

st.markdown("### Test Set Performance")
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="metric-card"><b>ResNet18</b><br>83.3% accuracy<br>90.0% Malignant recall</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><b>ViT-B/16</b><br>76.7% accuracy<br>86.7% Malignant recall</div>', unsafe_allow_html=True)

calibration_data = load_calibration_data()
if calibration_data:
    st.markdown("### Calibration (ECE, lower = better)")
    c3, c4 = st.columns(2)
    c3.metric("ResNet18", f"{calibration_data['resnet18']['ece']:.3f}")
    c4.metric("ViT-B16", f"{calibration_data['vit_b16']['ece']:.3f}")

st.markdown("### Navigate")
st.markdown("""
Use the sidebar to move between pages:
- **Analyze** — upload images, get dual-model predictions with explainability and OOD checks
- **History** — session log of every prediction made, with export
- **Calibration** — reliability diagrams for both models
- **Threshold Explorer** — interactively trade off Malignant precision vs. recall
""")

st.divider()
st.warning(
    "Research prototype trained on ~280 images. Not validated for clinical use. "
    "Not a diagnostic tool."
)