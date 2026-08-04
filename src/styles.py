import streamlit as st
import json

PAGE_STYLE = """
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #10182b;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        color: #4a5568;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f4f6f9;
        border: 1px solid #d9e0ea;
        border-left: 3px solid #1a3a6b;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: center;
        color: #10182b;
    }
    .agree-badge {
        background: #eaf0f8;
        color: #1a3a6b;
        border: 1px solid #b8c9e0;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .disagree-badge {
        background: #f4f6f9;
        color: #10182b;
        border: 1px solid #10182b;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
"""


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []


@st.cache_resource
def load_calibration_data():
    try:
        with open("models/calibration_data.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None