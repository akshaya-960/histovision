import threading
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx


@st.cache_resource(show_spinner=False)
def start_full_preload():
    """
    Everything heavy — third-party library imports AND model loading — runs
    on a single background thread, kicked off here. This function itself
    only creates and starts the thread, which takes milliseconds, so Home.py
    never blocks on it and no loading text needs to appear at all.
    show_spinner=False stops Streamlit's own default "Running
    start_full_preload()..." status message from being displayed.
    """
    def _preload():
        import pandas  # noqa: F401
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401
        import cv2  # noqa: F401

        from model_loaders import (
            load_resnet, load_vit, load_gradcam,
            load_vit_rollout, load_ood_detector, load_transform,
        )
        resnet_model = load_resnet()
        vit_model = load_vit()
        load_gradcam(resnet_model)
        load_vit_rollout(vit_model)
        load_ood_detector(resnet_model)
        load_transform()

    thread = threading.Thread(target=_preload, daemon=True)
    ctx = get_script_run_ctx()
    if ctx is not None:
        add_script_run_ctx(thread, ctx)
    thread.start()
    return thread