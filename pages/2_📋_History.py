import streamlit as st
import pandas as pd
import sys
sys.path.append("src")
from styles import PAGE_STYLE, init_session_state

st.set_page_config(page_title="History — HistoVision AI", layout="wide")
st.markdown(PAGE_STYLE, unsafe_allow_html=True)
init_session_state()

st.markdown('<div class="main-header">History</div>', unsafe_allow_html=True)
st.caption("Every prediction made this session.")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Images analyzed", len(df))
    c2.metric("Model agreement rate", f"{(df['Agree'] == 'Yes').mean()*100:.0f}%")
    c3.metric("OOD flags", int((df["OOD"] == "Yes").sum()))

    st.download_button("Download full session history (CSV)", data=df.to_csv(index=False),
                        file_name="histovision_session_history.csv", mime="text/csv")

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No predictions yet this session. Go to the Analyze page to get started.")