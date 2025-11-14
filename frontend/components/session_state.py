import streamlit as st


def ensure_session_state_keys():
    if "history" not in st.session_state:
        st.session_state.history = None
    if "step_idx" not in st.session_state:
        st.session_state.step_idx = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "thetas" not in st.session_state:
        st.session_state.thetas = None
    if "losses" not in st.session_state:
        st.session_state.losses = None
    if "grads" not in st.session_state:
        st.session_state.grads = None
    if "theta_closed" not in st.session_state:
        st.session_state.theta_closed = None
    if "data" not in st.session_state:
        st.session_state.data = None
    if "contour" not in st.session_state:
        st.session_state.contour = None