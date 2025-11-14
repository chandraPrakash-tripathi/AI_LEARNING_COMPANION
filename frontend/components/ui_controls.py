import streamlit as st

def render_controls_stepplay():
    control_col, vis_col = st.columns([1, 3])
    with control_col:
        st.subheader("Playback")
        if st.button("Play ▶️"):
            st.session_state.playing = True
        if st.button("Pause ⏸"):
            st.session_state.playing = False
        if st.button("Step ←"):
            if st.session_state.thetas is not None:
                st.session_state.step_idx = max(st.session_state.step_idx - 1, 0)
        if st.button("Step →"):
            if st.session_state.thetas is not None:
                st.session_state.step_idx = min(st.session_state.step_idx + 1, len(st.session_state.thetas) - 1)
        if st.button("Reset"):
            st.session_state.step_idx = 0
        st.write("Current step index:", st.session_state.step_idx)
        max_steps = len(st.session_state.thetas) - 1 if st.session_state.thetas is not None else 0
        if max_steps > 0:
            idx = st.slider("Scrub steps", 0, max_steps, st.session_state.step_idx, key="scrub")
            st.session_state.step_idx = idx
    return vis_col