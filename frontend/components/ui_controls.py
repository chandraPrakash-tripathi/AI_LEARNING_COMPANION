import streamlit as st

def render_controls_stepplay():
    st.session_state.setdefault("playing", False)
    st.session_state.setdefault("step_idx", 0)
    st.session_state.setdefault("thetas", None)

    st.subheader("Playback Controls")

    # ONE LEVEL OF COLUMNS ONLY
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if st.button("▶️"):
            st.session_state.playing = True

    with c2:
        if st.button("⏸"):
            st.session_state.playing = False

    with c3:
        if st.button("←"):
            if st.session_state.thetas is not None:
                st.session_state.step_idx = max(st.session_state.step_idx - 1, 0)

    with c4:
        if st.button("→"):
            if st.session_state.thetas is not None:
                st.session_state.step_idx = min(
                    st.session_state.step_idx + 1,
                    len(st.session_state.thetas) - 1
                )

    with c5:
        if st.button("🔄"):
            st.session_state.step_idx = 0

    st.write("**Current Step Index:**", st.session_state.step_idx)

    if st.session_state.thetas is not None:
        max_steps = len(st.session_state.thetas) - 1
        st.session_state.step_idx = st.slider(
            "Scrub timeline",
            0,
            max_steps,
            st.session_state.step_idx,
            key="scrub_slider",
        )
