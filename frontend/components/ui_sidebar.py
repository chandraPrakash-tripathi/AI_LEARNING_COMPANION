import streamlit as st

def render_sidebar(state):


    # ROW 1 — Mode / Optimizer / Seed
    col1, col2, col3 = st.columns(3)
    with col1:
        mode = st.selectbox("Mode", ["Linear Regression", "Quadratic (1D)"])
    with col2:
        optimizer = st.selectbox("Optimizer", ["Batch", "SGD", "Mini-batch", "Momentum", "RMSProp", "Adam"])
    with col3:
        seed = st.number_input("Seed", min_value=0, value=42)

    # ROW 2 — LR / Epochs
    col4, col5 = st.columns(2)
    with col4:
        lr = st.number_input("LR (α)", value=0.1, format="%.5f", step=0.01)
    with col5:
        epochs = st.number_input("Epochs", min_value=1, max_value=2000, value=60)

    sidebar_values = {
        "mode": mode,
        "optimizer": optimizer,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
    }

    # OPTIMIZER PARAMS (COMPACT IN EXPANDER)
    with st.expander("Optimizer Hyperparams", expanded=False):
        col6, col7, col8, col9, col10 = st.columns(5)

        with col6:
            momentum_beta = st.number_input("mom β", value=0.9, format="%.3f")
        with col7:
            rms_beta = st.number_input("rms β", value=0.9, format="%.3f")
        with col8:
            adam_beta1 = st.number_input("β1", value=0.9, format="%.3f")
        with col9:
            adam_beta2 = st.number_input("β2", value=0.999, format="%.4f")
        with col10:
            adam_eps = st.number_input("eps", value=1e-8, format="%.8f")

        sidebar_values.update({
            "momentum_beta": momentum_beta,
            "rms_beta": rms_beta,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "adam_eps": adam_eps,
        })


    # MODE SECTIONS (COMPACT)
    if mode == "Linear Regression":
        with st.expander("Dataset & Init", expanded=True):
            colA, colB, colC = st.columns(3)
            with colA:
                n_samples = st.slider("Samples", 10, 1000, 80)
            with colB:
                noise = st.slider("Noise", 0.0, 30.0, 8.0)
            with colC:
                x_scale = st.slider("Scale", 1.0, 10.0, 3.0)

            colD, colE, colF = st.columns(3)
            with colD:
                batch_size = (
                    1 if optimizer == "SGD" else
                    st.slider("Batch", 1, 100, 16)
                )
            with colE:
                init_theta0 = st.number_input("θ0", value=0.0, format="%.3f")
            with colF:
                init_theta1 = st.number_input("θ1", value=0.0, format="%.3f")

        sidebar_values.update({
            "n_samples": n_samples,
            "noise": noise,
            "x_scale": x_scale,
            "batch_size": batch_size,
            "init_theta0": init_theta0,
            "init_theta1": init_theta1,
        })

    else:
        with st.expander("Initialization", expanded=True):
            colG, colH = st.columns(2)
            with colG:
                theta_init = st.number_input("θ init", value=4.0, format="%.3f")
            with colH:
                noise_scale = st.number_input("Noise scale", value=0.0, format="%.3f")

        sidebar_values.update({
            "theta_init": theta_init,
            "noise_scale": noise_scale
        })

    # RUN BUTTON
    run_button = st.button("Run ▶️")
    sidebar_values["run_button"] = run_button

    return sidebar_values
