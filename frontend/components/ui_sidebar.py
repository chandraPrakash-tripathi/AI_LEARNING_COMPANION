import streamlit as st


def render_sidebar(state):
    st.set_page_config(layout="wide", page_title="AI LEARNING COMPANION")

    mode = st.sidebar.selectbox("Mode", ["Linear Regression", "Quadratic (1D)"])
    optimizer = st.sidebar.selectbox("Optimizer", ["Batch", "SGD", "Mini-batch", "Momentum", "RMSProp", "Adam"])
    lr = st.sidebar.number_input("Learning rate (α)", value=0.1, format="%.5f", step=0.01)
    epochs = st.sidebar.number_input("Epochs (or steps)", min_value=1, max_value=2000, value=60)
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42)

    # optimizer hyperparams
    st.sidebar.subheader("Optimizer hyperparams")
    momentum_beta = st.sidebar.number_input("momentum beta (for Momentum, Adam)", value=0.9, format="%.3f")
    rms_beta = st.sidebar.number_input("rms beta (for RMSProp)", value=0.9, format="%.3f")
    adam_beta1 = st.sidebar.number_input("adam beta1", value=0.9, format="%.3f")
    adam_beta2 = st.sidebar.number_input("adam beta2", value=0.999, format="%.4f")
    adam_eps = st.sidebar.number_input("adam eps", value=1e-8, format="%.8f")

    sidebar_values = {
        "mode": mode,
        "optimizer": optimizer,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "momentum_beta": momentum_beta,
        "rms_beta": rms_beta,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "adam_eps": adam_eps,
    }

    if mode == "Linear Regression":
        st.sidebar.subheader("Dataset")
        n_samples = st.sidebar.slider("Samples", 10, 1000, 80)
        noise = st.sidebar.slider("Noise (std dev)", 0.0, 30.0, 8.0)
        x_scale = st.sidebar.slider("Feature scale", 1.0, 10.0, 3.0)
        batch_size = 1 if optimizer == "SGD" else st.sidebar.slider("Mini-batch size (for Mini-batch)", 1, 100, 16)
        st.sidebar.write("---")
        st.sidebar.subheader("Initialization")
        init_theta0 = st.sidebar.number_input("θ0 (intercept) init", value=0.0, format="%.3f")
        init_theta1 = st.sidebar.number_input("θ1 (slope) init", value=0.0, format="%.3f")
        sidebar_values.update({
            "n_samples": n_samples,
            "noise": noise,
            "x_scale": x_scale,
            "batch_size": batch_size,
            "init_theta0": init_theta0,
            "init_theta1": init_theta1
        })
    else:
        st.sidebar.subheader("Initialization")
        theta_init = st.sidebar.number_input("θ init", value=4.0, format="%.3f")
        noise_scale = st.sidebar.number_input("SGD noise scale (for demo)", value=0.0, format="%.3f")
        sidebar_values.update({
            "theta_init": theta_init,
            "noise_scale": noise_scale
        })

    run_button = st.sidebar.button("Run ▶️")
    sidebar_values["run_button"] = run_button
    return sidebar_values










