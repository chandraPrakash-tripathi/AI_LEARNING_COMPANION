import streamlit as st
from functools import partial
from typing import Optional, Dict, Any
from components.activations import ACTIVATIONS

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





def render_toolbar() -> Dict[str, Any]:
    """Render the sidebar controls and return a dict of values.

    Usage:
        from components.ui_sidebar import render_sidebar
        controls = render_sidebar()
        activation_name = controls['activation_name']
        # ...

    Returns a dictionary with the following keys:
      - activation_name (str)
      - input_min (float)
      - input_max (float)
      - show_derivative (bool)
      - sample_points (int)
      - alpha (Optional[float])
      - beta (Optional[float])
      - hidden_units (int)
      - epochs (int)
      - lr (float)
      - batch_size (int)
      - train_button (bool)

    This isolates all sidebar UI logic so app.py can remain focused on layout
    and plotting.
    """

    with st.sidebar:
        st.header("Controls")
        activation_name = st.selectbox("Activation", list(ACTIVATIONS.keys()), index=3)
        input_min, input_max = st.slider("Plot range (x)", -10.0, 10.0, (-6.0, 6.0), step=0.5)
        show_derivative = st.checkbox("Show derivative", True)
        sample_points = st.slider("Number of sample points to show in table", 5, 50, 11)

        # Activation-specific params
        alpha: Optional[float] = None
        beta: Optional[float] = None
        if activation_name == "leaky_relu":
            alpha = st.slider("Leaky ReLU alpha", 0.001, 0.5, 0.01)
        if activation_name == "elu":
            alpha = st.slider("ELU alpha", 0.1, 3.0, 1.0)
        if activation_name == "swish":
            beta = st.slider("Swish beta", 0.1, 5.0, 1.0)

        st.markdown("---")
        st.subheader("Toy NN settings")
        hidden_units = st.slider("Hidden units (single hidden layer)", 1, 128, 16)
        epochs = st.slider("Training epochs", 1, 1000, 200)
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=0.01, format="%.6f")
        batch_size = st.slider("Batch size", 4, 128, 32)
        train_button = st.button("Train toy model")

    return {
        "activation_name": activation_name,
        "input_min": input_min,
        "input_max": input_max,
        "show_derivative": show_derivative,
        "sample_points": sample_points,
        "alpha": alpha,
        "beta": beta,
        "hidden_units": hidden_units,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "train_button": train_button,
    }




    