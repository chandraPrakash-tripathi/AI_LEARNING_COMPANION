# app.py
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from functools import partial
import requests
import json
from components.constants import BACKEND_URL
from components.activations import ACTIVATIONS, elu, elu_deriv, leaky_relu, leaky_relu_deriv, swish, swish_deriv
from components.viz import alt_line_chart, render_activation_plots
from components.ui_sidebar import render_toolbar


st.set_page_config(page_title="Activation Function Playground", layout="wide")

#toolbar
controls = render_toolbar()
activation_name = controls["activation_name"]
input_min, input_max = controls["input_min"], controls["input_max"]
show_derivative = controls["show_derivative"]
sample_points = controls["sample_points"]
alpha, beta = controls["alpha"], controls["beta"]
hidden_units = controls["hidden_units"]
epochs = controls["epochs"]
lr = controls["lr"]
batch_size = controls["batch_size"]
train_button = controls["train_button"]


#using toolbar inputs
# ---------- Prepare functions and data (local for plots) ----------
x = np.linspace(input_min, input_max, 400)
act_fn, act_deriv_fn = ACTIVATIONS[activation_name]

# wrap functions with params if needed
if activation_name == "leaky_relu":
    act = partial(leaky_relu, alpha=alpha)
    act_deriv = partial(leaky_relu_deriv, alpha=alpha)
elif activation_name == "elu":
    act = partial(elu, alpha=alpha)
    act_deriv = partial(elu_deriv, alpha=alpha)
elif activation_name == "swish":
    act = partial(swish, beta=beta)
    act_deriv = partial(swish_deriv, beta=beta)
else:
    act = act_fn
    act_deriv = act_deriv_fn

y = act(x)
dy = act_deriv(x)

#using viz plot
render_activation_plots(
    activation_name=activation_name,
    x=x,
    y=y,
    dy=dy,
    show_derivative=show_derivative
)

# ---------- Sample values table ----------
st.subheader("Sample inputs → activation outputs")
xs = np.linspace(input_min, input_max, sample_points)
outs = act(xs)
sample_df = pd.DataFrame({"x": xs, f"{activation_name}(x)": outs})
st.dataframe(sample_df)

# ---------- Toy NN (single hidden layer) dataset (local, for plotting training results) ----------
rng = np.random.RandomState(42)
N = 256
X = rng.uniform(-6, 6, size=(N, 1))
Y = np.sin(X)

# ---------- Train: call backend ----------
st.markdown("---")
st.write("This will  train a tiny 1-hidden-layer network using the selected activation in the hidden layer.")

if train_button:
    payload = {
        "activation": activation_name,
        "hidden_units": int(hidden_units),
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        # include activation params if any
        "alpha": float(alpha) if alpha is not None else None,
        "beta": float(beta) if beta is not None else None,
        "seed": 42
    }

    # call backend
    with st.spinner("Sending training job to backend..."):
        try:
            resp = requests.post(f"{BACKEND_URL}/train", json=payload, timeout=600)
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()

    if resp.status_code == 200:
        try:
            data = resp.json()
        except json.JSONDecodeError:
            st.error("Backend did not return valid JSON.")
            st.stop()

        # show results
        if "losses" in data and len(data["losses"]) > 0:
            st.write(f"Sampled losses: {data['losses'][:5]} ... (final {data['losses'][-1]:.6f})")

            loss_df = pd.DataFrame({"epoch": data["loss_epochs"], "mse": data["losses"]})
            st.altair_chart(alt_line_chart(loss_df, "epoch", "mse", title="Loss curve"), use_container_width=True)
        else:
            st.info("No loss data returned by backend.")

        if all(k in data for k in ("x_test", "y_pred", "y_true")):
            pred_df = pd.DataFrame({
                "x": data["x_test"],
                "y_pred": data["y_pred"],
                "y_true": data["y_true"]
            })

            # original training points
            train_df = pd.DataFrame({"x": X.flatten(), "y": Y.flatten()})
            points = alt.Chart(train_df).mark_circle(size=20).encode(
                x="x:Q",
                y="y:Q",
                tooltip=["x", "y"]
            )

            pred_line = alt.Chart(pred_df).mark_line().encode(
                x="x:Q",
                y="y_pred:Q",
                tooltip=["x", "y_pred"]
            )

            true_line = alt.Chart(pred_df).mark_line(strokeDash=[5,5]).encode(
                x="x:Q",
                y="y_true:Q",
                tooltip=["x", "y_true"]
            )

            layered = alt.layer(points, pred_line, true_line).properties(title="Model fit")
            st.altair_chart(layered, use_container_width=True)

            # also show small preview table
            st.subheader("Predictions (sample)")
            st.dataframe(pred_df.head(10))
        else:
            st.info("No predictions returned by backend.")
    else:
        st.error(f"Training failed: {resp.status_code} {resp.text}")

else:
    st.info("Click Train toy model")
