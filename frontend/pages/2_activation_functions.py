# app.py
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from functools import partial
import requests
import json

# ---------- Config ----------
BACKEND = "http://localhost:5000"  # change to your backend URL if needed

st.set_page_config(page_title="Activation Function Playground", layout="wide")

# ---------- Activation functions ----------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x >= 0, 1.0, alpha)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def elu_deriv(x, alpha=1.0):
    return np.where(x >= 0, 1.0, alpha * np.exp(x))

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def swish_deriv(x, beta=1.0):
    s = sigmoid(beta * x)
    return s + beta * x * s * (1 - s)

def mish(x):
    soft = np.log1p(np.exp(x))
    return x * np.tanh(soft)

def mish_deriv(x):
    soft = np.log1p(np.exp(x))
    tanh_soft = np.tanh(soft)
    sigmoid_x = sigmoid(x)
    return tanh_soft + x * (1 - tanh_soft**2) * sigmoid_x

ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
    "relu": (relu, relu_deriv),
    "leaky_relu": (leaky_relu, leaky_relu_deriv),
    "elu": (elu, elu_deriv),
    "swish": (swish, swish_deriv),
    "mish": (mish, mish_deriv),
}

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")
    activation_name = st.selectbox("Activation", list(ACTIVATIONS.keys()), index=3)
    input_min, input_max = st.slider("Plot range (x)", -10.0, 10.0, (-6.0, 6.0), step=0.5)
    show_derivative = st.checkbox("Show derivative", True)
    sample_points = st.slider("Number of sample points to show in table", 5, 50, 11)

    # Activation-specific params
    alpha = None
    beta = None
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
    train_button = st.button("Train toy model (backend)")

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

# ---------- Helper to make Altair charts easily ----------
def alt_line_chart(df, x_col, y_col, title=None, width=None, height=300):
    chart = alt.Chart(df).mark_line().encode(
        x=x_col, y=y_col, tooltip=[x_col, y_col]
    )
    props = {}
    if title is not None:
        props["title"] = title
    if width is not None:
        props["width"] = width
    if height is not None:
        props["height"] = height
    if props:
        chart = chart.properties(**props)
    return chart


# ---------- Layout: Activation + Derivative ----------
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader(f"{activation_name} — value")
    df_val = pd.DataFrame({"x": x, f"{activation_name}(x)": y})
    chart_val = alt_line_chart(df_val, "x", f"{activation_name}(x)")
    st.altair_chart(chart_val, use_container_width=True)

with col2:
    st.subheader(f"{activation_name} — derivative")
    if show_derivative:
        df_der = pd.DataFrame({"x": x, "derivative": dy})
        chart_der = alt_line_chart(df_der, "x", "derivative")
        st.altair_chart(chart_der, use_container_width=True)
    else:
        st.info("Derivative hidden (toggle from sidebar)")

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
st.write("This will call the backend to train a tiny 1-hidden-layer network using the selected activation in the hidden layer. Predictions and loss curve returned by backend will be shown here.")

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
            resp = requests.post(f"{BACKEND}/train", json=payload, timeout=600)
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
        st.subheader("Training finished (backend)")
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
    st.info("Click 'Train toy model (backend)' in the sidebar to train a small network using the selected activation.")
