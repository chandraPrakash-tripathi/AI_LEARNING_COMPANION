import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

st.set_page_config(page_title="Activation Fucntion Playground", layout="wide")


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
    # mish: x * tanh(softplus(x)) where softplus = ln(1+e^x)
    soft = np.log1p(np.exp(x))
    return x * np.tanh(soft)

def mish_deriv(x):
    # Approximate derivative (good enough for demonstration)
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

# -------------------- UI --------------------



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
    train_button = st.button("Train toy model")

# -------------------- Plot activation and derivative --------------------

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

col1, col2 = st.columns([1, 1])
with col1:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, y)
    ax.set_title(f"{activation_name} — value")
    ax.grid(True)
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    if show_derivative:
        ax2.plot(x, dy)
        ax2.set_title(f"{activation_name} — derivative")
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, "Derivative hidden (toggle from sidebar)", ha='center')
    st.pyplot(fig2)

# -------------------- Sample values table --------------------

st.subheader("Sample inputs → activation outputs")
xs = np.linspace(input_min, input_max, sample_points)
outs = act(xs)

sample_table = np.vstack([xs, outs]).T
st.dataframe({"x": xs, f"{activation_name}(x)": outs})

# -------------------- Toy NN (single hidden layer) --------------------

st.markdown("---")
st.write("This trains a tiny 1-hidden-layer network (NumPy) using the selected activation in the hidden layer. Predictions and loss curve will be shown.")

# Prepare dataset
rng = np.random.RandomState(42)
N = 256
X = rng.uniform(-6, 6, size=(N, 1))
Y = np.sin(X)

# simple utilities

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def forward(params, X_batch, activation_hidden):
    W1, b1, W2, b2 = params
    Z1 = X_batch.dot(W1) + b1  # (B, H)
    A1 = activation_hidden(Z1)
    Z2 = A1.dot(W2) + b2      # (B, 1)
    return Z1, A1, Z2


def init_params(in_dim, hidden_dim, out_dim, scale=0.1):
    rng = np.random.RandomState(1)
    W1 = rng.randn(in_dim, hidden_dim) * scale
    b1 = np.zeros((1, hidden_dim))
    W2 = rng.randn(hidden_dim, out_dim) * scale
    b2 = np.zeros((1, out_dim))
    return [W1, b1, W2, b2]

# training loop (vectorized)

params = init_params(1, hidden_units, 1)

if train_button:
    losses = []
    n_batches = int(np.ceil(N / batch_size))

    for ep in range(epochs):
        # shuffle
        perm = rng.permutation(N)
        X_sh = X[perm]
        Y_sh = Y[perm]

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            xb = X_sh[start:end]
            yb = Y_sh[start:end]

            # forward
            Z1, A1, Z2 = forward(params, xb, act)

            # loss and gradient at output (MSE)
            pred = Z2
            loss = mse(yb, pred)

            # gradients (manual backprop)
            dZ2 = (2.0 / xb.shape[0]) * (pred - yb)  # (B,1)
            W1, b1, W2, b2 = params

            dW2 = A1.T.dot(dZ2)                      # (H,1)
            db2 = np.sum(dZ2, axis=0, keepdims=True) # (1,1)

            # backprop into hidden
            dA1 = dZ2.dot(W2.T)                      # (B,H)
            dZ1 = dA1 * act_deriv(Z1)                # (B,H)

            dW1 = xb.T.dot(dZ1)                      # (1,H)
            db1 = np.sum(dZ1, axis=0, keepdims=True) # (1,H)

            # gradient descent step
            params[0] -= lr * dW1
            params[1] -= lr * db1
            params[2] -= lr * dW2
            params[3] -= lr * db2

        # evaluate on full set every 10 epochs
        if (ep % max(1, epochs // 20)) == 0 or ep == epochs - 1:
            _, _, Z2_full = forward(params, X, act)
            cur_loss = mse(Y, Z2_full)
            losses.append(cur_loss)

    # After training show results
    st.subheader("Training finished")
    st.write(f"Final MSE: {losses[-1]:.6f}")

    # show loss curve
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.plot(np.linspace(0, epochs, len(losses)), losses)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("MSE")
    ax3.set_title("Loss curve (sampled points)")
    st.pyplot(fig3)

    # show predictions
    X_test = np.linspace(-6, 6, 400).reshape(-1, 1)
    _, _, Y_pred = forward(params, X_test, act)

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.scatter(X.flatten(), Y.flatten(), s=10, label="train data")
    ax4.plot(X_test.flatten(), Y_pred.flatten(), label="model pred", linewidth=2)
    ax4.plot(X_test.flatten(), np.sin(X_test).flatten(), label="true sin(x)", linestyle='--')
    ax4.legend()
    ax4.set_title("Model fit")
    st.pyplot(fig4)

else:
    st.info("Click 'Train toy model' in the sidebar to train a small network using the selected activation.")



