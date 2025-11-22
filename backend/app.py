import time
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
from ml_models import (
    ACTIVATIONS,
    build_model_safe,
    elu,
    elu_deriv,
    init_params,
    leaky_relu,
    leaky_relu_deriv,
    load_dataset,
    make_linear_dataset,
    mse,
    run_gradient_descent_linear,
    run_gradient_descent_quadratic,
    compute_contour_grid,
    swish,
    swish_deriv,
    forward
)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from scipy.stats import uniform
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    roc_curve
)

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "msg": "backend running"})

@app.route("/run_linear", methods=["POST"])
def run_linear():
    payload = request.get_json() or {}
    # taking val from fd
    n_samples = payload.get("n_samples", 80)
    noise = payload.get("noise", 8.0)
    x_scale = payload.get("x_scale", 3.0)
    lr = float(payload.get("lr", 0.1))
    epochs = int(payload.get("epochs", 60))
    optimizer = payload.get("optimizer", "Batch")
    batch_size = int(payload.get("batch_size", 16))
    init_theta0 = float(payload.get("init_theta0", 0.0))
    init_theta1 = float(payload.get("init_theta1", 0.0))
    seed = int(payload.get("seed", 42))
    # optional optimizer hyperparams
    momentum_beta = float(payload.get("momentum_beta", 0.9))
    rms_beta = float(payload.get("rms_beta", 0.9))
    adam_beta1 = float(payload.get("adam_beta1", 0.9))
    adam_beta2 = float(payload.get("adam_beta2", 0.999))
    adam_eps = float(payload.get("adam_eps", 1e-8))
    noise_scale = float(payload.get("noise_scale", 0.0))

    X, y, theta_closed = make_linear_dataset(n_samples=n_samples, noise=noise, x_scale=x_scale, seed=seed)
    theta_init = np.array([init_theta0, init_theta1], dtype=float)
    theta_final, history = run_gradient_descent_linear(
        X, y, theta_init, lr, epochs,
        optimizer=optimizer, batch_size=batch_size, seed=seed,
        momentum_beta=momentum_beta, rms_beta=rms_beta,
        adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_eps=adam_eps,
        noise_scale=noise_scale
    )

    response = {
        "theta_final": [float(theta_final[0]), float(theta_final[1])],
        "theta_closed": theta_closed,
        "history": history,
        "data_sample": {"x": X.flatten().tolist(), "y": y.tolist()},
    }
    return jsonify(response)

@app.route("/run_quadratic", methods=["POST"])
def run_quadratic():
    payload = request.get_json() or {}
    theta_init = float(payload.get("theta_init", 4.0))
    lr = float(payload.get("lr", 0.1))
    epochs = int(payload.get("epochs", 60))
    optimizer = payload.get("optimizer", "Batch")
    seed = int(payload.get("seed", 42))
    momentum_beta = float(payload.get("momentum_beta", 0.9))
    rms_beta = float(payload.get("rms_beta", 0.9))
    adam_beta1 = float(payload.get("adam_beta1", 0.9))
    adam_beta2 = float(payload.get("adam_beta2", 0.999))
    adam_eps = float(payload.get("adam_eps", 1e-8))
    noise_scale = float(payload.get("noise_scale", 0.0))

    theta_final, history = run_gradient_descent_quadratic(
        theta_init, lr, epochs, optimizer=optimizer, seed=seed,
        momentum_beta=momentum_beta, rms_beta=rms_beta,
        adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_eps=adam_eps,
        noise_scale=noise_scale
    )
    return jsonify({"theta_final": float(theta_final), "history": history})

@app.route("/contour_grid", methods=["POST"])
def contour_grid():
    """
    Expects:
      {
        "n_samples": 80, "noise": 8.0, "x_scale": 3.0, "seed": 42,
        "theta0_min": -10, "theta0_max": 10,
        "theta1_min": -3, "theta1_max": 3,
        "resolution": 80
      }
    Returns JJ grid and theta0/1 linspaces as lists.
    """
    payload = request.get_json() or {}
    n_samples = payload.get("n_samples", 80)
    noise = payload.get("noise", 8.0)
    x_scale = payload.get("x_scale", 3.0)
    seed = int(payload.get("seed", 42))
    theta0_min = float(payload.get("theta0_min", -10.0))
    theta0_max = float(payload.get("theta0_max", 10.0))
    theta1_min = float(payload.get("theta1_min", -3.0))
    theta1_max = float(payload.get("theta1_max", 3.0))
    resolution = int(payload.get("resolution", 80))

    X, y, _ = make_linear_dataset(n_samples=n_samples, noise=noise, x_scale=x_scale, seed=seed)
    theta0_vals = np.linspace(theta0_min, theta0_max, resolution)
    theta1_vals = np.linspace(theta1_min, theta1_max, resolution)
    T0, T1, JJ = compute_contour_grid(X, y, theta0_vals, theta1_vals)
    return jsonify({
        "theta0_vals": theta0_vals.tolist(),
        "theta1_vals": theta1_vals.tolist(),
        "JJ": JJ.tolist()
    })


## activation functions endpoint
@app.route("/activation", methods=["POST"])
def activation_endpoint():
    req = request.json or {}
    name = req.get("activation", "relu")
    n = int(req.get("n_points", 400))
    mn = float(req.get("input_min", -6))
    mx = float(req.get("input_max", 6))
    alpha = req.get("alpha", None)
    beta = req.get("beta", None)

    x = np.linspace(mn, mx, n)

    if name not in ACTIVATIONS and name not in ("leaky_relu", "elu", "swish"):
        return jsonify({"error": f"Unknown activation '{name}'"}), 400

    # compute with params if required
    if name == "leaky_relu":
        a = float(alpha) if alpha is not None else 0.01
        y = leaky_relu(x, alpha=a)
        dy = leaky_relu_deriv(x, alpha=a)
    elif name == "elu":
        a = float(alpha) if alpha is not None else 1.0
        y = elu(x, alpha=a)
        dy = elu_deriv(x, alpha=a)
    elif name == "swish":
        b = float(beta) if beta is not None else 1.0
        y = swish(x, beta=b)
        dy = swish_deriv(x, beta=b)
    else:
        fn, der = ACTIVATIONS[name]
        y = fn(x)
        dy = der(x)

    return jsonify({"x": x.tolist(), "y": y.tolist(), "dy": dy.tolist()})

@app.route("/train", methods=["POST"])
def train_endpoint():
    req = request.json or {}
    activation_name = req.get("activation", "relu")
    hidden_units = int(req.get("hidden_units", 16))
    epochs = int(req.get("epochs", 200))
    lr = float(req.get("lr", 0.01))
    batch_size = int(req.get("batch_size", 32))
    seed = int(req.get("seed", 42))
    alpha = req.get("alpha", None)
    beta = req.get("beta", None)

    if activation_name not in ACTIVATIONS and activation_name not in ("leaky_relu", "elu", "swish"):
        return jsonify({"error": f"Unknown activation '{activation_name}'"}), 400

    rng = np.random.RandomState(seed)
    N = 256
    X = rng.uniform(-6, 6, size=(N, 1))
    Y = np.sin(X)

    # choose activation wrappers for training
    if activation_name == "leaky_relu":
        a = float(alpha) if alpha is not None else 0.01
        act = lambda z: leaky_relu(z, alpha=a)
        act_deriv = lambda z: leaky_relu_deriv(z, alpha=a)
    elif activation_name == "elu":
        a = float(alpha) if alpha is not None else 1.0
        act = lambda z: elu(z, alpha=a)
        act_deriv = lambda z: elu_deriv(z, alpha=a)
    elif activation_name == "swish":
        b = float(beta) if beta is not None else 1.0
        act = lambda z: swish(z, beta=b)
        act_deriv = lambda z: swish_deriv(z, beta=b)
    else:
        act, act_deriv = ACTIVATIONS[activation_name]

    params = init_params(1, hidden_units, 1, seed=seed)
    losses = []
    loss_epochs = []
    n_batches = int(np.ceil(N / batch_size))

    for ep in range(epochs):
        perm = rng.permutation(N)
        X_sh = X[perm]
        Y_sh = Y[perm]

        for i in range(n_batches):
            start = i * batch_size
            xb = X_sh[start:start + batch_size]
            yb = Y_sh[start:start + batch_size]

            Z1, A1, Z2 = forward(params, xb, act)
            pred = Z2
            dZ2 = (2.0 / xb.shape[0]) * (pred - yb)

            W1, b1, W2, b2 = params
            dW2 = A1.T.dot(dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = dZ2.dot(W2.T)
            dZ1 = dA1 * act_deriv(Z1)

            dW1 = xb.T.dot(dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # gradient descent
            params[0] -= lr * dW1
            params[1] -= lr * db1
            params[2] -= lr * dW2
            params[3] -= lr * db2

        if ep % max(1, epochs // 50) == 0 or ep == epochs - 1:
            _, _, Z2_full = forward(params, X, act)
            cur_loss = mse(Y, Z2_full)
            losses.append(float(cur_loss))
            loss_epochs.append(ep)

    # predictions on test grid
    X_test = np.linspace(-6, 6, 400).reshape(-1, 1)
    _, _, Y_pred = forward(params, X_test, act)

    return jsonify({
        "loss_epochs": loss_epochs,
        "losses": losses,
        "x_test": X_test.flatten().tolist(),
        "y_pred": Y_pred.flatten().tolist(),
        "y_true": np.sin(X_test).flatten().tolist()
    })




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
