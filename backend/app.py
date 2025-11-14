# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from ml_models import (
    make_linear_dataset,
    run_gradient_descent_linear,
    run_gradient_descent_quadratic,
    compute_contour_grid,
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
