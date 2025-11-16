import time
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
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

## Hyperparameter tuning endpoint
@app.route("/train", methods=["POST"])
def train():
    start = time.time()
    try:
        payload = request.get_json(force=True)
        dataset_spec = payload.get("dataset", "Iris")
        X, y, feature_names = load_dataset(dataset_spec)
        model_name = payload.get("model", "LogisticRegression")
        hyperparams = payload.get("hyperparams", {}) or {}
        prep = payload.get("preprocessing", {}) or {}
        test_size = float(payload.get("test_size", 0.2))
        random_state = int(payload.get("random_state", 42))
        cv_folds = int(payload.get("cv_folds", 1))
        search = payload.get("search", {"mode": "none"})
        seeds = payload.get("seeds", [random_state])

        results = []
        grid_heatmap = None

        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y if len(np.unique(y))>1 else None
            )

            # preprocessing
            deg = int(prep.get("poly_degree", 1) or 1)
            if deg > 1:
                pf = PolynomialFeatures(deg, include_bias=False)
                X_train = pf.fit_transform(X_train)
                X_test = pf.transform(X_test)

            if prep.get("scale") == "standard":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            elif prep.get("scale") == "minmax":
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if search and search.get("mode") in ("grid", "random"):
                mode = search.get("mode")
                param_grid = search.get("param_grid", {})
                base_model = build_model_safe(model_name, hyperparams)
                if mode == "grid":
                    gs = GridSearchCV(base_model, param_grid=param_grid, cv=max(2, min(5, cv_folds or 3)))
                    gs.fit(X_train, y_train)
                    best = gs.best_estimator_
                    best_params = gs.best_params_
                    # heatmap for exactly two params
                    if len(param_grid.keys()) == 2:
                        keys = list(param_grid.keys())
                        vals1 = param_grid[keys[0]]
                        vals2 = param_grid[keys[1]]
                        heat = []
                        for v1 in vals1:
                            row = []
                            for v2 in vals2:
                                params = dict(best_params)
                                params.update({keys[0]: v1, keys[1]: v2})
                                m = build_model_safe(model_name, params)
                                m.fit(X_train, y_train)
                                row.append(m.score(X_test, y_test))
                            heat.append(row)
                        grid_heatmap = {
                            "x_name": keys[0], "y_name": keys[1],
                            "x_vals": list(map(str, vals1)), "y_vals": list(map(str, vals2)),
                            "scores": heat
                        }
                else:
                    # RandomizedSearchCV: pass param_distributions directly (lists are fine)
                    rs = RandomizedSearchCV(build_model_safe(model_name, hyperparams), param_distributions=param_grid,
                                            n_iter=int(search.get("n_iter", 20)), cv=max(2, min(5, cv_folds or 3)), random_state=seed)
                    rs.fit(X_train, y_train)
                    best = rs.best_estimator_
                    best_params = rs.best_params_
            else:
                best = build_model_safe(model_name, hyperparams)
                best.fit(X_train, y_train)
                best_params = hyperparams or {}

            train_score = best.score(X_train, y_train)
            val_score = best.score(X_test, y_test)

            metrics = {"train_accuracy": float(train_score), "val_accuracy": float(val_score), "params": best_params}
            try:
                y_pred = best.predict(X_test)
                metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
            except Exception:
                metrics["confusion_matrix"] = None

            try:
                if hasattr(best, "predict_proba"):
                    probs = best.predict_proba(X_test)
                    if probs.shape[1] == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, probs[:, 1]))
                        fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
                        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                    else:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, probs, multi_class="ovr"))
                else:
                    metrics["roc_auc"] = None
            except Exception:
                metrics["roc_auc"] = None

            try:
                if hasattr(best, "feature_importances_"):
                    metrics["feature_importances"] = dict(zip(feature_names, list(map(float, best.feature_importances_[:len(feature_names)]))))
                elif hasattr(best, "coef_"):
                    coef = np.array(best.coef_)
                    if coef.ndim > 1:
                        coef = coef.mean(axis=0)
                    metrics["coefficients"] = dict(zip(feature_names, list(map(float, coef[:len(feature_names)]))))
            except Exception:
                pass

            results.append(metrics)

        resp = {"status": "ok", "time_taken": time.time() - start, "results": results}
        if grid_heatmap:
            resp["grid_heatmap"] = grid_heatmap
        return jsonify(resp)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"status": "error", "message": str(e), "traceback": tb}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
