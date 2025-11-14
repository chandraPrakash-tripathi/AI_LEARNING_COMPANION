# backend/ml_models.py
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# ---------- basic losses / grads ----------
def quadratic_loss(theta):
    return 0.5 * (theta ** 2)

def quadratic_grad(theta):
    return theta

def compute_linear_loss_and_grad(theta, X, y):
    """
    theta: array([theta0, theta1, theta2...]) where theta[1:] correspond to features
    X: shape (m, n_features)
    y: shape (m,)
    """
    m = X.shape[0]
    preds = theta[0] + X.dot(theta[1:])
    residuals = preds - y
    J = (1.0 / (2 * m)) * np.sum(residuals ** 2)
    grad0 = (1.0 / m) * np.sum(residuals)
    grad_rest = (1.0 / m) * (X.T.dot(residuals))
    grad = np.concatenate(([grad0], grad_rest))
    return J, grad

# ---------- dataset helper ----------
def make_linear_dataset(n_samples=80, noise=8.0, x_scale=3.0, seed=42):
    X_raw, y = make_regression(n_samples=int(n_samples), n_features=1, noise=float(noise), random_state=int(seed), bias=0.0)
    X_raw = X_raw.flatten() * float(x_scale)
    X = X_raw.reshape(-1, 1)
    lr_model = LinearRegression().fit(X, y)
    theta_closed = [float(lr_model.intercept_), float(lr_model.coef_[0])]
    return X, y, theta_closed

# ---------- optimizer implementations ----------
def run_gradient_descent_linear(
    X, y, theta_init, lr, epochs,
    optimizer="Batch", batch_size=32, seed=42,
    momentum_beta=0.9, rms_beta=0.9, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
    noise_scale=0.0
):
    """
    Supports optimizer in: Batch, SGD, Mini-batch, Momentum, RMSProp, Adam
    For Momentum/RMSProp/Adam we run in Batch mode per-epoch by default (but will also accept per-batch updates if SGD/Mini-batch).
    Returns final theta and history {theta: [...], loss: [...]}
    """
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    theta = theta_init.copy().astype(float)
    history = {"theta": [], "loss": []}

    # initialize momentum / rmsprop / adam states
    v = np.zeros_like(theta)          # for momentum (velocity) or adam first moment
    s = np.zeros_like(theta)          # for RMSProp (squared gradient) or adam second moment
    t_step = 0

    def batch_update(theta, grad, lr, opt):
        nonlocal v, s, t_step
        t_step += 1
        if opt == "Momentum":
            v = momentum_beta * v + (1 - momentum_beta) * grad
            theta = theta - lr * v
        elif opt == "RMSProp":
            s = rms_beta * s + (1 - rms_beta) * (grad ** 2)
            theta = theta - lr * grad / (np.sqrt(s) + 1e-8)
        elif opt == "Adam":
            v = adam_beta1 * v + (1 - adam_beta1) * grad
            s = adam_beta2 * s + (1 - adam_beta2) * (grad ** 2)
            v_hat = v / (1 - adam_beta1 ** t_step)
            s_hat = s / (1 - adam_beta2 ** t_step)
            theta = theta - lr * v_hat / (np.sqrt(s_hat) + adam_eps)
        else:
            theta = theta - lr * grad
        return theta

    for epoch in range(int(epochs)):
        if optimizer == "Batch" or optimizer in ("Momentum", "RMSProp", "Adam"):
            # compute full-batch gradient
            J, grad = compute_linear_loss_and_grad(theta, X, y)
            if noise_scale > 0.0:
                grad = grad + rng.normal(scale=noise_scale, size=grad.shape)
            theta = batch_update(theta, grad, lr, optimizer if optimizer in ("Momentum","RMSProp","Adam") else "BGD")
            history["theta"].append(theta.copy().tolist())
            history["loss"].append(float(J))
        else:
            # SGD or Mini-batch: iterate over shuffled batches
            idx = np.arange(m)
            rng.shuffle(idx)
            batches = [idx[i:i+batch_size] for i in range(0, m, batch_size)]
            for b in batches:
                Xb = X[b]
                yb = y[b]
                Jb, gradb = compute_linear_loss_and_grad(theta, Xb, yb)
                if noise_scale > 0.0:
                    gradb = gradb + rng.normal(scale=noise_scale, size=gradb.shape)
                # For SGD/Mini-batch, support applying momentum/rms/adam per-update if requested by setting optimizer accordingly
                if optimizer in ("Momentum", "RMSProp", "Adam"):
                    theta = batch_update(theta, gradb, lr, optimizer)
                else:
                    theta = theta - lr * gradb
                history["theta"].append(theta.copy().tolist())
                history["loss"].append(float(Jb))

    return theta.tolist(), history

def run_gradient_descent_quadratic(theta_init, lr, epochs, optimizer="Batch", seed=42,
                                   momentum_beta=0.9, rms_beta=0.9, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8, noise_scale=0.0):
    rng = np.random.default_rng(seed)
    theta = float(theta_init)
    history = {"theta": [], "loss": []}

    v = 0.0
    s = 0.0
    t_step = 0

    for epoch in range(int(epochs)):
        grad = quadratic_grad(theta)
        if noise_scale > 0.0:
            grad = grad + rng.normal(scale=noise_scale)
        t_step += 1
        if optimizer == "Momentum":
            v = momentum_beta * v + (1 - momentum_beta) * grad
            theta = theta - lr * v
        elif optimizer == "RMSProp":
            s = rms_beta * s + (1 - rms_beta) * (grad ** 2)
            theta = theta - lr * grad / (np.sqrt(s) + 1e-8)
        elif optimizer == "Adam":
            v = adam_beta1 * v + (1 - adam_beta1) * grad
            s = adam_beta2 * s + (1 - adam_beta2) * (grad ** 2)
            v_hat = v / (1 - adam_beta1 ** t_step)
            s_hat = s / (1 - adam_beta2 ** t_step)
            theta = theta - lr * v_hat / (np.sqrt(s_hat) + adam_eps)
        else:
            theta = theta - lr * grad

        history["theta"].append(float(theta))
        history["loss"].append(float(quadratic_loss(theta)))

    return float(theta), history

# ---------- contour grid (loss surface) ----------
def compute_contour_grid(X, y, theta0_vals, theta1_vals):
    """
    Compute loss J for each (theta0, theta1) pair.
    theta0_vals, theta1_vals are 1D arrays (linspace).
    Returns grid JJ shaped (len(theta1_vals), len(theta0_vals)) so that
    plotting with meshgrid(T0, T1) works similarly to earlier frontend.
    NOTE: this is CPU heavy for large grids; keep resolution small (e.g., 60-200).
    """
    T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
    JJ = np.zeros_like(T0, dtype=float)
    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            t = np.array([T0[i, j], T1[i, j]])
            Jtmp, _ = compute_linear_loss_and_grad(t, X, y)
            JJ[i, j] = Jtmp
    return T0, T1, JJ
