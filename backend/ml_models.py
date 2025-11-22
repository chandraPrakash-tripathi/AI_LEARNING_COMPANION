# backend/ml_models.py
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
# ---------- basic losses / grads ----------
def quadratic_loss(theta):
    return 0.5 * (theta ** 2)

def quadratic_grad(theta):
    return theta

def compute_linear_loss_and_grad(theta, X, y):
    """
    theta: array([theta0, theta1, ...])
    X: shape (m, n_features)
    y: shape (m,)
    Returns: J (float), grad (ndarray), preds (ndarray)
    """
    m = X.shape[0]
    preds = theta[0] + X.dot(theta[1:])
    residuals = preds - y
    J = (1.0 / (2 * m)) * np.sum(residuals ** 2)
    grad0 = (1.0 / m) * np.sum(residuals)
    grad_rest = (1.0 / m) * (X.T.dot(residuals))
    grad = np.concatenate(([grad0], grad_rest))
    return J, grad, preds

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
    Supports Batch, SGD, Mini-batch, Momentum, RMSProp, Adam.
    History now includes 'grad' and (for linear) 'preds' at each recorded update step.
    """
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    theta = theta_init.copy().astype(float)
    history = {"theta": [], "loss": [], "grad": [], "preds": []}

    # optimizer states
    v = np.zeros_like(theta)
    s = np.zeros_like(theta)
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
        if optimizer in ("Batch", "Momentum", "RMSProp", "Adam"):
            # full-batch evaluation
            J, grad, preds = compute_linear_loss_and_grad(theta, X, y)
            if noise_scale > 0.0:
                grad = grad + rng.normal(scale=noise_scale, size=grad.shape)
            theta = batch_update(theta, grad, lr, optimizer if optimizer in ("Momentum","RMSProp","Adam") else "BGD")
            history["theta"].append(theta.copy().tolist())
            history["loss"].append(float(J))
            history["grad"].append(grad.copy().tolist())
            history["preds"].append(preds.tolist())
        else:
            # SGD or Mini-batch: per-batch updates and record each update
            idx = np.arange(m)
            rng.shuffle(idx)
            batches = [idx[i:i+batch_size] for i in range(0, m, batch_size)]
            for b in batches:
                Xb = X[b]
                yb = y[b]
                Jb, gradb, preds_b = compute_linear_loss_and_grad(theta, Xb, yb)
                if noise_scale > 0.0:
                    gradb = gradb + rng.normal(scale=noise_scale, size=gradb.shape)
                if optimizer in ("Momentum", "RMSProp", "Adam"):
                    theta = batch_update(theta, gradb, lr, optimizer)
                else:
                    theta = theta - lr * gradb
                # store full-batch loss for clarity? store batch loss (Jb) and grad computed on batch
                history["theta"].append(theta.copy().tolist())
                history["loss"].append(float(Jb))
                history["grad"].append(gradb.copy().tolist())
                # store preds for the batch indices only as a small helpful hint (keep size small)
                # For simplicity, store preds computed on the batch (not full dataset)
                history["preds"].append(preds_b.tolist())

    return theta.tolist(), history

def run_gradient_descent_quadratic(theta_init, lr, epochs, optimizer="Batch", seed=42,
                                   momentum_beta=0.9, rms_beta=0.9, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8, noise_scale=0.0):
    rng = np.random.default_rng(seed)
    theta = float(theta_init)
    history = {"theta": [], "loss": [], "grad": []}

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
        history["grad"].append(float(grad))

    return float(theta), history

# ---------- contour grid (loss surface) ----------
def compute_contour_grid(X, y, theta0_vals, theta1_vals):
    T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
    JJ = np.zeros_like(T0, dtype=float)
    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            t = np.array([T0[i, j], T1[i, j]])
            Jtmp, _, _ = compute_linear_loss_and_grad(t, X, y)
            JJ[i, j] = Jtmp
    return T0, T1, JJ


## activation functions
# sigmoid 
def sigmoid(x): return 1 / (1 + np.exp(-x))
# sigmoid derivative
def sigmoid_deriv(x): s = sigmoid(x); return s * (1 - s)
def tanh(x): return np.tanh(x)
def tanh_deriv(x): return 1 - np.tanh(x)**2
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def leaky_relu(x, alpha=0.01): return np.where(x >= 0, x, alpha * x)
def leaky_relu_deriv(x, alpha=0.01): return np.where(x >= 0, 1.0, alpha)
def elu(x, alpha=1.0): return np.where(x >= 0, x, alpha * (np.exp(x) - 1))
def elu_deriv(x, alpha=1.0): return np.where(x >= 0, 1.0, alpha * np.exp(x))
def swish(x, beta=1.0): return x * sigmoid(beta * x)
def swish_deriv(x, beta=1.0):
    s = sigmoid(beta * x); return s + beta * x * s * (1 - s)
def mish(x):
    soft = np.log1p(np.exp(x)); return x * np.tanh(soft)
def mish_deriv(x):
    soft = np.log1p(np.exp(x)); tanh_soft = np.tanh(soft); return tanh_soft + x*(1 - tanh_soft**2)*sigmoid(x)
#func and derivative pairs
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
    "relu": (relu, relu_deriv),
    "leaky_relu": (leaky_relu, leaky_relu_deriv),
    "elu": (elu, elu_deriv),
    "swish": (swish, swish_deriv),
    "mish": (mish, mish_deriv),
}
#eq
def mse(y_true, y_pred): return np.mean((y_true - y_pred)**2)
#eq
def forward(params, X_batch, activation_hidden):
    W1, b1, W2, b2 = params
    Z1 = X_batch.dot(W1) + b1
    A1 = activation_hidden(Z1)
    Z2 = A1.dot(W2) + b2
    return Z1, A1, Z2
#eq
def init_params(in_dim, hidden_dim, out_dim, scale=0.1, seed=1):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(in_dim, hidden_dim) * scale
    b1 = np.zeros((1, hidden_dim))
    W2 = rng.randn(hidden_dim, out_dim) * scale
    b2 = np.zeros((1, out_dim))
    return [W1, b1, W2, b2]


