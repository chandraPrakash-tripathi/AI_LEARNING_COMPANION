# components/activations.py
import numpy as np

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
