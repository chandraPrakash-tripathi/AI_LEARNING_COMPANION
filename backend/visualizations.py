import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1. Gradient Descent Visualization
# -----------------------------
def gradient_descent_visualization(learning_rate=0.1, steps=20):
    x = np.linspace(-10, 10, 100)
    y = x**2  # cost function: J(w) = w^2

    w = 8
    trajectory = [w]
    for _ in range(steps):
        grad = 2 * w
        w -= learning_rate * grad
        trajectory.append(w)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, label="Cost Function J(w) = w²")
    ax.scatter(trajectory, [t**2 for t in trajectory], color='r', label="Descent Path")
    ax.set_title(f"Gradient Descent (lr={learning_rate}, steps={steps})")
    ax.set_xlabel("Weight (w)")
    ax.set_ylabel("Cost (J)")
    ax.legend()
    return fig


# -----------------------------
# 2. Activation Functions Plot
# -----------------------------
def activation_plot(func_name):
    x = np.linspace(-10, 10, 200)
    if func_name == "ReLU":
        y = np.maximum(0, x)
    elif func_name == "Sigmoid":
        y = 1 / (1 + np.exp(-x))
    elif func_name == "Tanh":
        y = np.tanh(x)
    elif func_name == "LeakyReLU":
        y = np.where(x > 0, x, 0.01 * x)
    else:
        raise ValueError("Unknown activation function")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, linewidth=2)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f"{func_name} Activation Function")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    return fig
