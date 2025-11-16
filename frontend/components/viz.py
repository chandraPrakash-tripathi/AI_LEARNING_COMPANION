import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import altair as alt
import pandas as pd

def plot_linear_visuals(data_sample, theta_closed, thetas, losses, grads, contour, step):
    X = np.array(data_sample["x"])
    y = np.array(data_sample["y"])

    # 3 PARALLEL COLUMNS
    col1, col2, col3 = st.columns(3)

    # ===== COL 1: DATA & FITS =====
    with col1:
        st.subheader("Data & Fits")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X, y, label="Data", alpha=0.7)
        xs = np.linspace(X.min(), X.max(), 200)

        ax.plot(xs, theta_closed[0] + theta_closed[1] * xs,
                label="Closed-form fit", linewidth=2)

        current_theta = thetas[step] if thetas.size > 0 else np.array([0.0, 0.0])
        ax.plot(xs, current_theta[0] + current_theta[1] * xs,
                '--', label=f"GD fit @ step {step}", linewidth=2)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        st.pyplot(fig)

    # ===== COL 2: LOSS CURVE =====
    with col2:
        st.subheader("Loss Curve")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(np.arange(len(losses)), losses, '-o', markersize=3, alpha=0.7)
        ax2.axvline(step, color='red', linestyle='--')
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")
        st.pyplot(fig2)

    # ===== COL 3: CONTOUR & GD TRAJECTORY =====
    with col3:
        st.subheader("Contour & GD Path")
        fig3, ax3 = plt.subplots(figsize=(6, 4))

        if contour is not None:
            theta0_vals = contour["theta0_vals"]
            theta1_vals = contour["theta1_vals"]
            JJ = contour["JJ"]
            cs = ax3.contour(theta1_vals, theta0_vals, np.array(JJ), levels=30)
            ax3.clabel(cs, inline=1, fontsize=8)

        ax3.plot(thetas[:, 1], thetas[:, 0], '-o', markersize=3, label="GD path")
        ax3.scatter([thetas[step, 1]], [thetas[step, 0]], s=120, c='red')
        ax3.scatter([theta_closed[1]], [theta_closed[0]], marker='*', s=120, c='green')
        ax3.set_xlabel("θ1")
        ax3.set_ylabel("θ0")
        st.pyplot(fig3)


def plot_quadratic_visuals(thetas, losses, grads, step):
    # 2 PARALLEL COLUMNS
    col1, col2 = st.columns(2)

    # ===== COL 1: θ TRAJECTORY =====
    with col1:
        st.subheader("θ evolution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(len(thetas)), thetas, '-o')
        ax.axvline(step, color='red', linestyle='--')
        ax.set_xlabel("Step")
        ax.set_ylabel("θ")
        st.pyplot(fig)

    # ===== COL 2: LOSS =====
    with col2:
        st.subheader("Loss Curve")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(np.arange(len(losses)), losses, '-o')
        ax2.axvline(step, color='red', linestyle='--')
        ax2.set_xlabel("Step")
        ax2.set_ylabel("J(θ)")
        st.pyplot(fig2)

def alt_line_chart(df, x_col, y_col, title=None, width=None, height=300):
    chart = alt.Chart(df).mark_line().encode(x=x_col, y=y_col, tooltip=[x_col, y_col])
    if title:
        chart = chart.properties(title=title, width=width, height=height)
    return chart

def plot_activation(x, y, name):
    df = pd.DataFrame({"x": x, name: y})
    return alt_line_chart(df, "x", name, title=f"{name} — value")

def plot_derivative(x, dy):
    df = pd.DataFrame({"x": x, "derivative": dy})
    return alt_line_chart(df, "x", "derivative", title="Derivative")

#activation functions and their derivatives for visualization
def plot_activation(ax, x, y, label=None):
    ax.plot(x, y, linewidth=2, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)

def plot_derivative(ax, x, dy, label=None):
    ax.plot(x, dy, linestyle="--", linewidth=1.5, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("f'(x)")
    ax.grid(True)

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

def render_activation_plots(activation_name, x, y, dy, show_derivative):
    """
    Renders the two side-by-side charts:
        - Activation value plot
        - Derivative plot (toggle-able)

    Parameters
    ----------
    activation_name : str
        Name of the activation function
    x : np.ndarray
        Input domain
    y : np.ndarray
        Activation(x)
    dy : np.ndarray
        Activation derivative values
    show_derivative : bool
        Whether to display the derivative
    """

    col1, col2 = st.columns([1, 1])

    # --- Activation plot ---
    with col1:
        st.subheader(f"{activation_name} — value")
        df_val = pd.DataFrame({"x": x, f"{activation_name}(x)": y})
        chart_val = alt_line_chart(df_val, "x", f"{activation_name}(x)")
        st.altair_chart(chart_val, use_container_width=True)

    # --- Derivative plot ---
    with col2:
        st.subheader(f"{activation_name} — derivative")
        if show_derivative:
            df_der = pd.DataFrame({"x": x, "derivative": dy})
            chart_der = alt_line_chart(df_der, "x", "derivative")
            st.altair_chart(chart_der, use_container_width=True)
        else:
            st.info("Derivative hidden (toggle from sidebar)")
