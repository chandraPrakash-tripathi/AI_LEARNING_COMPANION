import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def plot_linear_visuals(data_sample, theta_closed, thetas, losses, grads, contour, step):
    X = np.array(data_sample["x"])
    y = np.array(data_sample["y"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data & Fits")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(X, y, label="Data", alpha=0.7)
        xs = np.linspace(X.min(), X.max(), 200)
        ax.plot(xs, theta_closed[0] + theta_closed[1]*xs, label="Closed-form fit", linewidth=2)
        current_theta = thetas[step] if thetas.size>0 else np.array([0.0,0.0])
        ax.plot(xs, current_theta[0] + current_theta[1]*xs, '--', label=f"GD fit @ step {step}", linewidth=2)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Loss & Trajectory")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.plot(np.arange(len(losses)), losses, '-o', markersize=3, alpha=0.6)
        ax2.axvline(step, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel("Step"); ax2.set_ylabel("Loss"); ax2.set_title("Loss vs step")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(6,5))
        if contour is not None:
            theta0_vals = contour["theta0_vals"]
            theta1_vals = contour["theta1_vals"]
            JJ = contour["JJ"]
            cs = ax3.contour(theta1_vals, theta0_vals, np.array(JJ), levels=30)
            ax3.clabel(cs, inline=1, fontsize=8)
        ax3.plot(thetas[:,1], thetas[:,0], '-o', markersize=3, label="GD path")
        ax3.scatter([thetas[step,1]],[thetas[step,0]], s=120, c='red', label=f"step {step}")
        ax3.scatter([theta_closed[1]],[theta_closed[0]], marker='*', s=120, c='green', label='Closed form')
        ax3.set_xlabel("θ1"); ax3.set_ylabel("θ0"); ax3.legend()
        st.pyplot(fig3)


def plot_quadratic_visuals(thetas, losses, grads, step):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("θ evolution")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(np.arange(len(thetas)), thetas, '-o')
        ax.axvline(step, color='red', linestyle='--')
        ax.set_xlabel("Step"); ax.set_ylabel("θ"); st.pyplot(fig)
    with col2:
        st.subheader("Loss curve")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.plot(np.arange(len(losses)), losses, '-o')
        ax2.axvline(step, color='red', linestyle='--')
        ax2.set_xlabel("Step"); ax2.set_ylabel("J(θ)"); st.pyplot(fig2)
