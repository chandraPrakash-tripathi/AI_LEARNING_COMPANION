# frontend/streamlit_app.py
import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide", page_title="AI LEARNING COMPANION")

backend_url = "http://localhost:5000"

mode = st.sidebar.selectbox("Problem", ["Linear Regression ", "Quadratic (Toy Example)"])
optimizer = st.sidebar.selectbox("Optimizer", ["Batch", "SGD", "Mini-batch", "Momentum", "RMSProp", "Adam"])
lr = st.sidebar.number_input("Learning rate (α)", value=0.1, format="%.5f", step=0.01)
epochs = st.sidebar.number_input("Epochs (or steps)", min_value=1, max_value=10000, value=60)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42)

# optimizer hyperparams
st.sidebar.subheader("Optimizer hyperparams")
momentum_beta = st.sidebar.number_input("momentum beta (for Momentum, Adam)", value=0.9, format="%.3f")
rms_beta = st.sidebar.number_input("rms beta (for RMSProp)", value=0.9, format="%.3f")
adam_beta1 = st.sidebar.number_input("adam beta1", value=0.9, format="%.3f")
adam_beta2 = st.sidebar.number_input("adam beta2", value=0.999, format="%.4f")
adam_eps = st.sidebar.number_input("adam eps", value=1e-8, format="%.8f")

if mode == "Linear Regression ":
    st.sidebar.subheader("Dataset")
    n_samples = st.sidebar.slider("Samples", 10, 1000, 80)
    noise = st.sidebar.slider("Noise (std dev)", 0.0, 30.0, 8.0)
    x_scale = st.sidebar.slider("Feature scale", 1.0, 10.0, 3.0)
    batch_size = 1 if optimizer == "SGD" else st.sidebar.slider("Mini-batch size (for Mini-batch)", 1, 100, 16)
    st.sidebar.write("---")
    st.sidebar.subheader("Initialization")
    init_theta0 = st.sidebar.number_input("θ0 (intercept) init", value=0.0, format="%.3f")
    init_theta1 = st.sidebar.number_input("θ1 (slope) init", value=0.0, format="%.3f")
else:
    st.sidebar.subheader("Initialization")
    theta_init = st.sidebar.number_input("θ init", value=4.0, format="%.3f")
    noise_scale = st.sidebar.number_input("SGD noise scale (for demo)", value=0.0, format="%.3f")

run_button = st.sidebar.button("Run")

# local state for stepping/animation
if "history" not in st.session_state:
    st.session_state.history = None
if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "thetas" not in st.session_state:
    st.session_state.thetas = None
if "losses" not in st.session_state:
    st.session_state.losses = None
if "theta_closed" not in st.session_state:
    st.session_state.theta_closed = None
if "data" not in st.session_state:
    st.session_state.data = None

def fetch_contour_and_plot(theta0_range, theta1_range, resolution=80):
    payload = {
        "n_samples": int(n_samples),
        "noise": float(noise),
        "x_scale": float(x_scale),
        "seed": int(seed),
        "theta0_min": float(theta0_range[0]),
        "theta0_max": float(theta0_range[1]),
        "theta1_min": float(theta1_range[0]),
        "theta1_max": float(theta1_range[1]),
        "resolution": int(resolution)
    }
    r = requests.post(f"{backend_url.rstrip('/')}/contour_grid", json=payload, timeout=120)
    r.raise_for_status()
    resp = r.json()
    theta0_vals = np.array(resp["theta0_vals"])
    theta1_vals = np.array(resp["theta1_vals"])
    JJ = np.array(resp["JJ"])
    return theta0_vals, theta1_vals, JJ

def start_play():
    st.session_state.playing = True

def stop_play():
    st.session_state.playing = False

def step_forward():
    if st.session_state.thetas is None:
        return
    st.session_state.step_idx = min(st.session_state.step_idx + 1, len(st.session_state.thetas) - 1)

def step_back():
    if st.session_state.thetas is None:
        return
    st.session_state.step_idx = max(st.session_state.step_idx - 1, 0)

def reset_step():
    st.session_state.step_idx = 0

# Run request to backend
if run_button:
    try:
        if mode == "Linear Regression (2 params)":
            payload = {
                "n_samples": int(n_samples),
                "noise": float(noise),
                "x_scale": float(x_scale),
                "lr": float(lr),
                "epochs": int(epochs),
                "optimizer": optimizer,
                "batch_size": int(batch_size),
                "init_theta0": float(init_theta0),
                "init_theta1": float(init_theta1),
                "seed": int(seed),
                "momentum_beta": float(momentum_beta),
                "rms_beta": float(rms_beta),
                "adam_beta1": float(adam_beta1),
                "adam_beta2": float(adam_beta2),
                "adam_eps": float(adam_eps),
                "noise_scale": float(noise_scale) if "noise_scale" in locals() else 0.0
            }
            r = requests.post(f"{backend_url.rstrip('/')}/run_linear", json=payload, timeout=120)
            r.raise_for_status()
            resp = r.json()
            st.session_state.history = resp["history"]
            st.session_state.thetas = np.array(st.session_state.history["theta"])
            st.session_state.losses = np.array(st.session_state.history["loss"])
            st.session_state.theta_closed = np.array(resp["theta_closed"])
            st.session_state.data = resp["data_sample"]
            st.session_state.step_idx = 0

            # Fetch a contour grid around path + closed form solution for smooth plotting
            if st.session_state.thetas.size > 0:
                t0_min, t0_max = st.session_state.thetas[:,0].min(), st.session_state.thetas[:,0].max()
                t1_min, t1_max = st.session_state.thetas[:,1].min(), st.session_state.thetas[:,1].max()
                pad0 = max(1.0, (t0_max - t0_min) * 0.6 + 0.5)
                pad1 = max(1.0, (t1_max - t1_min) * 0.6 + 0.5)
                center0 = (t0_min + t0_max) / 2.0
                center1 = (t1_min + t1_max) / 2.0
                theta0_range = (center0 - pad0, center0 + pad0)
                theta1_range = (center1 - pad1, center1 + pad1)
            else:
                theta0_range = (-10, 10); theta1_range = (-3, 3)

            try:
                theta0_vals, theta1_vals, JJ = fetch_contour_and_plot(theta0_range, theta1_range, resolution=80)
                st.session_state.contour = {"theta0_vals": theta0_vals, "theta1_vals": theta1_vals, "JJ": JJ}
            except Exception as e:
                st.warning(f"Contour fetch failed: {e}")
                st.session_state.contour = None

        else:
            payload = {
                "theta_init": float(theta_init),
                "lr": float(lr),
                "epochs": int(epochs),
                "optimizer": optimizer,
                "seed": int(seed),
                "momentum_beta": float(momentum_beta),
                "rms_beta": float(rms_beta),
                "adam_beta1": float(adam_beta1),
                "adam_beta2": float(adam_beta2),
                "adam_eps": float(adam_eps),
                "noise_scale": float(noise_scale),
            }
            r = requests.post(f"{backend_url.rstrip('/')}/run_quadratic", json=payload, timeout=60)
            r.raise_for_status()
            resp = r.json()
            st.session_state.history = resp["history"]
            st.session_state.thetas = np.array(st.session_state.history["theta"])
            st.session_state.losses = np.array(st.session_state.history["loss"])
            st.session_state.step_idx = 0

    except Exception as e:
        st.error(f"Request failed: {e}")

control_col, vis_col = st.columns([1, 3])
with control_col:
    st.subheader("Playback")
    if st.button("Play ▶️"):
        start_play()
    if st.button("Pause ⏸"):
        stop_play()
    if st.button("Step ←"):
        step_back()
    if st.button("Step →"):
        step_forward()
    if st.button("Reset"):
        reset_step()
    st.write("Current step index:", st.session_state.step_idx)
    max_steps = len(st.session_state.thetas) - 1 if st.session_state.thetas is not None else 0
    if max_steps > 0:
        idx = st.slider("Scrub steps", 0, max_steps, st.session_state.step_idx, key="scrub")
        st.session_state.step_idx = idx

with vis_col:
    if st.session_state.history is None:
        st.info("RUN")
    else:
        # Drawing based on current step
        step = st.session_state.step_idx
        if mode == "Linear Regression (2 params)":
            X = np.array(st.session_state.data["x"])
            y = np.array(st.session_state.data["y"])
            theta_final = st.session_state.thetas[-1] if st.session_state.thetas.size > 0 else None
            theta_closed = st.session_state.theta_closed

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Data & Fits")
                fig, ax = plt.subplots(figsize=(6,4))
                ax.scatter(X, y, label="Data", alpha=0.7)
                xs = np.linspace(X.min(), X.max(), 200)
                ax.plot(xs, theta_closed[0] + theta_closed[1]*xs, label="Closed-form fit", linewidth=2)
                # show current update line
                current_theta = st.session_state.thetas[step] if st.session_state.thetas.size>0 else np.array([0.0,0.0])
                ax.plot(xs, current_theta[0] + current_theta[1]*xs, '--', label=f"GD fit @ step {step}", linewidth=2)
                ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
                st.pyplot(fig)

            with col2:
                st.subheader("Loss & Trajectory")
                # Loss curve up to current step
                fig2, ax2 = plt.subplots(figsize=(6,4))
                ax2.plot(np.arange(len(st.session_state.losses)), st.session_state.losses, '-o', markersize=3, alpha=0.6)
                ax2.axvline(step, color='red', linestyle='--', linewidth=1)
                ax2.set_xlabel("Step"); ax2.set_ylabel("Loss"); ax2.set_title("Loss vs step")
                st.pyplot(fig2)

                # Trajectory with contour if available
                fig3, ax3 = plt.subplots(figsize=(6,5))
                if st.session_state.contour is not None:
                    # contour expects theta1 on x axis and theta0 on y axis as before
                    theta0_vals = st.session_state.contour["theta0_vals"]
                    theta1_vals = st.session_state.contour["theta1_vals"]
                    JJ = st.session_state.contour["JJ"]
                    # contour expects meshgrid(theta1, theta0) but stored as JJ shaped (len(theta1), len(theta0))
                    cs = ax3.contour(theta1_vals, theta0_vals, np.array(JJ), levels=30)
                    ax3.clabel(cs, inline=1, fontsize=8)
                # full path
                thetas = st.session_state.thetas
                if thetas.size > 0:
                    ax3.plot(thetas[:,1], thetas[:,0], '-o', markersize=3, label="GD path")
                    # highlight current point
                    ax3.scatter([thetas[step,1]],[thetas[step,0]], s=120, c='red', label=f"step {step}")
                    ax3.scatter([theta_closed[1]],[theta_closed[0]], marker='*', s=120, c='green', label='Closed form')
                ax3.set_xlabel("θ1"); ax3.set_ylabel("θ0"); ax3.legend()
                st.pyplot(fig3)

        else:
            # Quadratic visualizations
            thetas = st.session_state.thetas
            losses = st.session_state.losses
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

# Playback loop (if playing)
if st.session_state.playing and st.session_state.thetas is not None:
    max_idx = len(st.session_state.thetas) - 1
    if st.session_state.step_idx < max_idx:
        st.session_state.step_idx += 1
        time.sleep(0.08)
        # rerun to update UI
        st.experimental_rerun()
    else:
        st.session_state.playing = False
