import streamlit as st
import numpy as np
import time
from components.constants import backend_url
from components.session_state import ensure_session_state_keys
from components.ui_sidebar import render_sidebar
from components.ui_controls import render_controls_stepplay
from components.backend_client import fetch_contour_and_plot, run_linear_backend, run_quadratic_backend
from components.viz import plot_linear_visuals, plot_quadratic_visuals
from components.math_panels import linear_math_panel, quadratic_math_panel

st.set_page_config(
    layout="wide",
    page_title="AI LEARNING COMPANION"
)
def main():
    ensure_session_state_keys()

    col_sidebar, col_controls = st.columns([2, 1])

    with col_sidebar:
        sidebar_vals = render_sidebar(st.session_state)

    with col_controls:
        render_controls_stepplay()   


    run_button = sidebar_vals.get("run_button")
    mode = sidebar_vals.get("mode")

    if run_button:
        try:
            if mode == "Linear Regression":
                payload = {
                    "n_samples": int(sidebar_vals["n_samples"]),
                    "noise": float(sidebar_vals["noise"]),
                    "x_scale": float(sidebar_vals["x_scale"]),
                    "lr": float(sidebar_vals["lr"]),
                    "epochs": int(sidebar_vals["epochs"]),
                    "optimizer": sidebar_vals["optimizer"],
                    "batch_size": int(sidebar_vals["batch_size"]),
                    "init_theta0": float(sidebar_vals["init_theta0"]),
                    "init_theta1": float(sidebar_vals["init_theta1"]),
                    "seed": int(sidebar_vals["seed"]),
                    "momentum_beta": float(sidebar_vals["momentum_beta"]),
                    "rms_beta": float(sidebar_vals["rms_beta"]),
                    "adam_beta1": float(sidebar_vals["adam_beta1"]),
                    "adam_beta2": float(sidebar_vals["adam_beta2"]),
                    "adam_eps": float(sidebar_vals["adam_eps"]),
                    "noise_scale": float(sidebar_vals.get("noise_scale", 0.0))
                }

                resp = run_linear_backend(payload)
                st.session_state.history = resp["history"]
                st.session_state.thetas = np.array(resp["history"]["theta"])
                st.session_state.losses = np.array(resp["history"]["loss"])
                st.session_state.grads = np.array(resp["history"]["grad"])
                st.session_state.theta_closed = np.array(resp["theta_closed"])
                st.session_state.data = resp["data_sample"]
                st.session_state.step_idx = 0

                # Contour
                try:
                    if st.session_state.thetas.size > 0:
                        t0 = st.session_state.thetas[:, 0]
                        t1 = st.session_state.thetas[:, 1]
                        pad0 = max(1, (t0.max() - t0.min()) * 0.6 + 0.5)
                        pad1 = max(1, (t1.max() - t1.min()) * 0.6 + 0.5)
                        center0 = (t0.max() + t0.min()) / 2
                        center1 = (t1.max() + t1.min()) / 2

                        theta0_range = (center0 - pad0, center0 + pad0)
                        theta1_range = (center1 - pad1, center1 + pad1)
                    else:
                        theta0_range, theta1_range = (-10, 10), (-3, 3)

                    t0_vals, t1_vals, JJ = fetch_contour_and_plot(
                        theta0_range,
                        theta1_range,
                        resolution=80,
                        n_samples=sidebar_vals["n_samples"],
                        noise=sidebar_vals["noise"],
                        x_scale=sidebar_vals["x_scale"],
                        seed=sidebar_vals["seed"]
                    )
                    st.session_state.contour = {
                        "theta0_vals": t0_vals,
                        "theta1_vals": t1_vals,
                        "JJ": JJ
                    }
                except Exception as e:
                    st.warning(f"Contour failed: {e}")
                    st.session_state.contour = None

            else:
                payload = {
                    "theta_init": float(sidebar_vals["theta_init"]),
                    "lr": float(sidebar_vals["lr"]),
                    "epochs": int(sidebar_vals["epochs"]),
                    "optimizer": sidebar_vals["optimizer"],
                    "seed": int(sidebar_vals["seed"]),
                    "momentum_beta": float(sidebar_vals["momentum_beta"]),
                    "rms_beta": float(sidebar_vals["rms_beta"]),
                    "adam_beta1": float(sidebar_vals["adam_beta1"]),
                    "adam_beta2": float(sidebar_vals["adam_beta2"]),
                    "adam_eps": float(sidebar_vals["adam_eps"]),
                    "noise_scale": float(sidebar_vals["noise_scale"]),
                }

                resp = run_quadratic_backend(payload)
                st.session_state.history = resp["history"]
                st.session_state.thetas = np.array(resp["history"]["theta"])
                st.session_state.losses = np.array(resp["history"]["loss"])
                st.session_state.grads = np.array(resp["history"]["grad"])
                st.session_state.step_idx = 0

        except Exception as e:
            st.error(f"Backend error: {e}")

    # ==== VISUALIZATION BELOW THE PARALLEL ROW ====
    st.divider()
    st.subheader("Visualization")

    if st.session_state.history is None:
        st.info("Click Run to start")
    else:
        step = st.session_state.step_idx
        if mode == "Linear Regression":
            plot_linear_visuals(
                st.session_state.data,
                st.session_state.theta_closed,
                st.session_state.thetas,
                st.session_state.losses,
                st.session_state.grads,
                st.session_state.contour,
                step
            )
            linear_math_panel(
                step,
                st.session_state.thetas,
                st.session_state.grads,
                st.session_state.losses,
                st.session_state.data,
                sidebar_vals["lr"]
            )
        else:
            plot_quadratic_visuals(
                st.session_state.thetas,
                st.session_state.losses,
                st.session_state.grads,
                step
            )
            quadratic_math_panel(
                step,
                st.session_state.thetas,
                st.session_state.grads,
                st.session_state.losses,
                sidebar_vals["lr"]
            )

    # ==== PLAYBACK LOOP ====
    if st.session_state.playing and st.session_state.thetas is not None:
        if st.session_state.step_idx < len(st.session_state.thetas) - 1:
            st.session_state.step_idx += 1
            time.sleep(0.08)
            st.rerun()
        else:
            st.session_state.playing = False


if __name__ == "__main__":
    main()
