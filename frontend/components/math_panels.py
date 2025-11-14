import streamlit as st
import numpy as np
import pandas as pd


def linear_math_panel(step, thetas, grads, losses, data_sample, lr):
    with st.expander("Maths Involved"):
        st.subheader(f"Step {step} calculations")
        theta_k = thetas[step]
        grad_k = grads[step] if grads is not None and len(grads)>step else None
        J_k = losses[step] if len(losses)>step else None
        st.markdown("### Loss (MSE/2)")
        m = np.array(data_sample['x']).shape[0]
        if grad_k is not None:
            g0 = float(grad_k[0])
            g1 = float(grad_k[1]) if len(grad_k)>1 else 0.0
            st.latex(r"J(\theta) = \frac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2")
            if J_k is not None:
                st.markdown(f"**Numeric J:** {J_k:.6f}")
            st.markdown("### Gradient components")
            st.latex(r"\frac{\partial J}{\partial \theta_0} = \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})")
            st.markdown(f"**Numeric:** $\\frac{{\\partial J}}{{\\partial \\theta_0}} = {g0:.6f}$")
            st.latex(r"\frac{\partial J}{\partial \theta_1} = \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}")
            st.markdown(f"**Numeric:** $\\frac{{\\partial J}}{{\\partial \\theta_1}} = {g1:.6f}$")
            st.markdown("### Parameter update")
            st.latex(r"\theta := \theta - \alpha \nabla J(\theta)")
            st.markdown(f"**Numeric update (vector):** $\\theta_{{k+1}} = {theta_k.tolist()} - {lr} \\cdot [{g0:.6f}, {g1:.6f}]$")
            updated = [theta_k[0] - lr * g0, theta_k[1] - lr * g1]
            st.markdown(f"**Resulting θ (after update):** {np.array(updated).round(6).tolist()}")
            df = pd.DataFrame({
                "value": [theta_k[0], theta_k[1], g0, g1, lr, J_k],
            }, index=["theta0 (current)", "theta1 (current)", "grad theta0", "grad theta1", "learning rate", "loss"])
            st.table(df)
        else:
            st.write("No gradient information available for this step.")


def quadratic_math_panel(step, thetas, grads, losses, lr):
    with st.expander("Math at current step (show calculations)"):
        st.subheader(f"Step {step} calculations (Quadratic)")
        theta_k = thetas[step]
        grad_k = grads[step] if grads is not None and len(grads)>step else None
        J_k = losses[step] if len(losses)>step else None
        st.latex(r"J(\theta) = \tfrac{1}{2}\theta^2 \quad\Longrightarrow\quad \frac{dJ}{d\theta} = \theta")
        st.markdown(f"**Current θ:** {theta_k:.6f}")
        if grad_k is not None:
            st.markdown(f"**Gradient (numeric):** {float(grad_k):.6f}")
            st.latex(r"\theta := \theta - \alpha \frac{dJ}{d\theta}")
            st.markdown(f"**Numeric update:** θ_new = {theta_k:.6f} - {lr} * {float(grad_k):.6f} = {theta_k - lr*float(grad_k):.6f}")
            st.markdown(f"**Loss at step:** {J_k:.6f}")
        else:
            st.write("No gradient info available.")


