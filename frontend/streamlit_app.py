import streamlit as st
import requests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.visualizations import gradient_descent_visualization, activation_plot

# -----------------------
# STREAMLIT CONFIG
# -----------------------
st.set_page_config(page_title="AI Learning Companion", layout="wide")

st.title("🧠 AI Learning Companion")
st.write("An interactive app to understand ML concepts visually!")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Gradient Descent", "Activation Functions", "Model Comparison"])

# -----------------------
# PAGE 1 — Gradient Descent
# -----------------------
if page == "Gradient Descent":
    st.header("🎢 Visualize Gradient Descent")

    lr = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
    steps = st.slider("Steps", 10, 100, 20)

    fig = gradient_descent_visualization(learning_rate=lr, steps=steps)
    st.pyplot(fig)

    st.markdown("""
    **Concept:**  
    Gradient-Descent minimizes the cost function by updating weights in the opposite direction of the gradient.
    """)

# -----------------------
# PAGE 2 — Activation Functions
# -----------------------
elif page == "Activation Functions":
    st.header("⚡ Activation Function Playground")

    func_name = st.selectbox(
        "Choose Activation Function",
        ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"]
    )
    fig = activation_plot(func_name)
    st.pyplot(fig)

    st.markdown("""
    **Tip:** Try comparing these functions to understand how they transform input signals.
    """)

# -----------------------
# PAGE 3 — Model Comparison
# -----------------------
elif page == "Model Comparison":
    st.header("🤖 Compare ML Models")

    st.info("Click below to fetch model comparison results from Flask backend.")
    if st.button("Compare Models"):
        try:
            response = requests.get("http://127.0.0.1:5000/compare")
            if response.status_code == 200:
                results = response.json()
                st.bar_chart(results)
            else:
                st.error("Failed to connect to Flask backend.")
        except Exception as e:
            st.error(f"Error: {e}")
