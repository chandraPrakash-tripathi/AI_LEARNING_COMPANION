# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import io
import seaborn as sns

st.set_page_config(layout="wide", page_title="AI Learning Companion — Hyperparams")

# Backend URL (editable in sidebar)
BACKEND_URL = st.sidebar.text_input("Backend URL", "http://localhost:5000")

st.title("AI Learning Companion — Adjust Hyperparameters")

# ---------------------------
# Dataset selector / upload
# ---------------------------
st.sidebar.header("Dataset")
dataset_option = st.sidebar.selectbox(
    "Choose a dataset",
    ["Iris", "Wine", "BreastCancer", "Blobs", "Moons", "Upload CSV"]
)
uploaded_csv = None
if dataset_option == "Upload CSV":
    u = st.sidebar.file_uploader("Upload CSV (last column = label)", type=["csv"])
    if u is not None:
        uploaded_csv = u.getvalue().decode("utf-8")
        st.sidebar.success("CSV loaded")

# ---------------------------
# Model selector and hyperparams
# ---------------------------
model = st.sidebar.selectbox("Model", ["LogisticRegression", "SVM", "DecisionTree", "RandomForest", "KNN", "NeuralNet"])

st.sidebar.header("Hyperparameters")
hyperparams = {}
if model == "LogisticRegression":
    C_exp = st.sidebar.slider("log10(C) (inverse regularization)", -4.0, 1.0, 0.0, step=0.1)
    hyperparams["C"] = float(10 ** C_exp)
    penalty = st.sidebar.selectbox("penalty", ["l2", "none"])
    if penalty != "none":
        hyperparams["penalty"] = penalty
    solver = st.sidebar.selectbox("solver", ["lbfgs", "saga"])
    hyperparams["solver"] = solver
elif model == "SVM":
    C_exp = st.sidebar.slider("log10(C)", -3.0, 2.0, 0.0, step=0.1)
    hyperparams["C"] = float(10 ** C_exp)
    kernel = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly"])
    hyperparams["kernel"] = kernel
    if kernel in ("rbf", "poly"):
        gamma_exp = st.sidebar.slider("log10(gamma)", -4.0, 1.0, -1.0, step=0.1)
        hyperparams["gamma"] = float(10 ** gamma_exp)
elif model == "DecisionTree":
    max_depth = st.sidebar.slider("max_depth (50 = None)", 1, 50, 5)
    hyperparams["max_depth"] = None if max_depth == 50 else int(max_depth)
    min_leaf = st.sidebar.slider("min_samples_leaf", 1, 50, 1)
    hyperparams["min_samples_leaf"] = int(min_leaf)
elif model == "RandomForest":
    n_estimators = st.sidebar.slider("n_estimators", 10, 500, 100, step=10)
    hyperparams["n_estimators"] = int(n_estimators)
    max_depth = st.sidebar.slider("max_depth (50 = None)", 1, 50, 10)
    hyperparams["max_depth"] = None if max_depth == 50 else int(max_depth)
    max_features = st.sidebar.selectbox("max_features", ["auto", "sqrt", "log2", "None"])
    hyperparams["max_features"] = None if max_features == "None" else max_features
elif model == "KNN":
    n_neighbors = st.sidebar.slider("n_neighbors", 1, 30, 5)
    hyperparams["n_neighbors"] = int(n_neighbors)
    weights = st.sidebar.selectbox("weights", ["uniform", "distance"])
    hyperparams["weights"] = weights
elif model == "NeuralNet":
    lr_exp = st.sidebar.slider("log10(learning_rate)", -4.0, -1.0, -3.0, step=0.1)
    hyperparams["learning_rate_init"] = float(10 ** lr_exp)
    hidden = st.sidebar.slider("hidden_units (per layer)", 8, 512, 64, step=8)
    hyperparams["hidden_layer_sizes"] = [int(hidden)]
    batch_size = st.sidebar.selectbox("batch_size", [16, 32, 64, 128])
    hyperparams["batch_size"] = int(batch_size)
    epochs = st.sidebar.slider("epochs", 5, 200, 50)
    hyperparams["max_iter"] = int(epochs)
    act = st.sidebar.selectbox("activation", ["relu", "tanh", "logistic"])
    hyperparams["activation"] = act

# ---------------------------
# Preprocessing
# ---------------------------
st.sidebar.header("Preprocessing")
scale = st.sidebar.selectbox("Scale features", ["none", "standard", "minmax"])
hyperparams_prep = {"scale": scale if scale != "none" else None}
poly_deg = st.sidebar.slider("Polynomial degree (0 = none)", 0, 3, 0)
hyperparams_prep["poly_degree"] = int(poly_deg) if poly_deg > 0 else 1

# ---------------------------
# Training controls
# ---------------------------
st.sidebar.header("Training")
test_size = st.sidebar.slider("Validation fraction", 0.05, 0.5, 0.2, step=0.05)
cv_folds = st.sidebar.selectbox("CV folds (1 = train/val split)", [1, 3, 5, 10])
seed = st.sidebar.number_input("Random seed", value=42)
auto_retrain = st.sidebar.checkbox("Auto retrain on change", value=False)

# ---------------------------
# Search controls
# ---------------------------
st.sidebar.header("Search")
search_mode = st.sidebar.selectbox("Search mode", ["none", "grid", "random"])
param_grid = {}
if search_mode != "none":
    st.sidebar.write("Define up to two hyperparameters for search (JSON). Example: {\"C\": [0.001,0.01,0.1,1]}")
    param_text = st.sidebar.text_area("param_grid (JSON)", value='{"C": [0.001, 0.01, 0.1, 1]}')
    try:
        param_grid = json.loads(param_text)
    except Exception as e:
        st.sidebar.error("param_grid JSON parse error: " + str(e))
n_iter = st.sidebar.number_input("n_iter (random search)", min_value=1, max_value=500, value=20)

# ---------------------------
# Multi-seed
# ---------------------------
st.sidebar.header("Multi-seed")
multi_seed = st.sidebar.checkbox("Run multiple seeds (boxplot)", value=False)
if multi_seed:
    seeds_text = st.sidebar.text_input("Seeds (comma separated)", value=f"{seed},{seed+1},{seed+2}")
    seeds = [int(s.strip()) for s in seeds_text.split(",") if s.strip().isdigit()]
else:
    seeds = [int(seed)]

# Train button
train_button = st.sidebar.button("Train / Run")

# ---------------------------
# Helpers
# ---------------------------
def normalize_payload(obj):
    """Convert numpy types and lists to plain python types so JSON is clean."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [normalize_payload(x) for x in obj]
    if isinstance(obj, dict):
        return {k: normalize_payload(v) for k, v in obj.items()}
    return obj

def build_payload():
    ds = dataset_option
    if dataset_option == "Upload CSV" and uploaded_csv:
        ds = {"csv": uploaded_csv}
    payload = {
        "dataset": ds,
        "model": model,
        "hyperparams": normalize_payload(hyperparams),
        "preprocessing": normalize_payload(hyperparams_prep),
        "test_size": float(test_size),
        "random_state": int(seed),
        "cv_folds": int(cv_folds),
        "search": {"mode": search_mode, "param_grid": normalize_payload(param_grid), "n_iter": int(n_iter)},
        "seeds": seeds
    }
    return payload

# ---------------------------
# UI: Dataset preview
# ---------------------------
st.header("Dataset preview")
if uploaded_csv:
    try:
        df = pd.read_csv(io.StringIO(uploaded_csv))
        st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Failed reading uploaded CSV: {e}")
else:
    st.info(f"Using built-in dataset: {dataset_option}")

# Auto retrain behavior
if auto_retrain and not train_button:
    train_button = True

# ---------------------------
# Main: send request and render
# ---------------------------
if train_button:
    payload = build_payload()
    try:
        with st.spinner("Sending request to backend..."):
            url = BACKEND_URL.rstrip("/") + "/train"
            try:
                r = requests.post(url, json=payload, timeout=300)
            except Exception as e:
                st.error(f"Failed to reach backend at {url}: {e}")
                st.stop()

        # Debug outputs
        st.write("Backend status code:", r.status_code)
        st.subheader("Raw backend response (first 8000 chars)")
        st.code(r.text[:8000])

        if r.status_code != 200:
            st.error(f"Backend returned HTTP {r.status_code} — see raw response above.")
            st.stop()

        try:
            resp = r.json()
        except Exception:
            st.error("Backend returned non-JSON response. See raw response above.")
            st.stop()

        # Accept responses that either include {"status":"ok",...} or direct payloads without "status"
        is_ok = (isinstance(resp, dict) and resp.get("status") == "ok") or ("status" not in resp)

        if not is_ok:
            st.error(f"Backend error: {resp.get('message','(no message)')}")
            if resp.get("traceback"):
                st.subheader("Backend traceback")
                st.text_area("Traceback", resp.get("traceback"), height=400)
            st.stop()

        # Normalize response payload to a common shape with `results` list
        payload_ok = resp if "status" not in resp else resp
        results = payload_ok.get("results")
        if results is None:
            single = {}
            # gather common keys if present
            for k in ("train_accuracy", "val_accuracy", "confusion_matrix", "roc_curve", "roc_auc",
                      "feature_importances", "coefficients", "loss_epochs", "losses"):
                if k in payload_ok:
                    single[k] = payload_ok[k]
            results = [single]

        # Render results
        st.success(f"Trained — time: {resp.get('time_taken', 0):.2f}s")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Result per run")
            for idx, res in enumerate(results):
                st.markdown(f"**Run {idx}**")
                st.write("Train acc:", res.get("train_accuracy"), "Val acc:", res.get("val_accuracy"))
                if res.get("confusion_matrix"):
                    cm = np.array(res.get("confusion_matrix"))
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                    ax.set_title(f"Confusion matrix (run {idx})")
                    st.pyplot(fig)

        # Boxplot if multiple results
        if len(results) > 1:
            vals = [r.get("val_accuracy") for r in results if r.get("val_accuracy") is not None]
            if vals:
                fig2, ax2 = plt.subplots()
                ax2.boxplot(vals, vert=True, labels=["val_accuracy"])
                ax2.set_title("Validation accuracy distribution across runs")
                st.pyplot(fig2)

        # First run summary (for losses, importances, roc, etc.)
        first = results[0] if results else {}

        # Loss curve
        if first.get("loss_epochs") and first.get("losses"):
            st.subheader("Loss vs epochs")
            figL, axL = plt.subplots()
            axL.plot(first["loss_epochs"], first["losses"], marker="o")
            axL.set_xlabel("Epoch")
            axL.set_ylabel("Loss")
            axL.set_title("Training loss")
            st.pyplot(figL)

        # Feature importances / coefficients
        if first.get("feature_importances"):
            fi = first["feature_importances"]
            df_fi = pd.DataFrame.from_dict(fi, orient="index", columns=["importance"]).sort_values("importance", ascending=False)
            st.subheader("Feature importances")
            st.bar_chart(df_fi.head(10))
        elif first.get("coefficients"):
            coeff = first["coefficients"]
            df_coef = pd.DataFrame.from_dict(coeff, orient="index", columns=["coef"]).sort_values("coef", key=abs, ascending=False)
            st.subheader("Coefficients (approx)")
            st.bar_chart(df_coef.head(10))

        # Grid heatmap (if present)
        if payload_ok.get("grid_heatmap"):
            gh = payload_ok["grid_heatmap"]
            st.subheader("Grid search heatmap")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.heatmap(np.array(gh["scores"]), annot=True, xticklabels=gh["x_vals"], yticklabels=gh["y_vals"], ax=ax3)
            ax3.set_xlabel(gh["x_name"])
            ax3.set_ylabel(gh["y_name"])
            st.pyplot(fig3)

        # ROC curve (if present)
        if first.get("roc_curve"):
            rc = first["roc_curve"]
            st.subheader("ROC curve")
            fig4, ax4 = plt.subplots()
            ax4.plot(rc["fpr"], rc["tpr"], label=f"AUC={first.get('roc_auc', None):.3f}" if first.get('roc_auc') else "AUC=N/A")
            ax4.plot([0, 1], [0, 1], linestyle="--", color="grey")
            ax4.set_xlabel("FPR")
            ax4.set_ylabel("TPR")
            ax4.legend()
            st.pyplot(fig4)

        # Raw JSON for debugging
        st.subheader("Raw result JSON (first run)")
        st.json(first)

    except Exception as e:
        st.error(f"Request failed: {e}")
