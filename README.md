# AI Learning Companion 

An interactive educational web app designed to help understand **Machine Learning concepts visually**. Built using **Streamlit** (frontend UI) and **Flask** (backend ML engine)

---

## Project Overview

**AI Learning Companion**  The app simplifies abstract topics by combining:

* Real-time visualizations
* Adjustable hyperparameters
* Mathematical explanations
* Interactive sliders and controls

This project replicates the experience of an “ML playground” — perfect for learning, teaching, and demonstrating ML fundamentals.
Also it implements CI/CD using GitHub Actions with two workflow pipelines: dev and main.
---

## Key Features

### 1. **Gradient Descent Visualization**

* Watch gradient descent move step-by-step.
* Adjust learning rate, iterations, and noise.
* View the contour plot, loss curve, and parameter updates.

### 2. **Activation Functions Playground**

* Explore ReLU, Sigmoid, Tanh, LeakyReLU, and more.
* Visual + mathematical formula breakdown.
* Adjustable input domain & parameters.

### 3. **ML Model Comparison Dashboard**

* Compare classic ML models:

  * Linear Regression
  * Logistic Regression
  * Decision Trees
  * SVM
  * KNN
* Side‑by‑side metrics & plots.

---

## Tech Stack

| Layer          | Technology                      |
| -------------- | ------------------------------- |
| Frontend UI    | Streamlit                       |
| Backend API    | Flask                           |
| ML Models      | scikit-learn, NumPy, Pandas     |
| Visualizations | Matplotlib / Altair             |
| Architecture   | Frontend ↔ REST API ↔ ML Engine |

---

## How It Works

1. **Streamlit** handles all UI interactions and visualizations.
2. User inputs → sent to **Flask backend**.
3. Flask computes ML outputs (model predictions, GD steps, activations).
4. Streamlit renders plots, animations, and mathematical panels.

---

## Project Structure

---

## Running the Project

### 1️ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️ Start the Flask backend

```bash
cd backend
python app.py
```

### 3️ Start the Streamlit frontend

```bash
cd frontend
streamlit run 0_HOME.py
```

---
For queries or collaboration, feel free to connect!
