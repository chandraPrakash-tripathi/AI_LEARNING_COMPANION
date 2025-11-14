import requests
import numpy as np
from .constants import backend_url




def fetch_contour_and_plot(theta0_range, theta1_range, resolution=80, *, n_samples, noise, x_scale, seed):
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




def run_linear_backend(payload):
    r = requests.post(f"{backend_url.rstrip('/')}/run_linear", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()




def run_quadratic_backend(payload):
    r = requests.post(f"{backend_url.rstrip('/')}/run_quadratic", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()