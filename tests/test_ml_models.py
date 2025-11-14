# tests/test_ml_models.py
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend import ml_models as mm

def test_quadratic_grad():
    theta = 3.0
    g = mm.quadratic_grad(theta)
    # numerical grad
    eps = 1e-6
    numerical = (mm.quadratic_loss(theta+eps) - mm.quadratic_loss(theta-eps)) / (2*eps)
    assert np.isclose(g, numerical, atol=1e-6)

def test_linear_grad_shapes():
    X, y, _ = mm.make_linear_dataset(n_samples=10, noise=0.1, x_scale=1.0, seed=1)
    theta = np.array([0.0, 0.0])
    J, grad = mm.compute_linear_loss_and_grad(theta, X, y)
    assert grad.shape == theta.shape
    assert isinstance(J, float)
