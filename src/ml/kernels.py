"""
Kernel functions for SVM.

Each kernel takes two vectors x, y (1-d arrays) and returns a scalar.
The ``get_kernel`` factory returns a vectorised callable that operates on
two matrices X (n, d) and Y (m, d) and produces the full (n, m) Gram matrix.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Low-level scalar kernels (used internally; you may also call them directly)
# ---------------------------------------------------------------------------

def linear(x: np.ndarray, y: np.ndarray) -> float:
    """K(x, y) = x^T y"""
    return float(np.dot(x, y))


def polynomial(x: np.ndarray, y: np.ndarray, degree: int = 3,
               gamma: float = 1.0, coef0: float = 1.0) -> float:
    """K(x, y) = (gamma * x^T y + coef0)^degree"""
    return float((gamma * np.dot(x, y) + coef0) ** degree)


def rbf(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """K(x, y) = exp(-gamma * ||x - y||^2)"""
    diff = x - y
    return float(np.exp(-gamma * np.dot(diff, diff)))


# ---------------------------------------------------------------------------
# Factory that returns a fast, vectorised kernel Gram-matrix builder
# ---------------------------------------------------------------------------

def get_kernel(name: str = "linear", **kwargs) -> Callable:
    """
    Return a callable ``kernel(X, Y) -> Gram matrix of shape (n, m)``.

    Parameters
    ----------
    name : str
        One of ``'linear'``, ``'poly'``/``'polynomial'``, ``'rbf'``.
    **kwargs
        Extra keyword arguments forwarded to the kernel
        (``gamma``, ``degree``, ``coef0``).

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], np.ndarray]
    """
    name = name.lower()

    if name == "linear":
        def kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return X @ Y.T
        return kernel

    if name in ("poly", "polynomial"):
        degree = kwargs.get("degree", 3)
        gamma = kwargs.get("gamma", 1.0)
        coef0 = kwargs.get("coef0", 1.0)

        def kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return (gamma * (X @ Y.T) + coef0) ** degree
        return kernel

    if name == "rbf":
        gamma = kwargs.get("gamma", 1.0)

        def kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 x_i . y_j
            X_norm = np.sum(X ** 2, axis=1)[:, np.newaxis]
            Y_norm = np.sum(Y ** 2, axis=1)[np.newaxis, :]
            sq_dist = X_norm + Y_norm - 2.0 * (X @ Y.T)
            # Clamp to avoid tiny negatives from floating-point arithmetic
            np.maximum(sq_dist, 0.0, out=sq_dist)
            return np.exp(-gamma * sq_dist)
        return kernel

    raise ValueError(f"Unknown kernel '{name}'. Choose from: linear, poly, rbf.")
