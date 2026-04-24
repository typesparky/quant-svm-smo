"""
Utility helpers: synthetic data generators and matplotlib plotting functions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic dataset generators
# ---------------------------------------------------------------------------

def make_linearly_separable(
    n_samples: int = 200,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-class linearly separable dataset (2-D)."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    return X, y


def make_moons(
    n_samples: int = 200,
    noise: float = 0.15,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles (classic non-linear toy dataset)."""
    rng = np.random.RandomState(random_state)
    n = n_samples // 2
    theta = np.linspace(0, np.pi, n)
    upper = np.column_stack([np.cos(theta), np.sin(theta)])
    lower = np.column_stack([1 - np.cos(theta), -np.sin(theta) + 0.5])
    X = np.vstack([upper, lower])
    X += rng.randn(*X.shape) * noise
    y = np.hstack([np.ones(n), -np.ones(n)])
    return X, y


def make_circles(
    n_samples: int = 200,
    noise: float = 0.08,
    factor: float = 0.5,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concentric circles (non-linear toy dataset)."""
    rng = np.random.RandomState(random_state)
    n = n_samples // 2
    theta = rng.uniform(0, 2 * np.pi, n)
    outer = np.column_stack([np.cos(theta), np.sin(theta)])
    inner = factor * np.column_stack([np.cos(theta), np.sin(theta)])
    X = np.vstack([outer, inner])
    X += rng.randn(*X.shape) * noise
    y = np.hstack([np.ones(n), -np.ones(n)])
    return X, y


def make_xor(
    n_samples: int = 200,
    noise: float = 0.15,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """XOR pattern -- requires non-linear decision boundary."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 2)
    raw = (X[:, 0] > 0) ^ (X[:, 1] > 0)
    y = np.where(raw, 1, -1).astype(float)
    X += rng.randn(n_samples, 2) * noise
    return X, y


# ---------------------------------------------------------------------------
# Plotting helpers (lazy-import matplotlib)
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "SVM Decision Boundary",
    ax=None,
    figsize: Tuple[int, int] = (7, 6),
    save_path: Optional[str] = None,
):
    """Plot the decision boundary with support vectors highlighted."""
    plt = _plt()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    # Contour plot
    ax.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.RdBu)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.8, levels=[-0.5, 0, 0.5])

    # Scatter data points
    colors = ["#e74c3c" if yi == -1 else "#2980b9" for yi in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, edgecolors="k", s=35, zorder=3)

    # Highlight support vectors
    if hasattr(model, "support_vectors_") and model.support_vectors_ is not None:
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=120,
            facecolors="none",
            edgecolors="gold",
            linewidths=2.0,
            zorder=4,
            label="Support Vectors",
        )
        ax.legend(loc="best", fontsize=9)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig, ax


def plot_parameter_effect(
    X: np.ndarray,
    y: np.ndarray,
    param_name: str,
    param_values: list,
    kernel: str = "rbf",
    gamma: float = 1.0,
    C: float = 1.0,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
):
    """Visualise the effect of varying a hyper-parameter on the decision boundary."""
    from svm.svm import SVM  # late import to avoid circular dependency

    plt = _plt()
    n = len(param_values)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 4.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, val in enumerate(param_values):
        kw = dict(kernel=kernel)
        if param_name == "C":
            kw["C"] = val
            kw["gamma"] = gamma
            title = f"$C = {val}$"
        elif param_name == "gamma":
            kw["gamma"] = val
            kw["C"] = C
            title = f"$\\gamma = {val}$"
        else:
            raise ValueError(f"Unknown param_name: {param_name}")

        model = SVM(**kw)
        model.fit(X, y)
        acc = np.mean(model.predict(X) == y) * 100
        plot_decision_boundary(
            model, X, y, title=f"{title}  (acc={acc:.0f}%)", ax=axes[i]
        )

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig, axes
