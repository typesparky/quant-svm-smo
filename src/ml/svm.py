"""
Support Vector Machine trained via Sequential Minimal Optimization (SMO).

References
----------
- Platt, J. (1998). *Sequential Minimal Optimization: A Fast Algorithm for
  Training Support Vector Machines*.
- Keerthi, S.S. et al. (2001). *Improvements to Platt's SMO Algorithm for
  SVM Classifier Design*.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .kernels import get_kernel


class SVM:
    """
    Binary SVM classifier trained with SMO.

    Parameters
    ----------
    C : float
        Regularisation parameter (soft-margin penalty).
        Use ``C=float('inf')`` for a hard-margin SVM.
    kernel : str
        Kernel name: ``'linear'``, ``'poly'``, or ``'rbf'``.
    degree : int
        Polynomial degree (only used when ``kernel='poly'``).
    gamma : float
        Kernel coefficient for ``'rbf'`` and ``'poly'``.
        If ``'auto'``, uses ``1 / n_features``.
    coef0 : float
        Independent term for ``'poly'`` kernel.
    tol : float
        Numerical tolerance for KKT violation checks.
    max_passes : int
        Maximum number of passes over the dataset without any alpha change
        before the outer loop terminates.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "linear",
        degree: int = 3,
        gamma: float | str = "auto",
        coef0: float = 1.0,
        tol: float = 1e-3,
        max_passes: int = 20,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma  # may be "auto"
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes

        # Populated during training
        self.alphas_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.support_vectors_: Optional[np.ndarray] = None
        self.support_alphas_: Optional[np.ndarray] = None
        self.support_y_: Optional[np.ndarray] = None
        self._kernel_fn = None
        self._cache = {}
        self._X_train_cached = None

    def _get_kernel_row(self, i: int) -> np.ndarray:
        """Compute or retrieve kernel row K[i, :]."""
        if i not in self._cache:
            # Compute row K[i, :] = K(x_i, X)
            self._cache[i] = self._kernel_fn(self._X_train_cached[i:i+1], self._X_train_cached).flatten()
        return self._cache[i]

    def _get_kernel_val(self, i: int, j: int) -> float:
        """Compute or retrieve kernel value K[i, j]."""
        return self._get_kernel_row(i)[j]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        """
        Train the SVM on data X with labels y in {-1, +1}.
        """
        n_samples, n_features = X.shape

        # Resolve gamma
        if self.gamma == "auto":
            gamma_val = 1.0 / n_features
        else:
            gamma_val = float(self.gamma)

        self._kernel_fn = get_kernel(
            self.kernel, gamma=gamma_val, degree=self.degree, coef0=self.coef0
        )
        
        # Save training data for kernel computations
        self._X_train_cached = X
        self._y_train = y
        self._cache = {}

        # SMO working variables
        alphas = np.zeros(n_samples)
        b = 0.0
        passes = 0


        # Cache error terms E_i = f(x_i) - y_i
        E = -y.copy().astype(float)

        while passes < self.max_passes:
            num_changed = 0

            for i in range(n_samples):
                E_i = E[i]
                y_i = y[i]
                alpha_i = alphas[i]

                # Check KKT violation
                if (
                    (y_i * E_i < -self.tol and alpha_i < self.C)
                    or (y_i * E_i > self.tol and alpha_i > 0)
                ):
                    # Select j != i (heuristic: max |E_i - E_j|)
                    j = self._select_j(i, n_samples, E)

                    alpha_j = alphas[j]
                    E_j = E[j]
                    y_j = y[j]

                    # Compute bounds L, H
                    if y_i != y_j:
                        L = max(0.0, alpha_j - alpha_i)
                        H = min(self.C, self.C + alpha_j - alpha_i)
                    else:
                        L = max(0.0, alpha_i + alpha_j - self.C)
                        H = min(self.C, alpha_i + alpha_j)

                    if abs(L - H) < 1e-12:
                        continue

                    # eta = 2 K(i,j) - K(i,i) - K(j,j)  (should be <= 0)
                    eta = 2 * self._get_kernel_val(i, j) - self._get_kernel_val(i, i) - self._get_kernel_val(j, j)
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    new_alpha_j = alpha_j - y_j * (E_i - E_j) / eta
                    new_alpha_j = np.clip(new_alpha_j, L, H)

                    if abs(new_alpha_j - alpha_j) < 1e-5:
                        continue

                    # Update alpha_i
                    new_alpha_i = alpha_i + y_i * y_j * (alpha_j - new_alpha_j)

                    # Compute bias
                    b1 = (
                        b
                        - E_i
                        - y_i * (new_alpha_i - alpha_i) * self._get_kernel_val(i, i)
                        - y_j * (new_alpha_j - alpha_j) * self._get_kernel_val(i, j)
                    )
                    b2 = (
                        b
                        - E_j
                        - y_i * (new_alpha_i - alpha_i) * self._get_kernel_val(i, j)
                        - y_j * (new_alpha_j - alpha_j) * self._get_kernel_val(j, j)
                    )

                    if 0 < new_alpha_i < self.C:
                        b_new = b1
                    elif 0 < new_alpha_j < self.C:
                        b_new = b2
                    else:
                        b_new = (b1 + b2) / 2.0

                    # Update cache -------------------------------------------------
                    delta_alpha_i = new_alpha_i - alpha_i
                    delta_alpha_j = new_alpha_j - alpha_j
                    
                    # Update E cache iteratively
                    for k in range(n_samples):
                        E[k] += (
                            delta_alpha_i * y_i * self._get_kernel_val(k, i)
                            + delta_alpha_j * y_j * self._get_kernel_val(k, j)
                            + (b_new - b)
                        )

                    # Commit
                    alphas[i] = new_alpha_i
                    alphas[j] = new_alpha_j
                    b = b_new
                    num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        # Store trained parameters
        self.alphas_ = alphas
        self.b_ = b

        sv_mask = alphas > 1e-7
        self.support_vectors_ = X[sv_mask]
        self.support_alphas_ = alphas[sv_mask]
        self.support_y_ = y[sv_mask]

        # Save training data for kernel computations
        self._X_train_cached = X
        self._y_train = y
        self._cache = {}

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return the raw decision function value f(x) for each row of X.
        """
        K_test = self._kernel_fn(X, self._X_train_cached)  # (n_test, n_train)
        return K_test @ (self.alphas_ * self._y_train) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels {-1, +1} for X."""
        return np.sign(self.decision_function(X))

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _select_j(i: int, n: int, E: np.ndarray) -> int:
        """Second-choice heuristic: pick j != i that maximises |E_i - E_j|."""
        j = i
        while j == i:
            j = np.random.randint(0, n)

        # Try to find a better j using the error cache
        non_bound = np.where((E != 0))[0]
        if len(non_bound) > 1:
            candidates = non_bound[non_bound != i]
            if len(candidates) > 0:
                idx = np.argmax(np.abs(E[i] - E[candidates]))
                j = candidates[idx]
        return j
