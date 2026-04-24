"""
svm-from-scratch -- Support Vector Machine via SMO.

Example
-------
>>> from svm import SVM
>>> import numpy as np
>>> X = np.array([[1,1], [2,1], [1,2], [2,2], [-1,-1], [-2,-1], [-1,-2], [-2,-2]])
>>> y = np.array([1, 1, 1, 1, -1, -1, -1, -1])
>>> clf = SVM(kernel="linear", C=1.0)
>>> clf.fit(X, y)
>>> clf.predict(np.array([[3, 3], [-3, -3]]))
array([ 1., -1.])
"""

from .svm import SVM
from .kernels import get_kernel, linear, polynomial, rbf
from .utils import (
    make_linearly_separable,
    make_moons,
    make_circles,
    make_xor,
    plot_decision_boundary,
    plot_parameter_effect,
)

__all__ = [
    "SVM",
    "get_kernel",
    "linear",
    "polynomial",
    "rbf",
    "make_linearly_separable",
    "make_moons",
    "make_circles",
    "make_xor",
    "plot_decision_boundary",
    "plot_parameter_effect",
]

__version__ = "1.0.0"
