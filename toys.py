"""Small test problems and a finite-difference check for derivatives."""

import numpy as np


def make_quadratic(A, b):
    """Build f, g, and h for the quadratic 0.5 * x.T @ A @ x - b.T @ x."""
    matrix = np.asarray(A)
    linear_term = np.asarray(b)

    def f(x):
        return 0.5 * (x @ matrix @ x) - linear_term @ x

    def g(x):
        return 0.5 * ((matrix + matrix.T) @ x) - linear_term

    def h(x):
        return 0.5 * (matrix + matrix.T)

    return f, g, h


def double_well():
    """Build a two-dimensional function with minima at (-1, 0) and (1, 0)."""
    def f(x):
        x1 = x[0]
        x2 = x[1]

        return x1 ** 4 - 2 * x1 ** 2 + 2 + x2 ** 2

    def g(x):
        x1 = x[0]
        x2 = x[1]

        return np.array([4 * x1 ** 3 - 4 * x1, 2 * x2])

    def h(x):
        x1 = x[0]
        return np.array([[12 * x1 ** 2 - 4, 0], [0, 2]])

    return f, g, h


def rosenbrock():
    """Build the Rosenbrock function and its derivatives."""
    def f(x):
        x1 = x[0]
        x2 = x[1]

        return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    def g(x):
        x1 = x[0]
        x2 = x[1]

        gap = x2 - x1 ** 2
        return np.array([-400 * x1 * gap - 2 * (1 - x1), 200 * gap])

    def h(x):
        x1 = x[0]
        x2 = x[1]

        diagonal = -400 * (x2 - x1 ** 2) + 800 * x1 ** 2 + 2
        cross_term = -400 * x1
        return np.array([[diagonal, cross_term], [cross_term, 200]])

    return f, g, h


def fd_check(f, g, h, x, eps=1e-6):
    """Compare the supplied gradient and Hessian with central differences."""
    x = np.asarray(x, dtype=float)
    dimensions = len(x)
    gradient = g(x)
    hessian = h(x)

    approx_gradient = np.zeros(dimensions, dtype=float)
    approx_hessian = np.zeros((dimensions, dimensions), dtype=float)

    # Perturb one coordinate at a time to get one column of the Hessian.
    for i in range(dimensions):
        step = np.zeros(dimensions, dtype=float)
        step[i] = eps
        ahead = x + step
        behind = x - step

        approx_gradient[i] = (f(ahead) - f(behind)) / (2.0 * eps)
        approx_hessian[:, i] = (g(ahead) - g(behind)) / (2.0 * eps)

    gradient_error = np.linalg.norm(gradient - approx_gradient)
    hessian_error = np.linalg.norm(hessian - approx_hessian)
    return (
        gradient_error / max(1.0, np.linalg.norm(gradient)),
        hessian_error / max(1.0, np.linalg.norm(hessian)),
    )
