"""DFP and BFGS with an inverse-Hessian estimate and Armijo step sizes."""

import numpy as np

from solvers import armijo


# Check both the absolute size of the dot product and its size relative to the vectors.
CURVATURE_TOL = 1e-12


def _curvature(a, b):
    dot = a @ b
    scale = max(1.0, np.linalg.norm(a) * np.linalg.norm(b))
    if np.isfinite(dot) and dot > CURVATURE_TOL * scale:
        return dot
    return None


def dfp_update(B, delta, gamma):
    """Apply the inverse DFP update when both curvature terms are safe."""
    with np.errstate(over="ignore", invalid="ignore"):
        curvature = _curvature(delta, gamma)
        if curvature is None:
            return B

        B_gamma = B @ gamma
        projected_curvature = _curvature(gamma, B_gamma)
        if projected_curvature is None:
            return B

        step_term = np.outer(delta, delta) / curvature
        correction = np.outer(B_gamma, B_gamma) / projected_curvature
        updated = B + step_term - correction

    return updated if np.isfinite(updated).all() else B


def bfgs_update(B, delta, gamma):
    """Apply the inverse BFGS update when the curvature is safe."""
    with np.errstate(over="ignore", invalid="ignore"):
        curvature = _curvature(delta, gamma)
        if curvature is None:
            return B

        rho = 1.0 / curvature
        V = np.eye(B.shape[0]) - rho * np.outer(delta, gamma)
        updated = V @ B @ V.T + rho * np.outer(delta, delta)

    return updated if np.isfinite(updated).all() else B


def _quasi_newton(x0, f, g, max_iter, tol, update):
    x = np.array(x0, dtype=float)
    if x.shape != (2,):
        raise ValueError("x0 must contain two parameters.")

    B = np.eye(x.size)
    history = [x.copy()]
    grad_norms = []
    fx = float("nan")

    try:
        fx = float(f(x))
        gradient = np.asarray(g(x), dtype=float)
        grad_norms.append(float(np.linalg.norm(gradient)))

        while True:
            grad_norm = grad_norms[-1]
            if not np.isfinite(fx) or not np.isfinite(grad_norm):
                status = "diverged"
                break
            if grad_norm < tol:
                status = "converged"
                break
            if len(history) - 1 >= max_iter:
                status = "max_iter"
                break

            direction = -B @ gradient
            alpha, x_new, f_new = armijo(f, x, fx, gradient, direction)
            if alpha is None:
                status = "diverged"
                break

            # Record the accepted point before computing its gradient.
            delta = x_new - x
            x, fx = x_new, float(f_new)
            history.append(x.copy())
            new_gradient = np.asarray(g(x), dtype=float)
            grad_norms.append(float(np.linalg.norm(new_gradient)))
            B = update(B, delta, new_gradient - gradient)
            gradient = new_gradient

    except np.linalg.LinAlgError:
        status = "linalg_error"
        if len(grad_norms) < len(history):
            grad_norms.append(float("nan"))

    return {
        "x": x,
        "f": fx,
        "iters": len(history) - 1,
        "status": status,
        "history": np.array(history),
        "grad_norms": grad_norms,
    }


def dfp(x0, f, g, h, max_iter=200, tol=1e-8):
    """Minimize f using gradient g and the inverse DFP update."""
    return _quasi_newton(x0, f, g, max_iter, tol, dfp_update)


def bfgs(x0, f, g, h, max_iter=200, tol=1e-8):
    """Minimize f using gradient g and the inverse BFGS update."""
    return _quasi_newton(x0, f, g, max_iter, tol, bfgs_update)
