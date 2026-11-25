"""Recovery signal and least-squares derivatives for one voxel."""

import numpy as np

LOG_PARAMETER_LIMIT = 50.0


def signal(x, t):
    """Predict the signal at delays t from x = (log T1, log S0)."""
    log_t1, log_s0 = x[:2]
    scaled_time = t * np.exp(-log_t1)
    recovery = -np.expm1(-scaled_time)
    return np.exp(log_s0) * recovery


def _terms(x, t, y):
    log_t1, log_s0 = x[:2]
    scaled_time = t * np.exp(-log_t1)
    s0 = np.exp(log_s0)

    predicted = s0 * -np.expm1(-scaled_time)
    residual = predicted - y

    r_u = -s0 * scaled_time * np.exp(-scaled_time)
    r_v = predicted
    r_uu = -r_u * (1 - scaled_time)
    return residual, r_u, r_v, r_uu


def _is_invalid(x):
    x = np.asarray(x, dtype=float)
    return not np.isfinite(x).all() or np.any(np.abs(x) > LOG_PARAMETER_LIMIT)


def loss(x, t, y):
    """Return half the sum of squared residuals."""
    if _is_invalid(x):
        return np.inf

    residual = signal(x, t) - y
    return 0.5 * np.sum(residual ** 2)


def grad(x, t, y):
    if _is_invalid(x):
        return np.full(2, np.nan)

    residual, r_u, r_v, _ = _terms(x, t, y)
    return np.array([np.sum(residual * r_u), np.sum(residual * r_v)], dtype=float)


def hess(x, t, y):
    if _is_invalid(x):
        return np.full((2, 2), np.nan)

    residual, r_u, r_v, r_uu = _terms(x, t, y)

    # Differentiating with respect to log S0 leaves the same factor of S0.
    h_uu = np.sum(r_u ** 2 + residual * r_uu)
    h_uv = np.sum(r_u * r_v + residual * r_u)
    h_vv = np.sum(r_v ** 2 + residual * r_v)
    return np.array([[h_uu, h_uv], [h_uv, h_vv]], dtype=float)


def make_fgh(t, y):
    """Fix the delays and observations so solvers only have to pass x."""
    def f(x):
        return loss(x, t, y)

    def g(x):
        return grad(x, t, y)

    def h(x):
        return hess(x, t, y)

    return f, g, h
