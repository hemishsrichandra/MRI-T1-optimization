"""Optimization methods that share the same result format."""

import numpy as np


def _make_result(x, history, f_vals, g_norms, converged, reason):
    return {
        "x": np.asarray(x, dtype=float),
        "path": np.asarray(history, dtype=float),
        "f_vals": np.asarray(f_vals, dtype=float),
        "g_norms": np.asarray(g_norms, dtype=float),
        "nit": len(f_vals) - 1,
        "converged": bool(converged),
        "reason": reason,
    }


def backtracking(f, grad, d, x, alpha0=1.0, rho=0.5, c1=1e-4, min_alpha=1e-12):
    """Shrink the step until it gives enough decrease in f."""
    slope = grad @ d

    if slope >= 0:
        return 0

    alpha, _, _ = _armijo_trials(f, x, f(x), d, slope, alpha0, c1, rho, min_alpha)
    return 0 if alpha is None else alpha


def armijo(f, x, fx, g_x, d, c1=1e-4, rho=0.5, max_back=40):
    """Try the full step and up to max_back reductions."""
    slope = g_x @ d
    if not np.isfinite(slope) or slope >= 0:
        return None, x, fx

    return _armijo_trials(
        f, x, fx, d, slope, 1.0, c1, rho, np.finfo(float).tiny,
        max_trials=max_back + 1,
    )


def _armijo_trials(f, x, fx, d, slope, alpha, c1, rho, min_alpha, max_trials=np.inf):
    trials = 0
    while alpha >= min_alpha and trials < max_trials:
        x_new = x + alpha * d
        f_new = f(x_new)

        if np.isfinite(f_new) and f_new <= fx + c1 * alpha * slope:
            return alpha, x_new, f_new

        alpha = rho * alpha
        trials += 1

    return None, x, fx


def steepest_descent(f, g, h, x0, tol=1e-6, max_iter=500):
    """Follow the negative gradient with a backtracking step size."""
    steps = 0
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    f_vals = [f(x)]
    g_norms = []

    while True:
        grad = g(x)
        if not np.all(np.isfinite(grad)):
            g_norms.append(np.nan)
            return _make_result(x, history, f_vals, g_norms, False, "non-finite gradient")

        g_norm = np.linalg.norm(grad)
        g_norms.append(g_norm)

        if g_norm < tol:
            return _make_result(x, history, f_vals, g_norms, True, "converged")

        if steps >= max_iter:
            return _make_result(x, history, f_vals, g_norms, False, "max_iter reached")

        direction = -grad
        alpha = backtracking(f, grad, direction, x)

        if alpha == 0.0:
            return _make_result(x, history, f_vals, g_norms, False, "line search failed")

        x = x + alpha * direction
        history.append(x.copy())
        f_vals.append(f(x))
        steps += 1


def classical_newton(f, g, h, x0, tol=1e-6, max_iter=100):
    """Take the full Newton step from the gradient and Hessian."""
    steps = 0
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    f_vals = [f(x)]
    g_norms = []

    while True:
        grad = g(x)
        hessian = h(x)
        if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hessian))):
            g_norms.append(np.nan)
            return _make_result(x, history, f_vals, g_norms, False, "overflow guard triggered")

        g_norm = np.linalg.norm(grad)
        g_norms.append(g_norm)

        if g_norm < tol:
            return _make_result(x, history, f_vals, g_norms, True, "converged")

        if steps >= max_iter:
            return _make_result(x, history, f_vals, g_norms, False, "max_iter reached")

        try:
            step = -np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            return _make_result(x, history, f_vals, g_norms, False, "singular Hessian")

        x = x + step
        fx_new = f(x)
        history.append(x.copy())
        f_vals.append(fx_new)

        if not np.isfinite(fx_new):
            g_norms.append(np.nan)
            return _make_result(x, history, f_vals, g_norms, False, "step overflowed")

        steps += 1


def modified_newton(f, g, h, x0, tol=1e-6, max_iter=200, delta=1e-3):
    """Shift the Hessian when needed, then use backtracking."""
    steps = 0
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    f_vals = [f(x)]
    g_norms = []

    while True:
        grad = g(x)
        hessian = h(x)
        if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hessian))):
            g_norms.append(np.nan)
            return _make_result(x, history, f_vals, g_norms, False, "non-finite derivatives")

        g_norm = np.linalg.norm(grad)
        g_norms.append(g_norm)

        if g_norm < tol:
            return _make_result(x, history, f_vals, g_norms, True, "converged")

        if steps >= max_iter:
            return _make_result(x, history, f_vals, g_norms, False, "max_iter reached")

        # Move the smallest eigenvalue to delta if it is zero or negative.
        smallest_eigenvalue = np.linalg.eigvalsh(hessian).min()
        shift = 0.0
        if smallest_eigenvalue <= 0:
            shift = delta - smallest_eigenvalue

        shifted_hessian = hessian + shift * np.eye(len(x))
        try:
            direction = -np.linalg.solve(shifted_hessian, grad)
        except np.linalg.LinAlgError:
            direction = -grad

        alpha = backtracking(f, grad, direction, x)
        if alpha == 0.0:
            return _make_result(x, history, f_vals, g_norms, False, "line search failed")

        x = x + alpha * direction
        history.append(x.copy())
        f_vals.append(f(x))
        steps += 1


def fqk(x, x_k, f, grad, hessian):
    """Evaluate the quadratic approximation around x_k at x."""
    step = x - x_k
    return f(x_k) + grad @ step + 0.5 * (step @ hessian @ step)


def find_delta(x_k, x_k_plus_1, delta, f, grad, hessian, delta_max=10.0):
    """Use the agreement between actual and predicted decrease to update the radius."""
    model_now = fqk(x_k, x_k, f, grad, hessian)
    model_trial = fqk(x_k_plus_1, x_k, f, grad, hessian)
    predicted_drop = model_now - model_trial
    trial_loss = f(x_k_plus_1)

    if not np.isfinite(trial_loss) or predicted_drop <= 0.0:
        ratio = -1.0
    else:
        ratio = (f(x_k) - trial_loss) / predicted_drop

    step = x_k_plus_1 - x_k
    if ratio < 0.25:
        delta = delta / 4.0
    elif ratio > 0.75 and np.isclose(np.linalg.norm(step), delta, rtol=1e-2):
        delta = min(2.0 * delta, delta_max)

    return ratio, delta


def cauchy_step(grad, hessian, delta):
    """Pick a step along the negative gradient within the trust radius."""
    g_norm = np.linalg.norm(grad)
    if g_norm == 0.0:
        return np.zeros_like(grad)

    curvature = grad @ hessian @ grad
    if curvature <= 0:
        alpha = delta / g_norm
    else:
        alpha = min(delta / g_norm, (grad @ grad) / curvature)

    return -alpha * grad


def trust_region(f, g, h, x0, tol=1e-6, max_iter=300, delta0=1.0, delta_max=10.0, eta=0.1):
    """Follow a Cauchy step and adjust the radius after each trial."""
    attempts = 0
    x = np.array(x0, dtype=float)
    delta = float(delta0)
    history = [x.copy()]
    f_vals = [f(x)]
    g_norms = []

    while True:
        grad = g(x)
        hessian = h(x)
        if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hessian))):
            g_norms.append(np.nan)
            return _make_result(x, history, f_vals, g_norms, False, "non-finite derivatives")

        g_norm = np.linalg.norm(grad)
        g_norms.append(g_norm)

        if g_norm < tol:
            return _make_result(x, history, f_vals, g_norms, True, "converged")

        if attempts >= max_iter:
            return _make_result(x, history, f_vals, g_norms, False, "max_iter reached")

        step = cauchy_step(grad, hessian, delta)
        trial = x + step
        ratio, delta = find_delta(x, trial, delta, f, grad, hessian, delta_max=delta_max)

        if ratio > eta:
            x = trial
            history.append(x.copy())
            f_vals.append(f(x))

        if delta < 1e-12:
            return _make_result(x, history, f_vals, g_norms, False, "trust radius collapsed")

        attempts += 1


ALL_SOLVERS = {
    "Steepest Descent": steepest_descent,
    "Classical Newton": classical_newton,
    "Modified Newton": modified_newton,
    "Trust-Region": trust_region,
}
