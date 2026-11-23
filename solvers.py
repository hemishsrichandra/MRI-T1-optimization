import numpy as np

def _make_result(x, history, f_vals, g_norms, converged, reason):
    # Packages optimization metrics into a standard result dictionary.
    return {
        "x": np.asarray(x, dtype=float),
        "path": np.asarray(history, dtype=float),    # iterate history trajectory for contour plots
        "f_vals": np.asarray(f_vals, dtype=float),   # f(x_k) per iteration
        "g_norms": np.asarray(g_norms, dtype=float), # ||grad||_2 per iteration
        "nit": len(f_vals) - 1,
        "converged": bool(converged),
        "reason": reason,
    }


def backtracking(f, grad, d, x, alpha0=1.0, rho=0.5, c1=1e-4, min_alpha=1e-12):
    alpha = alpha0
    slope = grad @ d

    # d must be a descent direction (slope < 0)
    if slope >= 0:
        return 0

    fx = f(x)

    while alpha >= min_alpha:
        x_new = x + alpha * d
        f_new = f(x_new)

        if np.isfinite(f_new) and f_new <= fx + c1 * alpha * slope:
            return alpha
        
        alpha = rho * alpha

    return 0


def steepest_descent(f, g, h, x0, tol=1e-6, max_iter=500):
    k = 0
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

        if k >= max_iter:
            return _make_result(x, history, f_vals, g_norms, False, "max_iter reached")

        d = -grad
        alpha = backtracking(f, grad, d, x)

        if alpha == 0.0:
            return _make_result(x, history, f_vals, g_norms, False, "line search failed")

        x = x + alpha * d
        history.append(x.copy())
        f_vals.append(f(x))
        k += 1


def classical_newton(f, g, h, x0, tol=1e-6, max_iter=100):
    k = 0
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

        if k >= max_iter:
            return _make_result(x, history, f_vals, g_norms, False, "max_iter reached")

        try:
            d = -np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            return _make_result(x, history, f_vals, g_norms, False, "singular Hessian")

        x = x + d
        fx_new = f(x)
        history.append(x.copy())
        f_vals.append(fx_new)

        if not np.isfinite(fx_new):
            g_norms.append(np.nan)
            return _make_result(x, history, f_vals, g_norms, False, "step overflowed")

        k += 1


def modified_newton(f, g, h, x0, tol=1e-6, max_iter=200, delta=1e-3):
    k = 0
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

        if k >= max_iter:
            return _make_result(x, history, f_vals, g_norms, False, "max_iter reached")

        # Eigenvalue check and shift to guarantee strict positive definiteness
        eigs = np.linalg.eigvalsh(hessian)
        min_eig = np.min(eigs)

        eta = 0.0
        if min_eig <= 0:
            eta = delta - min_eig

        B = hessian + eta * np.eye(len(x))
        try:
            d = -np.linalg.solve(B, grad)
        except np.linalg.LinAlgError:
            d = -grad

        alpha = backtracking(f, grad, d, x)
        if alpha == 0.0:
            return _make_result(x, history, f_vals, g_norms, False, "line search failed")

        x = x + alpha * d
        history.append(x.copy())
        f_vals.append(f(x))
        k += 1


def fqk(x, x_k, f, grad, hessian):
    d = x - x_k
    return f(x_k) + grad @ d + 0.5 * (d @ hessian @ d)


def find_delta(x_k, x_k_plus_1, delta, f, grad, hessian, delta_max=10.0):
    denom = fqk(x_k, x_k, f, grad, hessian) - fqk(x_k_plus_1, x_k, f, grad, hessian)
    f_cand = f(x_k_plus_1)

    if (not np.isfinite(f_cand)) or denom <= 0.0:
        rk = -1.0
    else:
        rk = (f(x_k) - f_cand) / denom

    d = x_k_plus_1 - x_k
    if rk < 0.25:
        delta = delta / 4.0
    elif rk > 0.75 and np.isclose(np.linalg.norm(d), delta, rtol=1e-2):
        delta = min(2.0 * delta, delta_max)

    return rk, delta


def cauchy_step(grad, hessian, delta):
    g_norm = np.linalg.norm(grad)
    if g_norm == 0.0:
        return np.zeros_like(grad)

    gHg = grad @ hessian @ grad
    if gHg <= 0:
        alpha = delta / g_norm
    else:
        alpha = min(delta / g_norm, (grad @ grad) / gHg)

    return -alpha * grad


def trust_region(f, g, h, x0, tol=1e-6, max_iter=300, delta0=1.0, delta_max=10.0, eta=0.1):
    k = 0
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

        if k >= max_iter:
            return _make_result(x, history, f_vals, g_norms, False, "max_iter reached")

        # Compute candidate displacement step d inside ||d|| <= delta
        d = cauchy_step(grad, hessian, delta)
        x_k_plus_1 = x + d

        rk, delta = find_delta(x, x_k_plus_1, delta, f, grad, hessian, delta_max=delta_max)

        # Accept step if agreement ratio rk is above threshold eta
        if rk > eta:
            x = x_k_plus_1
            history.append(x.copy())
            f_vals.append(f(x))

        if delta < 1e-12:
            return _make_result(x, history, f_vals, g_norms, False, "trust radius collapsed")

        k += 1


# Dictionary mapping solver names to callable functions
ALL_SOLVERS = {
    "Steepest Descent": steepest_descent,
    "Classical Newton": classical_newton,
    "Modified Newton": modified_newton,
    "Trust-Region": trust_region,
}