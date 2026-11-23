import numpy as np

def signal(x, t):
    u = x[0]
    v = x[1]

    a = t * np.exp(-u)
    phi = 1 - np.exp(-a)

    return phi * np.exp(v)

def _terms(x, t, y):
    u = x[0]
    v = x[1]

    a = t * np.exp(-u)
    E = np.exp(-a)
    phi = 1 - E

    r = np.exp(v) * phi - y
    r_u = - np.exp(v) * a * E
    r_v = np.exp(v) * phi
    r_uu = np.exp(v) * a * E * (1 - a)

    return r, r_u, r_v, r_uu

def _is_invalid(x):
    x = np.asarray(x, dtype=float)
    return (not np.all(np.isfinite(x))) or np.any(np.abs(x) > 50.0)

def loss(x, t, y):
    if _is_invalid(x):
        return np.inf
    r = signal(x, t) - y
    return 0.5 * np.sum(r ** 2)

def grad(x, t, y):
    if _is_invalid(x):
        return np.full(2, np.nan)

    r, r_u, r_v, _ = _terms(x, t, y)

    return np.array([
        np.sum(r * r_u),
        np.sum(r * r_v)
    ], dtype=float)

def hess(x, t, y):
    if _is_invalid(x):
        return np.full((2, 2), np.nan)
    
    r, r_u, r_v, r_uu = _terms(x, t, y)

    #r_uv == r_u and r_vv == r_v
    h_uu = np.sum(r_u ** 2 + r * r_uu)
    h_uv = np.sum(r_u * r_v + r * r_u)
    h_vv = np.sum(r_v ** 2 + r * r_v)

    return np.array([
        [h_uu, h_uv],
        [h_uv, h_vv]
    ], dtype=float)

def make_fgh(t, y):
    def f(x):
        return loss(x, t, y)
    def g(x):
        return grad(x, t, y)
    def h(x):
        return hess(x, t, y)
    
    return f, g, h