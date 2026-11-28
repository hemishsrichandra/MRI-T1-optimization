import numpy as np

def make_quadratic(A, b):
    A = np.asarray(A)
    b = np.asarray(b)

    def f(x):
        return 0.5 * (x @ A @ x) - b @ x
    def g(x):
        return 0.5 * ((A + A.T) @ x) - b
    def h(x):
        return 0.5 * (A + A.T)

    return f, g, h

def double_well():
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
        x2 = x[1]

        temp = [[12 * x1**2 - 4, 0], [0, 2]]
        return np.array(temp)

    return f, g, h

def rosenbrock():
    def f(x):
        x1 = x[0]
        x2 = x[1]

        return 100 * (x2 - x1 ** 2) **2 + (1 - x1) ** 2

    def g(x):
        x1 = x[0]
        x2 = x[1]

        return np.array([-400 * x1 * (x2 - x1 ** 2) - 2 * (1 - x1), 200 * (x2 - x1 ** 2)])

    def h(x):
        x1 = x[0]
        x2 = x[1]

        temp = [[-400 * (x2 - x1 ** 2) + 800 * x1 ** 2 + 2, -400 * x1], [-400 * x1, 200]]
        return np.array(temp)

    return f, g, h

def fd_check(f, g, h, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    n = len(x)
    g_exact = g(x)
    h_exact = h(x)

    g_fd = np.zeros(n, dtype=float)
    h_fd = np.zeros((n, n), dtype=float)

    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = eps

        g_fd[i] = (f(x + e_i) - f(x - e_i)) / (2.0 * eps)
        h_fd[:, i] = (g(x + e_i) - g(x - e_i)) / (2.0 * eps)

    grad_rel_err = np.linalg.norm(g_exact - g_fd) / max(1.0, np.linalg.norm(g_exact))
    hess_rel_err = np.linalg.norm(h_exact - h_fd) / max(1.0, np.linalg.norm(h_exact))
    return grad_rel_err, hess_rel_err