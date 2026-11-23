"""Solvers with a common call signature and result format for plotting."""

import numpy as np

from quasi_newton import bfgs, dfp
from solvers import classical_newton, modified_newton, steepest_descent, trust_region


def _for_benchmark(solver):
    def run(x0, f, g, h, max_iter=200, tol=1e-8):
        result = solver(f, g, h, x0, max_iter=max_iter, tol=tol)
        history = result["path"]

        # Trust-region trials can be rejected. Match the norms to accepted points.
        if solver is trust_region:
            grad_norms = [float(np.linalg.norm(g(point))) for point in history]
        else:
            grad_norms = result["g_norms"].tolist()

        if result["converged"]:
            status = "converged"
        elif result["reason"] == "max_iter reached":
            status = "max_iter"
        elif result["reason"] == "singular Hessian":
            status = "linalg_error"
        else:
            status = "diverged"

        return {
            "x": result["x"],
            "f": float(result["f_vals"][-1]),
            "iters": int(result["nit"]),
            "status": status,
            "history": history,
            "grad_norms": grad_norms,
        }

    return run


SOLVERS = {
    "SD": _for_benchmark(steepest_descent),
    "Newton": _for_benchmark(classical_newton),
    "ModNewton": _for_benchmark(modified_newton),
    "TR": _for_benchmark(trust_region),
    "DFP": dfp,
    "BFGS": bfgs,
}
