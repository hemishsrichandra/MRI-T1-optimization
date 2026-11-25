"""Show what each solver does on the same MRI voxel and starting point."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from benchmark import make_start
from model import make_fgh
from reference import reference_fit
from registry import SOLVERS


FIGURE_DIR = Path("figures")
SOLVER_COLORS = {
    "SD": "#0072B2",
    "Newton": "#D55E00",
    "ModNewton": "#009E73",
    "TR": "#83559E",
    "DFP": "#AE7700",
    "BFGS": "#263449",
}
LOG_PLOT_FLOOR = 1e-16


def save_figure(fig, out):
    """Save a PNG and close the figure so repeated calls free the memory."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return path


def _logged(values):
    values = np.asarray(values, dtype=float)
    logged = np.full_like(values, np.nan)
    valid = np.isfinite(values) & (values >= 0)
    logged[valid] = np.log10(np.maximum(values[valid], LOG_PLOT_FLOOR))
    return logged


def _fit_voxel(t, y, T1_true, factor):
    f, g, h = make_fgh(np.asarray(t, dtype=float), np.asarray(y, dtype=float))
    x0 = make_start(y, T1_true, factor)
    runs = {name: solver(x0.copy(), f, g, h) for name, solver in SOLVERS.items()}
    return x0, f, runs


def _caption(T1_true, factor, snr):
    caption = f"True T1 {T1_true:g} ms  |  Start {factor:g}x true T1"
    return caption if snr is None else f"SNR {snr:g}  |  {caption}"


def fig_trajectories(t, y, T1_true, factor, *, snr=None, out=FIGURE_DIR / "figure_1_trajectories.png"):
    """Save Figure 1 with the loss contours, solver paths, and reference fit."""
    x0, f, runs = _fit_voxel(t, y, T1_true, factor)
    ref_u, ref_v, _ = reference_fit(t, y)
    optimum = np.array([ref_u, ref_v])

    # Use the start and reference to set the view. A runaway step can leave it.
    padding = np.maximum(0.55 * np.abs(x0 - optimum), [0.7, 0.5])
    lower = np.minimum(x0, optimum) - padding
    upper = np.maximum(x0, optimum) + padding
    u = np.linspace(lower[0], upper[0], 120)
    v = np.linspace(lower[1], upper[1], 120)
    loss = [[f([log_t1, log_s0]) for log_t1 in u] for log_s0 in v]

    fig, ax = plt.subplots(figsize=(9, 5.8), layout="constrained")
    contours = ax.contourf(u, v, np.ma.masked_invalid(_logged(loss)), levels=20, cmap="Greys", alpha=0.74)
    fig.colorbar(contours, ax=ax, location="bottom", shrink=0.85, label=r"$\log_{10}(f)$")

    for name, result in runs.items():
        history = np.asarray(result["history"], dtype=float)
        inside = np.isfinite(history).all(axis=1) & ((history >= lower) & (history <= upper)).all(axis=1)
        visible = np.where(inside[:, None], history, np.nan)
        valid = visible[inside]
        label = f"{name} ({result['status']})"
        if not inside.all():
            label += " [clipped]"
        color = SOLVER_COLORS[name]
        ax.plot(*visible.T, label=label, color=color, linewidth=1.55)
        if valid.size:
            ax.scatter(*valid[-1], color=color, s=15, zorder=3)

    ax.scatter(*x0, s=76, marker="o", facecolor="white", edgecolor="black", label="Common start", zorder=5)
    ax.scatter(*optimum, s=108, marker="*", color="black", label="Reference fit", zorder=5)
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), xlabel="ln(T1)", ylabel="ln(S0)")
    ax.set_title("Solver paths on the loss surface", loc="left", fontweight="bold")
    fig.suptitle(_caption(T1_true, factor, snr), fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5)
    return save_figure(fig, out)


def fig_convergence(t, y, T1_true, factor, *, snr=None, out=FIGURE_DIR / "figure_2_convergence.png"):
    """Save Figure 2 with gradient norms starting at iteration zero."""
    _, _, runs = _fit_voxel(t, y, T1_true, factor)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")

    for name, result in runs.items():
        norms = _logged(result["grad_norms"])
        color = SOLVER_COLORS[name]
        ax.plot(np.arange(norms.size), norms, color=color, linewidth=1.55,
                label=f"{name} ({result['status']})")
        # A solver may stop before taking a step. Its initial norm is still a point.
        valid = np.flatnonzero(np.isfinite(norms))
        if valid.size:
            ax.scatter(valid[[0, -1]], norms[valid[[0, -1]]], s=17, color=color, zorder=3)

    ax.set(xlabel="Iteration (accepted steps)", ylabel=r"$\log_{10}(\|g(x)\|_2)$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=0)
    ax.grid(alpha=0.23)
    ax.set_title("Gradient norm by iteration", loc="left", fontweight="bold")
    fig.suptitle(_caption(T1_true, factor, snr), fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    return save_figure(fig, out)
