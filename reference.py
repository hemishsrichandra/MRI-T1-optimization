"""Grid-search baseline for comparing the MRI solvers."""

import numpy as np

from model import LOG_PARAMETER_LIMIT


def varpro_on_grid(u_grid, t, y, eps=1e-6):
    """Fit S0 at each grid point and return (log T1, S0, loss) for the best fit."""
    log_t1_grid = np.asarray(u_grid, dtype=float)
    delays = np.asarray(t, dtype=float)
    signal = np.asarray(y, dtype=float)

    if (
        log_t1_grid.ndim != 1
        or log_t1_grid.size == 0
        or not np.isfinite(log_t1_grid).all()
    ):
        raise ValueError("u_grid must be a nonempty vector of finite log-T1 values.")
    if delays.ndim != 1 or delays.shape != signal.shape or delays.size == 0:
        raise ValueError("t and y must be nonempty vectors with matching shapes.")
    if not np.isfinite(delays).all() or np.any(delays < 0) or not np.isfinite(signal).all():
        raise ValueError("Delays must be finite and nonnegative, and y must be finite.")

    s0_ceiling = np.exp(LOG_PARAMETER_LIMIT)
    if not np.isfinite(eps) or not 0 < eps <= s0_ceiling:
        raise ValueError("eps must be a positive, finite S0 floor within the model range.")

    # Each row is the recovery curve for one T1 value with S0 set to 1.
    recovery = -np.expm1(-delays * np.exp(-log_t1_grid[:, None]))
    curve_dot_signal = recovery @ signal
    curve_dot_itself = np.einsum("ij,ij->i", recovery, recovery)

    # If the recovery curve is all zeros, keep S0 at the floor.
    s0_values = np.full(log_t1_grid.shape, eps)
    np.divide(curve_dot_signal, curve_dot_itself, out=s0_values, where=curve_dot_itself > 0)
    np.clip(s0_values, eps, s0_ceiling, out=s0_values)

    residuals = recovery * s0_values[:, None] - signal
    losses = 0.5 * np.einsum("ij,ij->i", residuals, residuals)
    best_index = np.argmin(losses)
    return (
        float(log_t1_grid[best_index]),
        float(s0_values[best_index]),
        float(losses[best_index]),
    )


def reference_fit(t, y, rel_tol=1e-4, max_rounds=30):
    """Search for T1 on a finer grid each round and return (log T1, log S0, loss)."""
    if not np.isfinite(rel_tol) or rel_tol <= 0 or max_rounds < 1:
        raise ValueError("rel_tol must be positive and max_rounds must be at least 1.")

    limit = LOG_PARAMETER_LIMIT
    grid = np.linspace(-limit, limit, 401)
    s0_floor = np.exp(-limit)
    best_log_t1, best_s0, best_loss = varpro_on_grid(grid, t, y, eps=s0_floor)

    for _ in range(max_rounds - 1):
        best_index = np.searchsorted(grid, best_log_t1)
        left = max(0, best_index - 1)
        right = min(grid.size - 1, best_index + 1)

        # Zoom in between the neighbors and keep the best point from the last round.
        finer_grid = np.linspace(grid[left], grid[right], 33)
        grid = np.unique(np.append(finer_grid, best_log_t1))
        trial_log_t1, trial_s0, trial_loss = varpro_on_grid(grid, t, y, eps=s0_floor)

        loss_scale = max(best_loss, np.finfo(float).tiny)
        relative_change = abs(best_loss - trial_loss) / loss_scale
        if trial_loss < best_loss:
            best_log_t1, best_s0, best_loss = trial_log_t1, trial_s0, trial_loss

        # Check the grid width too, since the loss can flatten early.
        grid_width = grid[-1] - grid[0]
        if relative_change <= rel_tol and grid_width <= 1e-8:
            break

    return best_log_t1, float(np.log(best_s0)), best_loss


def reference_map(Y, rows, cols, t):
    """Fit each requested pixel and return arrays of log T1 and loss."""
    signal_data = np.asarray(Y, dtype=float)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    delays = np.asarray(t, dtype=float)

    if signal_data.ndim != 3 or delays.shape != (signal_data.shape[-1],):
        raise ValueError("Y must be shaped (row, column, delay), with matching t.")
    if rows.ndim != 1 or rows.shape != cols.shape:
        raise ValueError("rows and cols must be matching one-dimensional arrays.")

    log_t1_values = np.empty(rows.size)
    losses = np.empty(rows.size)
    for index, (row, col) in enumerate(zip(rows, cols)):
        pixel_signal = signal_data[row, col]
        log_t1_values[index], _, losses[index] = reference_fit(delays, pixel_signal)

    return log_t1_values, losses
