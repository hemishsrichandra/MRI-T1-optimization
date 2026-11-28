import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark import RUN_FIELDS, STATUS_CODES, noise_seed
from model import LOG_PARAMETER_LIMIT
from phantom import TISSUE_NAMES, create_labels, make_delays, make_noisy_data, tissue_maps
from plots_paths import FIGURE_DIR, SOLVER_COLORS, fig_convergence, fig_trajectories, save_figure


def load_results(path):
    with np.load(path, allow_pickle=False) as archive:
        res = {name: archive[name] for name in archive.files}

    vectors = ("snrs", "factors", "rows", "cols", "solver_names")
    for name in (*vectors, *RUN_FIELDS, "u_ref", "f_ref"):
        if name not in res:
            raise ValueError(f"The results file is missing {name}.")
    for name in vectors:
        if res[name].ndim != 1 or res[name].size == 0:
            raise ValueError(f"{name} must be a nonempty vector.")
    if res["cols"].shape != res["rows"].shape:
        raise ValueError("rows and cols must describe the same number of voxels.")

    run_shape = tuple(res[name].size for name in ("snrs", "factors", "rows", "solver_names"))
    ref_shape = (run_shape[0], run_shape[2])
    for name in (*RUN_FIELDS, "oracle_issue"):
        if name in res and res[name].shape != run_shape:
            raise ValueError(f"{name} must have shape {run_shape} (SNR, start, voxel, solver).")
    for name in ("u_ref", "f_ref"):
        if res[name].shape != ref_shape:
            raise ValueError(f"{name} must have shape {ref_shape} (SNR, voxel).")
    if "signals" in res:
        delays = res.get("delays")
        if delays is None or delays.ndim != 1 or res["signals"].shape != (*ref_shape, delays.size):
            raise ValueError("Stored signals must have shape (SNR, voxel, delay) with matching delays.")
    return res


def _median_by_start(values, keep):
    return [np.median(block[mask]) if mask.any() else np.nan for block, mask in zip(values, keep)]


def fig_success_and_cost(res, *, out=FIGURE_DIR / "figure_3_success_and_cost.png"):
    snrs = res["snrs"]
    starts = np.arange(res["factors"].size)
    start_labels = [f"{factor:g}x" for factor in res["factors"]]
    completed = res["status_code"] != STATUS_CODES["unfinished"]
    succeeded = res["success"] & completed
    counts = completed.sum(axis=2)
    rates = np.full(counts.shape, np.nan)
    np.divide(100 * succeeded.sum(axis=2), counts, out=rates, where=counts > 0)

    fig, axes = plt.subplots(3, snrs.size, figsize=(max(6, 3.5 * snrs.size), 8),
                             sharex="col", sharey="row", squeeze=False, layout="constrained")
    for snr_index, snr in enumerate(snrs):
        axes[0, snr_index].set_title(f"SNR {snr:g}", fontweight="bold")
        for solver_index, name in enumerate(res["solver_names"]):
            keep = succeeded[snr_index, :, :, solver_index]
            series = (
                rates[snr_index, :, solver_index],
                _median_by_start(res["iters"][snr_index, :, :, solver_index], keep),
                _median_by_start(1000 * res["time"][snr_index, :, :, solver_index], keep),
            )
            for ax, values in zip(axes[:, snr_index], series):
                ax.plot(starts, values, color=SOLVER_COLORS[name], marker="o", markersize=3.5,
                        linewidth=1.5, label=name)

        axes[0, snr_index].set_ylim(-3, 103)
        axes[-1, snr_index].set_xticks(starts, start_labels)
        axes[-1, snr_index].set_xlabel("Starting T1 factor")

    for ax, label in zip(axes[:, 0], ("Success (%)", "Median iterations", "Median runtime (ms)")):
        ax.set_ylabel(label)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.24)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=len(names), frameon=False, fontsize=9)
    fig.suptitle("Solver success and cost\nRates use completed runs. Both medians use runs meeting the loss target.",
                 fontsize=12)
    return save_figure(fig, out)


def _default_start(res):
    return np.argmin(np.abs(np.log(res["factors"])))


def _t1_estimates(res, snr_index, start_index):
    u = res["u_final"][snr_index, start_index]
    v = res["v_final"][snr_index, start_index]
    f = res["f_final"][snr_index, start_index]
    status = res["status_code"][snr_index, start_index]
    usable = (status == STATUS_CODES["converged"]) | (status == STATUS_CODES["max_iter"])
    usable &= np.isfinite(f) & (np.abs(u) <= LOG_PARAMETER_LIMIT) & (np.abs(v) <= LOG_PARAMETER_LIMIT)
    return np.exp(np.where(usable, u, np.nan))


def fig_error_maps(res, labels, T1_true, snr, *, out=FIGURE_DIR / "figure_4_error_maps.png"):
    matches = np.flatnonzero(res["snrs"] == snr)
    if matches.size == 0:
        raise ValueError(f"SNR {snr:g} is not in the results file.")
    solver_names = res["solver_names"].tolist()
    solver_name = "BFGS" if "BFGS" in solver_names else solver_names[0]
    start_index = _default_start(res)
    rows, cols = res["rows"], res["cols"]
    tissue = labels[rows, cols] > 0

    estimates = _t1_estimates(res, matches[0], start_index)[:, solver_names.index(solver_name)]
    truth = np.full(labels.shape, np.nan)
    fitted = np.full(labels.shape, np.nan)
    truth[rows[tissue], cols[tissue]] = T1_true[rows[tissue], cols[tissue]]
    fitted[rows[tissue], cols[tissue]] = estimates[tissue]
    error = np.abs(fitted - truth)

    t1_colors = plt.get_cmap("viridis").with_extremes(bad="#e8edf2")
    error_colors = plt.get_cmap("magma").with_extremes(bad="#e8edf2")
    t1_max = np.max(T1_true[labels > 0], initial=1)
    error_max = np.max(error[np.isfinite(error)], initial=1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), layout="constrained")
    truth_image = axes[0].imshow(truth, vmin=0, vmax=t1_max, cmap=t1_colors, interpolation="nearest")
    axes[1].imshow(fitted, vmin=0, vmax=t1_max, cmap=t1_colors, interpolation="nearest")
    error_image = axes[2].imshow(error, vmin=0, vmax=error_max, cmap=error_colors, interpolation="nearest")
    extend = "max" if np.any(fitted > t1_max) else "neither"
    fig.colorbar(truth_image, ax=axes[:2], location="bottom", shrink=0.72, label="T1 (ms)", extend=extend)
    fig.colorbar(error_image, ax=axes[2], location="bottom", shrink=0.85, label="Absolute error (ms)")
    for ax, title in zip(axes, ("True T1", "Estimated T1", "Absolute T1 error")):
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()

    usable = np.count_nonzero(np.isfinite(fitted))
    sampled = np.count_nonzero(tissue)
    factor = res["factors"][start_index]
    fig.suptitle(f"{solver_name} at SNR {snr:g}, start {factor:g}x\n"
                 f"{usable:,}/{sampled:,} sampled tissue voxels have usable fits",
                 fontsize=12)
    return save_figure(fig, out)


def fig_tissue_rmse(res, labels, T1_true, *, out=FIGURE_DIR / "figure_5_tissue_rmse.png"):
    start_index = _default_start(res)
    rows, cols = res["rows"], res["cols"]
    tissue = labels[rows, cols]
    truth = T1_true[rows, cols]
    names = [*res["solver_names"], "Oracle"]
    estimates = []
    for snr_index, ref_u in enumerate(res["u_ref"]):
        fitted = _t1_estimates(res, snr_index, start_index)
        oracle = np.exp(np.where(np.abs(ref_u) <= LOG_PARAMETER_LIMIT, ref_u, np.nan))
        estimates.append(np.column_stack((fitted, oracle)))

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), layout="constrained")
    x = np.arange(res["snrs"].size)
    for ax, label in zip(axes.flat, (1, 2, 3, 4)):
        selected = tissue == label
        total = np.count_nonzero(selected)
        counts = []
        rmse = np.full((x.size, len(names)), np.nan)
        for snr_index, values in enumerate(estimates):
            # All curves in a group use the same voxels, including the oracle.
            shared = selected & np.isfinite(values).all(axis=1)
            counts.append(np.count_nonzero(shared))
            if shared.any():
                errors = values[shared] - truth[shared, None]
                rmse[snr_index] = np.sqrt(np.mean(errors ** 2, axis=0))

        for name, values in zip(names, rmse.T):
            oracle = name == "Oracle"
            color = "#171717" if oracle else SOLVER_COLORS[name]
            ax.plot(x, values, color=color, linestyle="--" if oracle else "-",
                    marker="o", markersize=5.2 if oracle else 3.8, linewidth=1.3,
                    markerfacecolor="none" if oracle else color, label=name)
        title = TISSUE_NAMES[label]
        positive = rmse[np.isfinite(rmse) & (rmse > 0)]
        if positive.size and positive.max() > 100 * max(1, positive.min()):
            ax.set_yscale("symlog", linthresh=10)
            ax.yaxis.get_major_locator().set_params(numticks=6)
            title += " (log spacing above 10 ms)"
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x, [f"{snr:g}\n{count}/{total}" for snr, count in zip(res["snrs"], counts)])
        ax.set_xlabel("SNR and usable / sampled voxels")
        ax.set_ylabel("T1 RMSE (ms)")
        ax.grid(axis="y", alpha=0.24)

    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=len(names), frameon=False, fontsize=9)
    fig.suptitle(f"T1 error by tissue at start {res['factors'][start_index]:g}x\n"
                 "Shared voxels with finite fits for all methods. Finite iteration-limit fits count.",
                 fontsize=12)
    return save_figure(fig, out)


def _example_voxel(labels, rows, cols):
    white_matter = np.flatnonzero(labels[rows, cols] == 1)
    choices = white_matter if white_matter.size else np.arange(rows.size)
    center = (np.array(labels.shape) - 1) / 2
    distance = (rows[choices] - center[0]) ** 2 + (cols[choices] - center[1]) ** 2
    return choices[np.argmin(distance)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results.npz", help="completed benchmark archive")
    parser.add_argument("--out-dir", type=Path, default=FIGURE_DIR, help="folder for the five PNGs")
    args = parser.parse_args()
    res = load_results(args.results)
    if np.any(res["status_code"] == STATUS_CODES["unfinished"]):
        parser.error("The benchmark has unfinished runs. Use a completed results archive.")

    labels = create_labels()
    T1_true, S0_true = tissue_maps(labels)
    snr_index = np.argmin(np.abs(res["snrs"] - 20))
    snr = res["snrs"][snr_index]
    factor = res["factors"].max()
    voxel = _example_voxel(labels, res["rows"], res["cols"])
    row, col = res["rows"][voxel], res["cols"][voxel]

    if "signals" in res:
        delays = res["delays"]
        curve = res["signals"][snr_index, voxel]
    else:
        delays = res["delays"] if "delays" in res else make_delays()
        curve = make_noisy_data(T1_true, S0_true, delays, snr, seed=noise_seed(snr))[row, col]

    output = args.out_dir
    paths = (
        fig_trajectories(delays, curve, T1_true[row, col], factor, snr=snr,
                         out=output / "figure_1_trajectories.png"),
        fig_convergence(delays, curve, T1_true[row, col], factor, snr=snr,
                        out=output / "figure_2_convergence.png"),
        fig_success_and_cost(res, out=output / "figure_3_success_and_cost.png"),
        fig_error_maps(res, labels, T1_true, snr, out=output / "figure_4_error_maps.png"),
        fig_tissue_rmse(res, labels, T1_true, out=output / "figure_5_tissue_rmse.png"),
    )
    print(f"Example voxel ({row}, {col}) at SNR {snr:g}, start {factor:g}x.")
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
