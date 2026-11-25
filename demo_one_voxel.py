"""Run the solvers on one noisy recovery curve from two starting points."""

import numpy as np

from model import LOG_PARAMETER_LIMIT, make_fgh, signal
from quasi_newton import bfgs, dfp
from reference import reference_fit
from solvers import ALL_SOLVERS as CORE_SOLVERS

QUASI_SOLVERS = {
    "Quasi-Newton (DFP)": dfp,
    "Quasi-Newton (BFGS)": bfgs,
}
ALL_SIX_SOLVERS = {**CORE_SOLVERS, **QUASI_SOLVERS}


def run_single_voxel_experiment(T1_true=1300.0, S0_true=0.85, snr=20, seed=42):
    delays = np.linspace(100.0, 3000.0, 10)
    x_true = np.log([T1_true, S0_true])

    clean = signal(x_true, delays)
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(scale=S0_true / snr, size=delays.shape)
    f, g, h = make_fgh(delays, noisy)

    starts = {
        "Near start (T1=1000, S0=1.0)": np.array([np.log(1000.0), np.log(1.0)]),
        "Far start  (T1=150,  S0=0.2)": np.array([np.log(150.0), np.log(0.2)]),
    }

    ref_log_t1, ref_log_s0, ref_loss = reference_fit(delays, noisy)
    print(f"True values   T1 = {T1_true:.1f} ms | S0 = {S0_true:.3f} | SNR = {snr}")
    print(
        f"Grid fit      T1 = {np.exp(ref_log_t1):.1f} ms | "
        f"S0 = {np.exp(ref_log_s0):.3f} | loss = {ref_loss:.3e}"
    )

    all_runs = {}
    for label, x0 in starts.items():
        print(f"\n{label}")
        header = (
            f"{'Solver':<20} | {'Steps':>5} | {'Loss':>10} | "
            f"{'T1 (ms)':>9} | {'S0':>7} | {'T1 error':>8} | Status"
        )
        print(header)
        print("-" * len(header))

        runs_for_start = {}
        for name, solver in ALL_SIX_SOLVERS.items():
            if name in QUASI_SOLVERS:
                result = solver(x0, f, g, h)
                steps, final_loss = result["iters"], result["f"]
                status = result["status"]
                converged = status == "converged"
            else:
                result = solver(f, g, h, x0)
                steps, final_loss = result["nit"], result["f_vals"][-1]
                status, converged = result["reason"], result["converged"]
            runs_for_start[name] = result

            estimate = result["x"]
            if converged and np.all(np.abs(estimate) <= LOG_PARAMETER_LIMIT):
                t1_estimate, s0_estimate = np.exp(estimate)
                error_percent = abs(t1_estimate - T1_true) / T1_true * 100.0
                estimates = f"{t1_estimate:9.2f} | {s0_estimate:7.4f} | {error_percent:7.2f}%"
            else:
                estimates = f"{'-':>9} | {'-':>7} | {'-':>8}"

            print(
                f"{name:<20} | {steps:5d} | {final_loss:10.3e} | "
                f"{estimates} | {status}"
            )

        all_runs[label] = runs_for_start

    return delays, noisy, x_true, all_runs


if __name__ == "__main__":
    run_single_voxel_experiment()
