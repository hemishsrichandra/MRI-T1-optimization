import numpy as np
from model import signal, make_fgh
from solvers import ALL_SOLVERS as CORE_SOLVERS

# To be removed after Srikar implements quasi_newton.py
try:
    from quasi_newton import dfp, bfgs
    QUASI_SOLVERS = {
        "Quasi-Newton (DFP)": dfp,
        "Quasi-Newton (BFGS)": bfgs,
    }
except ImportError:
    QUASI_SOLVERS = {}

ALL_SIX_SOLVERS = {**CORE_SOLVERS, **QUASI_SOLVERS}

def run_single_voxel_experiment(T1_true=1300.0, S0_true=0.85, snr=20, seed=42):
    t = np.linspace(100.0, 3000.0, 10)
    x_true = np.array([np.log(T1_true), np.log(S0_true)], dtype=float)

    clean_y = signal(x_true, t)
    rng = np.random.default_rng(seed)
    noisy_y = clean_y + rng.normal(loc=0.0, scale=S0_true / snr, size=t.shape)

    f, g, h = make_fgh(t, noisy_y)

    starts = {
        "Near start (T1=1000, S0=1.0)": np.array([np.log(1000.0), np.log(1.0)]),
        "Far start  (T1=150,  S0=0.2)": np.array([np.log(150.0), np.log(0.2)]),
    }

    print(f"Ground Truth: T1 = {T1_true:.1f} ms, S0 = {S0_true:.3f} (SNR = {snr})")

    all_runs = {}

    for label, x0 in starts.items():
        print()
        print(f"Initialization: {label}  ->  x0 = [{x0[0]:.3f}, {x0[1]:.3f}]")
        print("-" * 86)
        header = (
            f"{'Solver':<23} | {'Conv':<5} | {'Iters':>5} | {'Final Loss':>11} | "
            f"{'T1_est (ms)':>11} | {'S0_est':>7} | {'T1 Err %':>8}"
        )
        print(header)
        print("-" * 86)

        runs_for_start = {}
        for name, solver_fn in ALL_SIX_SOLVERS.items():
            res = solver_fn(f, g, h, x0)
            runs_for_start[name] = res

            u_est, v_est = res["x"]
            # Guard against overflowed coordinates if Classical Newton diverged
            if res["converged"] and np.isfinite(u_est) and abs(u_est) < 50:
                T1_est = np.exp(u_est)
                S0_est = np.exp(v_est)
                err_pct = abs(T1_est - T1_true) / T1_true * 100.0
                t1_str = f"{T1_est:11.2f}"
                s0_str = f"{S0_est:7.4f}"
                err_str = f"{err_pct:7.2f}%"
            else:
                t1_str = f"{'DIVERGED':>11}"
                s0_str = f"{'---':>7}"
                err_str = f"{'---':>8}"

            loss_val = res["f_vals"][-1]
            loss_str = f"{loss_val:11.3e}" if np.isfinite(loss_val) else f"{'inf':>11}"

            print(
                f"{name:<23} | {str(res['converged']):<5} | {res['nit']:>5d} | "
                f"{loss_str} | {t1_str} | {s0_str} | {err_str}"
            )

        all_runs[label] = runs_for_start
        print()

    return t, noisy_y, x_true, all_runs


if __name__ == "__main__":
    run_single_voxel_experiment()