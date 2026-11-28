import argparse
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from model import make_fgh
from phantom import brain_pixels, create_labels, make_delays, make_noisy_data, tissue_maps
from reference import reference_map
from registry import SOLVERS


SAMPLE_SEED = 42
NOISE_SEED = 2025
LOSS_MARGIN = 0.001
ORACLE_ABS_TOL = 1e-12

# The stored status codes are part of the results file format.
STATUS_CODES = {
    "unfinished": -1,
    "converged": 0,
    "max_iter": 1,
    "diverged": 2,
    "linalg_error": 3,
}
RUN_FIELDS = ("f_final", "u_final", "v_final", "iters", "time", "status_code", "success")


def make_start(y, T1_true, factor):
    u0 = np.log(factor * T1_true)
    v0 = np.log(max(1e-3, np.max(y)))
    return np.array([u0, v0])


def is_success(f_final, f_ref):
    return bool(np.isfinite(f_final) and np.isfinite(f_ref) and f_final <= (1 + LOSS_MARGIN) * f_ref)


def noise_seed(snr):
    snr_bits = int(np.float64(snr).view(np.uint64))
    return NOISE_SEED, snr_bits


def _save_checkpoint(path, results):
    # Write beside the destination and replace it only after the archive is closed.
    path.parent.mkdir(parents=True, exist_ok=True)
    file = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
    temporary_path = Path(file.name)
    try:
        with file:
            np.savez_compressed(file, **results)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_benchmark(sample_fraction=1.0, snrs=(10, 20, 40), factors=(0.25, 1, 4), out="results.npz"):
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1].")

    snrs = np.asarray(snrs, dtype=float)
    factors = np.asarray(factors, dtype=float)
    if snrs.ndim != 1 or snrs.size == 0 or not np.all(snrs > 0):
        raise ValueError("snrs must be a nonempty sequence of positive values.")
    if factors.ndim != 1 or factors.size == 0 or not np.all(np.isfinite(factors) & (factors > 0)):
        raise ValueError("factors must be a nonempty sequence of positive finite values.")

    labels = create_labels()
    t1_map, s0_map = tissue_maps(labels)
    delays = make_delays()
    rows, cols = brain_pixels(labels)

    sample_size = max(1, round(sample_fraction * rows.size))
    rng = np.random.default_rng(SAMPLE_SEED)
    selected = np.sort(rng.choice(rows.size, size=sample_size, replace=False))
    rows, cols = rows[selected], cols[selected]
    true_t1 = t1_map[rows, cols]

    shape = (snrs.size, factors.size, rows.size, len(SOLVERS))
    oracle_shape = (snrs.size, rows.size)
    results = {
        "rows": rows,
        "cols": cols,
        "snrs": snrs,
        "factors": factors,
        "solver_names": np.array(list(SOLVERS)),
        "delays": delays,
        "signals": np.full((*oracle_shape, delays.size), np.nan),
        "f_ref": np.full(oracle_shape, np.nan),
        "u_ref": np.full(oracle_shape, np.nan),
        "f_final": np.full(shape, np.nan),
        "u_final": np.full(shape, np.nan),
        "v_final": np.full(shape, np.nan),
        "iters": np.full(shape, -1, dtype=int),
        "time": np.full(shape, np.nan),
        "status_code": np.full(shape, STATUS_CODES["unfinished"], dtype=np.int8),
        "success": np.zeros(shape, dtype=bool),
        "oracle_issue": np.zeros(shape, dtype=bool),
    }

    path = Path(out)
    for snr_index, snr in enumerate(snrs):
        noisy = make_noisy_data(t1_map, s0_map, delays, snr, seed=noise_seed(snr))
        curves = noisy[rows, cols]
        u_ref, f_ref = reference_map(noisy, rows, cols, delays)
        results["signals"][snr_index] = curves
        results["u_ref"][snr_index] = u_ref
        results["f_ref"][snr_index] = f_ref
        objectives = [make_fgh(delays, curve) for curve in curves]

        for start_index, factor in enumerate(factors):
            for voxel, (f, g, h) in enumerate(objectives):
                x0 = make_start(curves[voxel], true_t1[voxel], factor)
                target = f_ref[voxel]
                oracle_gap = max(ORACLE_ABS_TOL, LOSS_MARGIN * target)

                for solver_index, solver in enumerate(SOLVERS.values()):
                    index = (snr_index, start_index, voxel, solver_index)
                    start = x0.copy()
                    before = perf_counter()
                    fit = solver(start, f, g, h)
                    elapsed = perf_counter() - before

                    loss = fit["f"]
                    results["f_final"][index] = loss
                    results["u_final"][index], results["v_final"][index] = fit["x"]
                    results["iters"][index] = fit["iters"]
                    results["time"][index] = elapsed
                    results["status_code"][index] = STATUS_CODES[fit["status"]]
                    results["success"][index] = is_success(loss, target)
                    results["oracle_issue"][index] = (
                        np.isfinite(loss) and np.isfinite(target) and loss < target - oracle_gap
                    )

            _save_checkpoint(path, results)
            print(f"SNR {snr:g} | start {factor:g}x | checkpoint saved to {path}", flush=True)

    print(f"Saved {np.prod(shape):,} runs across {rows.size:,} voxels.")
    print(f"Flagged {np.count_nonzero(results['oracle_issue']):,} runs for oracle consistency.")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=float, default=1.0, help="fraction of brain voxels to fit")
    parser.add_argument("--out", default="results.npz", help="destination for the benchmark archive")
    args = parser.parse_args()
    run_benchmark(sample_fraction=args.sample, out=args.out)


if __name__ == "__main__":
    main()
