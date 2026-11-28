# MRI T1 Optimization

How do six optimization methods compare when estimating T1 from noisy MRI recovery curves? We simulate voxel signals with known T1 and S0, fit the same noisy data with each solver, and compare fit quality, T1 accuracy, and computational cost.

## Team

**Team name** — GRADIENT ASCENT

| Member name | Member ID |
|---|---|
| Hemish Sri Chandra | BA2025059 |
| Srikar Sai Sagala | BA2025042 |
| Sai Srujan | BA2025032 |

## Libraries used

- **NumPy** for signal simulation, array operations, linear algebra, and result archives.
- **Matplotlib** for recovery curves, convergence plots, and T1 maps.

The [project report](report.pdf) documents the mathematical formulation, solver implementations, and full benchmark results.

## Quick start

Run the commands below from the repository root. The workflow was tested with Python 3.14.2, NumPy 2.4.1, and Matplotlib 3.10.8.

```sh
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS or Linux, or `./.venv/Scripts/Activate.ps1` in Windows PowerShell. Then install the dependencies.

```sh
python -m pip install numpy==2.4.1 matplotlib==3.10.8
```

### Smoke test

Fit 5% of the brain voxels and write the five sample figures to a separate folder.

```sh
python benchmark.py --sample 0.05 --out sample_results.npz
python plots.py --results sample_results.npz --out-dir figures/sample
```

### Full experiment

Use the full run for reporting. It fits all 2,112 brain voxels for a total of 114,048 solver runs. The benchmark writes `results.npz`, and the plot script reads it and writes five PNGs to `figures/`.

```sh
python benchmark.py
python plots.py
```

The benchmark defaults to `results.npz` at every sample fraction. The smoke-test `--out` and `--out-dir` above keep its archive and figures separate from the full run. Checkpoints are saved after each SNR and starting-factor block. Generate plots once the benchmark completes.

## Experiment design

At delay `t`, the model predicts `S(t) = S0 * (1 - exp(-t / T1))`. Each solver minimizes half the sum of squared residuals, `0.5 * sum((S(t_i) - y_i)**2)`, over `[u, v] = [ln(T1), ln(S0)]`. The fitted values are `exp(u)` and `exp(v)`.

| Setting | Default |
|---|---|
| Phantom | 64 × 64 slice with 2,112 brain voxels across white matter, grey matter, CSF, and a synthetic lesion |
| Recovery delays | 10 geometrically spaced delays from 100 to 5,000 ms |
| Noise | SNRs 10, 20, and 40, with Gaussian noise of standard deviation `S0_true / SNR` |
| Starting point | T1 starts at 0.25×, 1×, or 4× the true T1. S0 starts at the maximum measured signal, with a floor of `0.001`. |
| Solvers | Steepest Descent (`SD`), Classical Newton (`Newton`), Modified Newton (`ModNewton`), Trust Region (`TR`), DFP (`DFP`), and BFGS (`BFGS`) |
| Solver defaults | 200 iterations and gradient tolerance `1e-8` |

The reference routine searches a grid of log-T1 values and fits S0 by least squares at each point. The benchmark uses its best loss `f_ref` for that noisy curve. A run meets the loss target when both losses are finite and `f_final <= 1.001 * f_ref`. The solver's stopping status is recorded separately. A loss lower than `f_ref` by more than `max(1e-12, 0.001 * f_ref)` also raises an oracle consistency flag.

Absolute T1 error is `abs(exp(u_final) - T1_true)`, in milliseconds. Tissue RMSE uses the known simulated T1 values as well. The oracle's T1 RMSE shows the estimation error that remains when fitting the noisy curve with the reference routine.

## Results archive and figures

| Archive fields | Axis order or meaning |
|---|---|
| `f_final`, `u_final`, `v_final`, `iters`, `time`, `status_code`, `success`, `oracle_issue` | `(SNR, start factor, voxel, solver)`, or `(3, 3, 2112, 6)` for the full run. Runtime is in seconds. |
| `f_ref`, `u_ref` | `(SNR, voxel)` |
| `signals` | `(SNR, voxel, delay)` |
| `snrs`, `factors`, `rows`, `cols`, `solver_names`, `delays` | Values and coordinates for those axes. `rows[j]` and `cols[j]` identify voxel `j`. |

Status codes are `-1` unfinished, `0` converged, `1` iteration limit, `2` diverged, and `3` linear algebra error.

The figure numbers link to the published full-run plots from the November 25, 2026 audit.

| Published figure | Default output | What it shows |
|---|---|---|
| [1](figures/full/figure_1_trajectories.png) | `figures/figure_1_trajectories.png` | Solver paths on the loss contours for an example voxel at SNR 20 and a 4× start, with the reference fit marked. |
| [2](figures/full/figure_2_convergence.png) | `figures/figure_2_convergence.png` | Base-10 log of the gradient norm versus iteration for the same voxel. |
| [3](figures/full/figure_3_success_and_cost.png) | `figures/figure_3_success_and_cost.png` | Loss-target success rates across SNRs and starts. Median step counts and runtimes use runs that met the target. |
| [4](figures/full/figure_4_error_maps.png) | `figures/figure_4_error_maps.png` | True T1, fitted T1, and absolute error for BFGS at SNR 20 and a 1× start. Background and unsampled pixels are masked. |
| [5](figures/full/figure_5_tissue_rmse.png) | `figures/figure_5_tissue_rmse.png` | T1 RMSE by tissue and SNR for the six solvers and the oracle at a 1× start. Each comparison uses the same voxels with usable fits for all methods. Finite fits that reach the iteration limit can count. |

## Verified runs

The seeded 5% smoke test completed 5,724 fits across 106 voxels. Of those, 5,149 met the loss target (90.0%), zero raised an oracle consistency flag, and the plotting command produced all five sample PNGs.

The completed full run used in the report met the loss target in 101,891 of 114,048 fits (89.3%) and raised zero oracle consistency flags. Each solver had 19,008 runs across the default SNRs and starts. The table pools those runs and calculates medians over the fits that met the target. Runtime reflects the machine used for the November 25, 2026 audit.

| Solver | Runs meeting target | Median steps | Median runtime (ms) |
|---|---:|---:|---:|
| `SD` | 18,273 | 146 | 15.12 |
| `Newton` | 7,882 | 3 | 0.61 |
| `ModNewton` | 19,004 | 6 | 1.01 |
| `TR` | 18,889 | 90 | 19.30 |
| `DFP` | 18,835 | 17 | 1.71 |
| `BFGS` | 19,008 | 11 | 1.07 |

For the BFGS error map in Figure 4, the median absolute T1 error was 66.8 ms across all 2,112 brain voxels. The full plotting command generated all five PNGs.

The CSF oracle RMSE at SNR 10 includes an extreme fit at voxel `(27, 36)`. The reference estimated T1 at about `7.02e20` ms for a simulated T1 of 4,000 ms. That fit achieved a lower loss on the noisy measurements than a fit with T1 fixed at 4,000 ms, so the loss target and T1 error capture different outcomes.

## Repository map and reproducibility

| File | Role |
|---|---|
| [`phantom.py`](phantom.py) | Generates the tissue labels, true parameter maps, delays, and noisy curves. |
| [`model.py`](model.py) | Builds the per-voxel loss, gradient, and Hessian. |
| [`reference.py`](reference.py) | Computes the independent grid-search fit for each noisy curve. |
| [`solvers.py`](solvers.py), [`quasi_newton.py`](quasi_newton.py), [`registry.py`](registry.py) | Implement the six methods and expose a common solver interface. |
| [`benchmark.py`](benchmark.py) | Runs the solvers against the shared reference targets and saves the archive. |
| [`plots_paths.py`](plots_paths.py), [`plots.py`](plots.py) | Draw the example solver paths and generate the five figures. |
| [`demo_one_voxel.py`](demo_one_voxel.py), [`toys.py`](toys.py) | Run the single-voxel demo and provide small optimization problems for solver checks. |

Seed `42` selects the same voxels across all SNRs, starts, and solvers. Noise uses base seed `2025` keyed by the SNR value, so reordering the SNRs preserves each noisy dataset. Use `python demo_one_voxel.py` for a quick solver check. `python benchmark.py --help` and `python plots.py --help` list the available options.
