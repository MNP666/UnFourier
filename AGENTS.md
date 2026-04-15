# AGENTS.md

This file is the quick project brief for coding agents working in this
repository. It should be kept current and compact. Longer diagnostic notes live
in `docs2/`, especially `docs2/CODEX.md` for the historical M8 boundary analysis
and `docs2/spec_0p10.md` for the next planned iteration.

## Project Overview

**unFourier** is a Rust implementation of Indirect Fourier Transformation (IFT)
for SAXS data. It recovers the pair-distance distribution function `P(r)` from a
measured scattering curve `I(q)`. This is an ill-posed inverse problem, solved as
a Tikhonov-regularised linear system with automatic lambda selection by GCV,
L-curve, or Bayesian evidence.

The current active model is spline-only:

```text
P(r) = sum_j c_j B_j(r)
```

The solver estimates spline coefficients `c_j`; output `P(r)` is then evaluated
on a dense real-space grid. Do not treat spline coefficients as sampled `P(r)`.

## Core Commands

```bash
# Build
cargo build --release
# Binary: target/release/unfourier

# Default GCV run
target/release/unfourier data.dat --rmax 150 --output pr.dat --fit-output fit.dat --verbose

# Specific lambda-selection method
target/release/unfourier data.dat --method lcurve --rmax 100 -o pr.dat
target/release/unfourier data.dat --method bayes --rmax 150 -o pr.dat --verbose

# Manual lambda
target/release/unfourier data.dat --method manual --lambda 0.01 --rmax 150 -o pr.dat

# Spline basis controls
target/release/unfourier data.dat --n-basis 20 --rmax 150 -o pr.dat
target/release/unfourier data.dat --knot-spacing 7.5 --min-basis 12 --max-basis 48 --rmax 150 -o pr.dat

# Preprocessing controls
target/release/unfourier data.dat --qmin 0.01 --qmax 0.35 --snr-cutoff 1.0 --rebin 200 -o pr.dat

# Real-data validation against GNOM references
python3 Dev/validate_real_data.py

# Synthetic and exploratory validation
python3 Dev/gen_debye.py
python3 Dev/gen_sphere.py
python3 Dev/validate_spline.py
python3 Dev/sweep_noise.py
python3 Dev/sweep_smoothness.py
python3 Dev/sweep_knot_density.py
python3 Dev/monte_carlo_coverage.py --n 200 --k 5 --n-basis 20
```

`unfourier.toml` is loaded from the current working directory and fills unset CLI
options. CLI flags take precedence over TOML values. This matters for validation:
`Dev/validate_real_data.py` defaults to running `unfourier` from the repo root via
`--config-dir`, so the root `unfourier.toml` is used unless the script is told
otherwise.

## Pipeline

The binary pipeline in `src/main.rs` is:

```text
parse .dat
-> preprocess data
-> choose rmax and spline basis
-> build weighted kernel
-> solve spline coefficients
-> evaluate spline P(r) on dense output grid
-> write P(r) and optional I(q) fit
```

Preprocessing order is:

```text
negative handling
-> q-range / SNR filtering
-> log rebinning
```

The planned 0.10 Guinier preflight should run after negative handling and before
`QRangeSelector`.

## Module Map

| Module | Role |
|--------|------|
| `src/data.rs` | `SaxsData`; lenient 3-column `.dat` parser for `q`, `I(q)`, `sigma` |
| `src/basis.rs` | `BasisSet`; active `CubicBSpline`; boundary modes and free-to-full coefficient map |
| `src/bspline.rs` | Clamped knot vectors, Greville points, B-spline evaluation, sinc-kernel integration |
| `src/kernel.rs` | Builds weighted systems from data and basis |
| `src/regularise.rs` | `Regulariser`; projected spline D1/D2 smoothness, plus derivative helpers |
| `src/solver.rs` | Tikhonov solve for manual lambda; Cholesky/LU linear solve plus projected-gradient NNLS |
| `src/lambda_select.rs` | GCV, L-curve, Bayesian evidence grid search; stores selected constrained solution |
| `src/nonneg.rs` | Non-negativity strategies: projected-gradient NNLS default, `NoConstraint` available internally |
| `src/preprocess.rs` | Negative handling, q-range/SNR selector, log rebinning |
| `src/config.rs` | TOML config structs with `deny_unknown_fields` |
| `src/output.rs` | Writes evaluated `P(r)` and back-calculated `I(q)` fit |

## Solver Details

For a fixed user-facing lambda, unFourier solves:

```text
(K^T K + lambda_eff L^T L) c = K^T I
lambda_eff = lambda_user * tr(K^T K) / tr(L^T L)
```

The code conceptually computes `c = A^-1 b`, but it does not invert `A`.
Automatic lambda evaluation and manual solves use Cholesky where possible, with
LU fallback in the manual helper. The final production solution is constrained
with projected-gradient NNLS, so current CLI output is `P(r) >= 0`, not a signed
contrast-variation solution.

For automatic methods, GCV/L-curve/evidence quantities are computed from the
unconstrained linear solution because those criteria assume a linear estimator.
The stored solution for the selected lambda is the non-negativity-constrained
coefficient vector.

## Spline and Boundary Rules

The old top-hat/rectangular basis is historical. Active code should not re-add
`--basis`, `--npoints`, `UniformGrid`, or rect-vs-spline validation paths unless
there is a deliberate new design.

The key M8 lesson from `docs2/CODEX.md`:

```text
Do not edit reported P(r) after solving unless I(q) is recomputed from the same
coefficients. Prefer constraints in the basis/linear system over post-hoc edits.
```

Current spline boundary modes:

```text
value_zero       -> full coefficients [0, c..., 0]
value_slope_zero -> full coefficients [0, 0, c..., 0, 0]
```

The same free-to-full mapping must be used for:

1. Kernel columns.
2. Regularisation.
3. Output evaluation.

`Solution.coeffs` are spline control weights. Published `P(r)` comes from
`basis.evaluate_pr(&solution.coeffs, &output_grid)`.

## Config Notes

Main TOML sections:

```toml
[regularisation]
method = "gcv"       # gcv | lcurve | bayes | manual
lambda_min = 1e0     # lower bound for automatic search grid
lambda_max = 1e3

[preprocessing]
qmin = 0.01
qmax = 0.35
negative_handling = "clip"  # clip | omit | keep

[basis]
n_basis = 20
knot_spacing = 7.5
min_basis = 12
max_basis = 48

[constraints]
spline_boundary = "value_zero"  # value_zero | value_slope_zero
d1_smoothness = 0.1             # absent/0 = default, -1 = off, >0 = explicit
d2_smoothness = 1.0             # absent = default, >=0 = explicit
```

Important precedence:

1. CLI values win.
2. TOML fills only unset CLI options.
3. Built-in defaults apply last.

`lambda_min` and `lambda_max` affect only automatic lambda methods. They do not
affect `--method manual --lambda ...`.

## Validation Notes

Use `Dev/validate_real_data.py` for the current five real-data fixtures in
`data/dat_ref` and `data/prs_ref`. The script discovers matching `.dat`/`.out`
pairs, writes both `P(r)` and fit files, and produces `Dev/validation_plot.png`
with `P(r)`, `I(q)` fit, and endpoint diagnostics.

Use Debye-chain synthetic data as the primary noisy benchmark. Use sphere data
only for kernel/evaluation sanity checks; proportional noise around sphere-form
factor zeros makes it unsuitable as a regularisation benchmark.

Bayesian error bars are written as a third `P(r)` column for `--method bayes`.
The current posterior helper exposes diagonal coefficient standard deviations
and does not include regularisation bias, so coverage is expected to be imperfect.

## Current / Next Iteration

0.9 resolved the main spline boundary architecture:

1. Spline-only active basis.
2. Coefficients separated from evaluated `P(r)`.
3. No post-solve boundary clamping.
4. Boundary conditions represented in the spline coefficient map.
5. Projected D1/D2 regularisation through the same map.

0.10 planning is in `docs2/spec_0p10.md`. The next feature is a Guinier low-q
preflight:

1. Report-only scan over increasing low-q truncations.
2. Optional `--auto-qmin guinier` mutation mode.
3. Explicit `--qmin` must win over automatic suggestions.
4. No promise that automatic trimming is generally reliable before 1.0.

## Development Cautions

1. Keep changes scoped; this repo is exploratory but the current spline path is
   coherent.
2. Do not restore historical rect-basis product surfaces by accident.
3. Do not mutate output `P(r)` independently of the solved coefficients.
4. Be careful with `unfourier.toml` in the current working directory when running
   validation or comparing lambda behavior.
5. Preserve typed library errors where practical; the binary can use `anyhow`.
6. Prefer adding focused tests for math helpers and use Python scripts for broader
   numerical validation.
