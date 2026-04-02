# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**unFourier** is a Rust implementation of Indirect Fourier Transformation (IFT) for Small Angle X-ray Scattering (SAXS) data analysis. It recovers the pair distance distribution function P(r) from measured scattering curves I(q) — an ill-posed inverse problem solved via Tikhonov regularisation with automatic λ selection (GCV, L-curve, or manual).

## Commands

```bash
# Build
cargo build --release
# Binary at target/release/unfourier

# Run with GCV (default)
unfourier data.dat --rmax 150 --output pr.dat --fit-output fit.dat --verbose

# Run with specific method
unfourier data.dat --method lcurve --rmax 100 -o pr.dat

# Run Bayesian IFT (produces 3-column output: r, P(r), σ_P(r))
unfourier data.dat --method bayes --rmax 150 --output pr.dat --verbose

# Generate synthetic test fixtures (requires numpy, matplotlib, scipy)
python Dev/gen_debye.py        # Noisy Debye chain data (primary benchmark)
python Dev/gen_sphere.py       # Sphere data (kernel accuracy, not noisy tests)

# Plot P(r) vs reference
python Dev/plot_pr.py

# M3 validation: sweep noise levels across GCV/L-curve
python Dev/sweep_noise.py

# M4 validation: Monte Carlo coverage check for Bayesian error bars
python Dev/monte_carlo_coverage.py --n 200 --k 5
```

## Architecture

The pipeline in `main.rs` is: **parse → preprocess → r-grid → build kernel → solve → non-negativity → output**.

Each stage is abstracted behind a trait, making the system extensible without modifying the pipeline:

| Module | Trait | Current Implementation |
|--------|-------|------------------------|
| `data.rs` | — | `SaxsData`: holds q, I(q), σ(q); parses 3-column `.dat` files |
| `basis.rs` | `BasisSet` | `UniformGrid`: piecewise constant rectangular bins |
| `kernel.rs` | — | Builds weighted system matrix K from basis + data |
| `regularise.rs` | `Regulariser` | `SecondDerivative`: finite-difference curvature penalty |
| `solver.rs` | `Solver` | `TikhonovSolver`: SVD-based with L-curve curvature analysis |
| `lambda_select.rs` | `LambdaSelector` | `GcvSelector`, `LCurveSelector`, `BayesianEvidence` |
| `nonneg.rs` | `NonNegativityStrategy` | `IterativeClipping`: re-solves with zero-clamped bins |
| `preprocess.rs` | `Preprocessor` | `Identity` (no-op); chains via `PreprocessingPipeline` |
| `output.rs` | — | Writes P(r) and back-calculated I(q) fit |

**Key design principles:**
- Trait interfaces are designed so future work slots in without touching the pipeline: B-spline basis (M5), resampling/Guinier preprocessing (M6)
- Library uses `thiserror` for typed errors; binary uses `anyhow`
- Each λ evaluation is a pure function — designed for `rayon::par_iter()` when needed

## Milestones

| M | Feature | Status |
|---|---------|--------|
| M1 | Naive least-squares end-to-end | ✅ |
| M2 | Tikhonov with manual λ | ✅ |
| M3 | Automatic λ (GCV + L-curve) | ✅ |
| M4 | Bayesian IFT with error bars | ✅ |
| M5 | Cubic B-spline basis | planned |
| M6 | Full preprocessing pipeline | planned |

## Validation Notes

- **Debye chain** (`Dev/gen_debye.py`) is the primary noisy benchmark. Use this for validating regularisation and λ selection.
- **Sphere** (`Dev/gen_sphere.py`) is used only for kernel accuracy tests (analytic P(r) is exact). It is **unsuitable for noisy-data validation** because the proportional noise model diverges near zeros of I(q) — see `saxs_ift_postmortem.md` for the full analysis.
- **Bayesian error bars** (`--method bayes`) produce a 3-column P(r) output. Calibration is checked via `Dev/monte_carlo_coverage.py` — expect ~60–68% coverage at ±1σ; systematic under-coverage reflects regularisation bias not captured by the posterior.
- There is no formal test suite; validation is done via the Python scripts in `Dev/`.
