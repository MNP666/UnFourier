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

# Choose the number of free cubic B-spline basis parameters
unfourier data.dat --n-basis 20 --rmax 150 -o pr.dat

# Preprocessing options: q-range, SNR cutoff, log-rebinning
unfourier data.dat --qmin 0.01 --qmax 0.35 --snr-cutoff 1.0 --rebin 200 -o pr.dat

# TOML config file (unfourier.toml in CWD) overrides unset CLI args
# Sections: [regularisation], [preprocessing], [basis], [constraints]

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

The pipeline in `main.rs` is: **parse → preprocess → r-grid → build kernel → solve coefficients → evaluate spline P(r) → output**.

Each stage is abstracted behind a trait, making the system extensible without modifying the pipeline:

| Module | Trait | Current Implementation |
|--------|-------|------------------------|
| `data.rs` | — | `SaxsData`: holds q, I(q), σ(q); parses 3-column `.dat` files |
| `basis.rs` | `BasisSet` | `CubicBSpline` (M5, interior B-splines with implicit zero endpoints) |
| `bspline.rs` | — | Clamped knot vectors, B-spline evaluation, sinc kernel integration |
| `kernel.rs` | — | Builds weighted system matrix K from basis + data |
| `regularise.rs` | `Regulariser` | `BoundaryAnchoredCombined`: D̃₁ + D̃₂ with zero-boundary anchoring; also `SecondDerivative`, `FirstDerivative`, `CombinedDerivative` |
| `solver.rs` | `Solver` | `TikhonovSolver`: SVD-based with L-curve curvature analysis |
| `lambda_select.rs` | `LambdaSelector` | `GcvSelector`, `LCurveSelector`, `BayesianEvidence` |
| `nonneg.rs` | `NonNegativityStrategy` | `IterativeClipping`: re-solves with zero-clamped bins |
| `preprocess.rs` | `Preprocessor` | `ClipNegative`, `OmitNonPositive`, `QRangeSelector` (q-range + SNR cutoff), `LogRebin` |
| `config.rs` | — | `UnfourierConfig`: TOML config from `unfourier.toml` (overrides unset CLI flags) |
| `output.rs` | — | Writes P(r) and back-calculated I(q) fit |

**Key design principles:**
- Trait interfaces designed so future work slots in without touching the pipeline
- Library uses `thiserror` for typed errors; binary uses `anyhow`
- Each λ evaluation is a pure function — designed for `rayon::par_iter()` when needed
- `[constraints] d1_smoothness` in `unfourier.toml` enables the first-derivative boundary term

## Milestones

| M | Feature | Status |
|---|---------|--------|
| M1 | Naive least-squares end-to-end | ✅ |
| M2 | Tikhonov with manual λ | ✅ |
| M3 | Automatic λ (GCV + L-curve) | ✅ |
| M4 | Bayesian IFT with error bars | ✅ |
| M5 | Cubic B-spline basis | ✅ |
| M6 | Full preprocessing pipeline (log-rebin, q-range, SNR, negative handling) | ✅ |
| M7 | Boundary-anchored regulariser + TOML config | ✅ |
| M8 | Smooth zero boundary conditions P(0)=P(D_max)=0 | 🔴 in progress |

## Open Issue: M8 — Smooth boundary conditions

**Problem:** P(r) at r=0 and r=D_max should go smoothly to zero. Earlier M8 work could not achieve this:

- **Without hard zeroing:** The interior bins adjacent to the boundary take non-zero values — P(r) is non-zero at the edges.
- **With hard zeroing** (post-hoc zero-clamp of first/last interior bin): P(r) hits zero at the boundary point but is discontinuous — the interior bins just inside are finite, so the curve steps to zero rather than sloping to zero.

**Current approach after Epic 2:**
1. `Solution` stores solved spline coefficients as `coeffs`, not sampled `P(r)`.
2. `CubicBSpline::output_grid()` creates a dense output grid including `r=0` and `r=D_max`.
3. `CubicBSpline::evaluate_pr()` evaluates the spline expansion for output; no post-solve coefficient clamp is applied.

**Remaining root cause:** The cubic B-spline basis enforces exact zero endpoint values, but optional derivative-zero boundary modes and projected regularisation are still future Epic 3/4 work. Smoothness near the boundaries is currently controlled by the spline basis and boundary-anchored regularisation rather than a full coefficient-mapping model.

**Candidate approaches still to explore:**
- For splines: add `value_zero` / `value_slope_zero` boundary modes through an explicit free-to-full coefficient mapping.
- Project the kernel and regulariser through that mapping so kernel, regulariser, and output evaluation all agree.
- Revisit Bayesian P(r) error propagation once full posterior covariance output is needed on the dense grid.

## Validation Notes

- **Debye chain** (`Dev/gen_debye.py`) is the primary noisy benchmark. Use this for validating regularisation and λ selection.
- **Sphere** (`Dev/gen_sphere.py`) is used only for kernel accuracy tests (analytic P(r) is exact). It is **unsuitable for noisy-data validation** because the proportional noise model diverges near zeros of I(q) — see `saxs_ift_postmortem.md` for the full analysis.
- **Bayesian error bars** (`--method bayes`) produce a 3-column P(r) output. Calibration is checked via `Dev/monte_carlo_coverage.py` — expect ~60–68% coverage at ±1σ; systematic under-coverage reflects regularisation bias not captured by the posterior.
- There is no formal test suite; validation is done via the Python scripts in `Dev/`.
