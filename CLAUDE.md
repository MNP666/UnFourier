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

# Or derive the basis count from Dmax with clamp bounds
unfourier data.dat --knot-spacing 7.5 --min-basis 12 --max-basis 48 --rmax 150 -o pr.dat

# Preprocessing options: q-range, SNR cutoff, log-rebinning
unfourier data.dat --qmin 0.01 --qmax 0.35 --snr-cutoff 1.0 --rebin 200 -o pr.dat

# TOML config file (unfourier.toml in CWD) overrides unset CLI args
# Sections: [regularisation], [preprocessing], [basis], [constraints]

# Generate synthetic test fixtures (requires numpy, matplotlib, scipy)
python Dev/gen_debye.py        # Noisy Debye chain data (primary benchmark)
python Dev/gen_sphere.py       # Sphere data (kernel accuracy, not noisy tests)

# Plot P(r) vs reference
python Dev/plot_pr.py

# Spline validation: synthetic regressions, noise levels, n_basis, and smoothness
python Dev/validate_spline.py
python Dev/sweep_noise.py

# Monte Carlo coverage check for Bayesian spline error bars
python Dev/monte_carlo_coverage.py --n 200 --k 5 --n-basis 20

# Epic 4 validation: sweep D1 smoothness on Debye or a supplied dataset
python Dev/sweep_smoothness.py

# M8/Epic 5 validation: sweep n_basis and knot_spacing
python Dev/sweep_knot_density.py
```

## Architecture

The pipeline in `main.rs` is: **parse → preprocess → build basis → build kernel → solve coefficients → evaluate spline P(r) → output**.

Each stage is abstracted behind a trait, making the system extensible without modifying the pipeline:

| Module | Trait | Current Implementation |
|--------|-------|------------------------|
| `data.rs` | — | `SaxsData`: holds q, I(q), σ(q); parses 3-column `.dat` files |
| `basis.rs` | `BasisSet` | `CubicBSpline` with `value_zero` / `value_slope_zero` boundary modes |
| `bspline.rs` | — | Clamped knot vectors, B-spline evaluation, sinc kernel integration |
| `kernel.rs` | — | Builds weighted system matrix K from basis + data |
| `regularise.rs` | `Regulariser` | `ProjectedSplineRegulariser`: combined D₁/D₂ penalty projected through the spline boundary map; also `SecondDerivative`, `FirstDerivative`, `CombinedDerivative` |
| `solver.rs` | `Solver` | `TikhonovSolver`: SVD-based with L-curve curvature analysis |
| `lambda_select.rs` | `LambdaSelector` | `GcvSelector`, `LCurveSelector`, `BayesianEvidence` |
| `nonneg.rs` | `NonNegativityStrategy` | `ProjectedGradient` (NNLS), `NoConstraint`; `IterativeClipping` remains as a legacy strategy |
| `preprocess.rs` | `Preprocessor` | `ClipNegative`, `OmitNonPositive`, `QRangeSelector` (q-range + SNR cutoff), `LogRebin` |
| `config.rs` | — | `UnfourierConfig`: TOML config from `unfourier.toml` (overrides unset CLI flags) |
| `output.rs` | — | Writes P(r) and back-calculated I(q) fit |

**Key design principles:**
- Trait interfaces designed so future work slots in without touching the pipeline
- Library uses `thiserror` for typed errors; binary uses `anyhow`
- Each λ evaluation is a pure function — designed for `rayon::par_iter()` when needed
- `[constraints] spline_boundary` selects `value_zero` (default) or `value_slope_zero`
- `[constraints] d1_smoothness` controls the neighboring-coefficient penalty: absent/0 = default 0.1, -1 = disabled, >0 = explicit
- `[constraints] d2_smoothness` controls curvature regularisation: absent = 1.0, >=0 = explicit

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
| M8 | Smooth zero boundary conditions P(0)=P(D_max)=0 | ✅ |

## M8 / 0.9 Boundary Status

P(r) at r=0 and r=D_max should go smoothly to zero. Earlier M8 work could not achieve this:

- **Without hard zeroing:** The interior bins adjacent to the boundary take non-zero values — P(r) is non-zero at the edges.
- **With hard zeroing** (post-hoc zero-clamp of first/last interior bin): P(r) hits zero at the boundary point but is discontinuous — the interior bins just inside are finite, so the curve steps to zero rather than sloping to zero.

**Current 0.9 approach:**
1. `Solution` stores solved spline coefficients as `coeffs`, not sampled `P(r)`.
2. `CubicBSpline::output_grid()` creates a dense output grid including `r=0` and `r=D_max`.
3. `CubicBSpline::evaluate_pr()` evaluates the spline expansion for output; no post-solve coefficient clamp is applied.
4. `CubicBSpline` uses an explicit free-to-full coefficient map for `value_zero` and `value_slope_zero`.
5. `ProjectedSplineRegulariser` applies D1/D2 smoothness in full coefficient space and projects it through that same map.

**Validation posture:** `value_zero` is the default boundary mode. `value_slope_zero`, `d1_smoothness`, `d2_smoothness`, `n_basis`, and `knot_spacing` remain explicit knobs for exploratory validation on Debye and real datasets.

**Open exploration:**
- Sweep `d1_smoothness` and `d2_smoothness` on Debye and real datasets.
- Decide whether `value_slope_zero` should remain optional or become a recommended setting for specific data.
- Revisit Bayesian P(r) error propagation once full posterior covariance output is needed on the dense grid.

## Validation Notes

- **Debye chain** (`Dev/gen_debye.py`) is the primary noisy benchmark. Use this for validating regularisation and λ selection.
- **Sphere** (`Dev/gen_sphere.py`) is used only for kernel accuracy tests (analytic P(r) is exact). It is **unsuitable for noisy-data validation** because the proportional noise model diverges near zeros of I(q) — see `saxs_ift_postmortem.md` for the full analysis.
- **Bayesian error bars** (`--method bayes`) produce a 3-column P(r) output. Calibration is checked on the emitted spline grid via `Dev/monte_carlo_coverage.py --n-basis 20` — expect ~60–68% coverage at ±1σ; systematic under-coverage reflects regularisation bias not captured by the posterior.
- There is no formal test suite; validation is done via the Python scripts in `Dev/`.
