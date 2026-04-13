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

The pipeline in `main.rs` is: **parse → preprocess → r-grid → build kernel → solve → boundary projection → output**.

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

**Problem:** P(r) at r=0 and r=D_max should go smoothly to zero. The current approach cannot achieve this:

- **Without hard zeroing:** The interior bins adjacent to the boundary take non-zero values — P(r) is non-zero at the edges.
- **With hard zeroing** (post-hoc zero-clamp of first/last interior bin): P(r) hits zero at the boundary point but is discontinuous — the interior bins just inside are finite, so the curve steps to zero rather than sloping to zero.

**Current approach (not fully solving the problem):**
1. `BoundaryAnchoredCombined` regulariser: penalises the slope of c[0] from/to the implicit zero boundary (c[-1]=0, c[n]=0), intended to discourage non-zero boundary-adjacent values via smooth regularisation pressure.
2. Post-hoc insertion of explicit (r=0, P=0) and (r=D_max, P=0) rows in the output.
3. Hard zero-clamp of the first and last interior coefficient after solve (`p_r[1] = 0.0`, `p_r[last-1] = 0.0` in `main.rs:549–550`).

**Root cause:** The cubic B-spline basis can enforce exact zero endpoint values, but the current output path still mutates solved values post-hoc. Zeroing interior coefficients after solving creates a visible discontinuity one sample inward from the boundary and makes the written P(r) inconsistent with the coefficients used for I(q).

**Candidate approaches still to explore:**
- For splines: verify that dropping the endpoint B-splines (as currently done in `CubicBSpline`) truly forces P(0)=P(D_max)=0 exactly — and whether optional derivative constraints should be exposed.
- Add explicit boundary constraint rows to the design matrix (augmented least-squares) rather than a post-hoc clamp.
- Evaluate the spline expansion for output instead of writing raw coefficients or post-hoc edited values.

## Validation Notes

- **Debye chain** (`Dev/gen_debye.py`) is the primary noisy benchmark. Use this for validating regularisation and λ selection.
- **Sphere** (`Dev/gen_sphere.py`) is used only for kernel accuracy tests (analytic P(r) is exact). It is **unsuitable for noisy-data validation** because the proportional noise model diverges near zeros of I(q) — see `saxs_ift_postmortem.md` for the full analysis.
- **Bayesian error bars** (`--method bayes`) produce a 3-column P(r) output. Calibration is checked via `Dev/monte_carlo_coverage.py` — expect ~60–68% coverage at ±1σ; systematic under-coverage reflects regularisation bias not captured by the posterior.
- There is no formal test suite; validation is done via the Python scripts in `Dev/`.
