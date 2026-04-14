# unFourier

**Work in progress.** The current project version is **0.9**. The codebase is
usable for experimentation, but it is not yet a validated analysis tool.
Version **1.0** will be the first version where at least one regularisation /
selection method, such as GCV, L-curve, Bayesian evidence, or a manual workflow,
is validated well enough to be considered a working recommended path.

A Rust implementation of Indirect Fourier Transformation (IFT) for Small Angle X-ray Scattering (SAXS) data.

---

## Goals

**SAXS IFT in Rust.** unFourier explores recovery of the pair distance distribution function P(r) from a measured scattering curve I(q). This is an ill-posed inverse problem: the kernel is a Fredholm integral equation of the first kind, and regularisation is essential to prevent the solution from fitting noise. The project implements and compares several approaches — Tikhonov regularisation with manual λ, automatic selection via GCV and L-curve, and Bayesian evidence maximisation — all operating on the same clean pipeline.

**Flexible and extensible by design.** The pipeline is built around a small set of traits (`BasisSet`, `Solver`, `Regulariser`, `LambdaSelector`, `Preprocessor`) so that components can be swapped or extended without restructuring the rest of the code. Want to try a new λ selection strategy? Implement `LambdaSelector`. The interfaces are more important than the initial implementations.

**Parallelism-ready.** Each λ evaluation is stateless and self-contained, making the grid search a natural target for `rayon::par_iter()`. The design avoids shared mutable state in hot paths so that parallelism can be added in one place when needed, without refactoring the solver.

**Inspired by GNOM and BayesApp.** The mathematical ideas — Tikhonov regularisation, perceptual smoothness criteria, and Bayesian evidence maximisation — draw on the foundational work behind [GNOM](https://www.embl-hamburg.de/biosaxs/gnom.html) (Svergun, 1992) and [BayesApp](https://bayesapp.org/) (Hansen, 2012). ATSAS file-format compatibility and workflow compatibility are explicit non-goals; unFourier is designed to be clean and comprehensible first.

**A Rust learning project.** This is also a hobby project for gaining hands-on experience with Rust — ownership, traits, error handling, nalgebra, and eventually rayon — using Claude as a coding partner. Expect the code to prioritise clarity and good Rust idioms over raw performance.

---

## Status

Current version: **0.9**.

The milestone features are implemented as code paths, but M3-M8 should still be
treated as work in progress. The next goal is not to add more surface area, but
to validate and harden at least one method into a reliable 1.0 workflow.

| Milestone | Description | Current state |
|-----------|-------------|---------------|
| M1 | Naive least-squares, end-to-end pipeline | Baseline implemented |
| M2 | Tikhonov regularisation with manual λ | Baseline implemented |
| M3 | Automatic λ selection (GCV, L-curve) | Implemented, validation in progress |
| M4 | Bayesian IFT with posterior error bars | Implemented, calibration in progress |
| M5 | Cubic B-spline basis functions | Implemented, behaviour still being validated |
| M6 | Full preprocessing pipeline (rebinning, q-range, SNR trimming) | Implemented, policy still being validated |
| M7 | Real-data validation and GCV robustness (GCV-fallback, NNLS) | Implemented as exploratory validation |
| M8 | Perceptual constraints (boundary enforcement, combined regulariser) | Implemented, tuning and validation in progress |

See [MILESTONES.md](MILESTONES.md) for the full plan and rationale.

---

## Installation

**Requirements:** Rust toolchain (1.75+), Cargo. Python 3.10+ with `numpy`, `matplotlib`, and `scipy` for the helper scripts in `Dev/`.

```bash
git clone https://github.com/MNP666/UnFourier
cd UnFourier
cargo build --release
```

The binary is at `target/release/unfourier`.

---

## Usage

```
unfourier [OPTIONS] <INPUT>

Arguments:
  <INPUT>  3-column .dat file: q (Å⁻¹), I(q), σ(q)

Options:
  --method <METHOD>          gcv | lcurve | bayes | manual  [default: gcv]
  --lambda <LAMBDA>          Regularisation strength (manual mode)
  --rmax <RMAX>              Maximum r in Å  [default: π / q_min]
  --n-basis <N>              Number of free cubic B-spline basis parameters
  --knot-spacing <F>         Derive n_basis as ceil(rmax / spacing)
  --min-basis <N>            Lower clamp for knot-spacing-derived n_basis
  --max-basis <N>            Upper clamp for knot-spacing-derived n_basis
  --lambda-count <N>         Grid size for automatic search  [default: 60]
  --lambda-min <F>           Lower bound of λ search grid
  --lambda-max <F>           Upper bound of λ search grid
  --negative-handling <H>    clip | omit | keep  [default: clip]
  --qmin <F>                 Discard points with q below this value (Å⁻¹)
  --qmax <F>                 Discard points with q above this value (Å⁻¹)
  --snr-cutoff <F>           Trim high-q tail where I/σ < threshold  [default: 0]
  --rebin <N>                Rebin into N log-spaced q bins  [default: 0]
  -o, --output <FILE>        Write P(r) here  [default: stdout]
  --fit-output <FILE>        Write back-calculated I(q) here
  -v, --verbose              Print diagnostics to stderr
  -h, --help                 Print help
  -V, --version              Print version
```

### Examples

```bash
# Automatic λ selection via GCV (default)
unfourier data.dat --rmax 150 --output pr.dat --fit-output fit.dat --verbose

# L-curve selection
unfourier data.dat --rmax 150 --method lcurve --output pr.dat

# Bayesian IFT — produces 3-column output with error bars: r, P(r), σ_P(r)
unfourier data.dat --rmax 150 --method bayes --output pr.dat --verbose

# Choose the number of free B-spline basis parameters
unfourier data.dat --rmax 150 --n-basis 20 --output pr.dat

# Or derive the number of free parameters from Dmax
unfourier data.dat --rmax 150 --knot-spacing 7.5 --min-basis 12 --max-basis 48 --output pr.dat

# Manual λ
unfourier data.dat --rmax 150 --lambda 0.01 --output pr.dat

# Preprocessing: trim q-range, SNR cutoff, and rebin
unfourier data.dat --rmax 55 --qmin 0.01 --qmax 0.30 --snr-cutoff 3 --rebin 200

# Wider λ search range
unfourier data.dat --rmax 150 --lambda-min 1e-8 --lambda-max 1e5
```

### TOML configuration

For reproducible runs, create `unfourier.toml` in the working directory. CLI flags take precedence over TOML values.

```toml
[regularisation]
method = "gcv"          # gcv | lcurve | bayes | manual
lambda_min = 1e-6       # lower bound of λ search grid
lambda_max = 1e3        # upper bound of λ search grid

[preprocessing]
qmin = 0.01             # Å⁻¹ — discard points below this q
qmax = 0.35             # Å⁻¹ — discard points above this q
negative_handling = "clip"  # clip | omit | keep for non-positive I(q)

[basis]
n_basis = 20            # number of free cubic B-spline basis parameters
# Alternative when n_basis is absent:
knot_spacing = 7.5      # derive n_basis = ceil(Dmax / knot_spacing)
min_basis = 12          # lower clamp for derived n_basis
max_basis = 48          # upper clamp for derived n_basis

[constraints]
spline_boundary = "value_zero"  # value_zero | value_slope_zero

# First- and second-derivative penalties, applied in full spline coefficient
# space and projected onto the free coefficients.
# d1_smoothness: absent/0 = default 0.1, -1 = disabled, >0 = explicit weight
# d2_smoothness: absent = 1.0, >=0 = explicit weight
d1_smoothness = 0.0
d2_smoothness = 1.0
```

Resolution rule: explicit `n_basis` wins over `knot_spacing`. If `n_basis` is
absent and `knot_spacing` is set, unFourier uses `ceil(Dmax / knot_spacing)`
clamped to `[min_basis, max_basis]`. If neither is set, the default is
`n_basis = 20`.

### Input format

Whitespace-delimited, three columns; lines starting with `#` are ignored:

```
# q(1/A)         I(q)          sigma(q)
1.00000000e-02   9.71e-01      9.71e-04
1.25000000e-02   9.55e-01      9.55e-04
...
```

### Output format

Two-column `r  P(r)`, with a `#` header line. With `--method bayes`, three columns: `r  P(r)  σ_P(r)`. If `--fit-output` is given, a four-column file `q  I_obs  I_calc  sigma` is also written.

---

## Project structure

```
UnFourier/
├── src/
│   ├── main.rs           # CLI entry point and pipeline wiring
│   ├── lib.rs            # Module exports
│   ├── data.rs           # .dat parser, SaxsData struct
│   ├── basis.rs          # BasisSet trait and CubicBSpline basis
│   ├── bspline.rs        # Cox–de Boor B-spline evaluation and quadrature (M5)
│   ├── kernel.rs         # Weighted system matrix and back-calculation
│   ├── solver.rs         # Solver trait, LeastSquaresSvd, TikhonovSolver
│   ├── regularise.rs     # Regulariser trait + derivative penalties and
│   │                     #   ProjectedSplineRegulariser
│   ├── nonneg.rs         # NonNegativityStrategy trait, ProjectedGradient NNLS
│   ├── lambda_select.rs  # LambdaSelector trait, GCV, L-curve, BayesianEvidence,
│   │                     #   GridMatrices, evaluate_lambda_grid
│   ├── preprocess.rs     # Preprocessor trait, ClipNegative, OmitNonPositive,
│   │                     #   QRangeSelector, LogRebin, PreprocessingPipeline (M6)
│   ├── config.rs         # unfourier.toml parser (TOML config, M7)
│   └── output.rs         # P(r) and fit file writers
├── Dev/
│   ├── gen_sphere.py          # Generate synthetic sphere SAXS data
│   ├── gen_debye.py           # Generate synthetic Debye/Gaussian-chain data
│   ├── plot_pr.py             # Plot P(r) output vs analytic reference
│   ├── sweep_noise.py         # Spline noise, n_basis, and smoothness sweeps
│   ├── sweep_smoothness.py    # Epic 4 validation: d1/d2 smoothness sweep
│   ├── sweep_knot_density.py  # M8/Epic 5 validation: n_basis and knot spacing
│   ├── monte_carlo_coverage.py# Bayesian spline error-bar coverage
│   ├── parse_gnom.py          # Parse GNOM .out files for reference P(r)
│   ├── validate_spline.py     # Spline-only synthetic regression checks
│   └── validate_real_data.py  # M7/M8 validation against SASDME2, SASDF42, SASDYU3
├── data/
│   └── dat_ref/               # Reference SAXS datasets (SASDME2, SASDF42, SASDYU3)
├── pipeline.md           # Pipeline description with mathematics
├── MILESTONES.md         # Detailed milestone plan with rationale
├── saxs_ift_postmortem.md# Mathematical analysis of sphere-data regularisation washout
├── Cargo.toml
└── LICENSE               # GPLv3
```

---

## Architecture

The pipeline in `main.rs` is: **parse → preprocess → build basis → build system → solve coefficients → evaluate spline P(r) → output**.

Each stage is abstracted behind a trait, making the system extensible without modifying the pipeline:

| Module | Trait | Implementations |
|--------|-------|-----------------|
| `data.rs` | — | `SaxsData`: holds q, I(q), σ(q); parses 3-column `.dat` files |
| `basis.rs` | `BasisSet` | `CubicBSpline` (clamped B-splines) |
| `kernel.rs` | — | `build_weighted_system`, `back_calculate` |
| `regularise.rs` | `Regulariser` | `SecondDerivative`, `FirstDerivative`, `CombinedDerivative`, `ProjectedSplineRegulariser` |
| `solver.rs` | `Solver` | `LeastSquaresSvd` (M1), `TikhonovSolver` (M2+) |
| `lambda_select.rs` | `LambdaSelector` | `GcvSelector`, `LCurveSelector`, `BayesianEvidence` |
| `nonneg.rs` | `NonNegativityStrategy` | `ProjectedGradient` (NNLS), `NoConstraint` |
| `preprocess.rs` | `Preprocessor` | `ClipNegative`, `OmitNonPositive`, `QRangeSelector`, `LogRebin` |
| `config.rs` | — | `UnfourierConfig`: TOML-based configuration |

See [pipeline.md](pipeline.md) for a detailed mathematical description of each stage.

---

## Validation strategy

**Debye/Gaussian chain (primary noisy benchmark):** The Debye form factor decays monotonically with no zeros, so σ stays well-conditioned at all q. Generate synthetic data with:

```bash
python Dev/gen_debye.py --rg 30 --k 5 --output Dev/debye_k5.dat \
    --pr-reference Dev/debye_pr_ref.dat
```

**Sphere (noiseless only):** The solid sphere has a fully analytic I(q) and P(r), making it a sharp test of kernel correctness. It is used without noise because the proportional noise model σ = I/k diverges at the sphere's exact intensity zeros, making any regulariser degenerate. See [saxs_ift_postmortem.md](saxs_ift_postmortem.md) for the analysis.

**Real SAXS data (M7/M8):** Three datasets from the SASBDB are used for end-to-end validation:

| Dataset | Description | Key challenge |
|---------|-------------|---------------|
| SASDME2 | Well-behaved protein | Baseline pass |
| SASDF42 | Moderate signal-to-noise | Open right boundary, λ sensitivity |
| SASDYU3 | Dense dataset (1696 points) | Requires log-rebinning |

Run the full validation suite with:

```bash
python Dev/validate_spline.py
python Dev/sweep_noise.py
python Dev/monte_carlo_coverage.py --n 200 --k 5 --n-basis 20
python Dev/validate_real_data.py
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| [clap](https://docs.rs/clap) | CLI argument parsing |
| [nalgebra](https://nalgebra.org) | Linear algebra (matrices, SVD, Cholesky) |
| [anyhow](https://docs.rs/anyhow) | Ergonomic error handling in the binary |
| [thiserror](https://docs.rs/thiserror) | Typed errors in the library |
| [serde](https://docs.rs/serde) + [toml](https://docs.rs/toml) | TOML configuration parsing |

---

## Acknowledgements

- **GNOM** (D. I. Svergun, 1992) and the ATSAS package for establishing the mathematical framework for IFT of SAXS data.
- **BayesApp** (S. Hansen, 2012) for the Bayesian evidence-maximisation approach that inspired M4.
- **Claude** (Anthropic) as a coding and mathematics partner throughout development.
- **Codex / ChatGPT** (OpenAI) as a coding, documentation, and validation partner.

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
