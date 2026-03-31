# unFourier

A Rust implementation of Indirect Fourier Transformation (IFT) for Small Angle X-ray Scattering (SAXS) data.

---

## Goals

**SAXS IFT in Rust.** unFourier recovers the pair distance distribution function P(r) from a measured scattering curve I(q). This is an ill-posed inverse problem: the kernel is a Fredholm integral equation of the first kind, and regularisation is essential to prevent the solution from fitting noise. The project implements and compares several approaches — Tikhonov regularisation with manual λ, automatic selection via GCV and L-curve, and Bayesian evidence maximisation — all operating on the same clean pipeline.

**Flexible and extensible by design.** The pipeline is built around a small set of traits (`BasisSet`, `Solver`, `Regulariser`, `LambdaSelector`, `Preprocessor`) so that components can be swapped or extended without restructuring the rest of the code. Want to replace rectangular bins with B-splines? Swap the `BasisSet`. Want to try a new λ selection strategy? Implement `LambdaSelector`. The interfaces are more important than the initial implementations.

**Parallelism-ready.** Each λ evaluation is stateless and self-contained, making the grid search a natural target for `rayon::par_iter()`. The design avoids shared mutable state in hot paths so that parallelism can be added in one place when needed, without refactoring the solver.

**Inspired by GNOM and BayesApp.** The mathematical ideas — Tikhonov regularisation, perceptual smoothness criteria, and Bayesian evidence maximisation — draw on the foundational work behind [GNOM](https://www.embl-hamburg.de/biosaxs/gnom.html) (Svergun, 1992) and [BayesApp](https://bayesapp.org/) (Hansen, 2012). ATSAS file-format compatibility and workflow compatibility are explicit non-goals; unFourier is designed to be clean and comprehensible first.

**A Rust learning project.** This is also a hobby project for gaining hands-on experience with Rust — ownership, traits, error handling, nalgebra, and eventually rayon — using Claude as a coding partner. Expect the code to prioritise clarity and good Rust idioms over raw performance.

---

## Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Naive least-squares, end-to-end pipeline | ✅ Done |
| M2 | Tikhonov regularisation with manual λ | ✅ Done |
| M3 | Automatic λ selection (GCV and L-curve) | ✅ Done |
| M4 | Bayesian IFT with posterior error bars | 🔲 Planned |
| M5 | Cubic B-spline basis functions | 🔲 Planned |
| M6 | Preprocessing pipeline and real-world data | 🔲 Planned |

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
  --method <METHOD>   gcv | lcurve | manual  [default: gcv]
  --lambda <LAMBDA>   Regularisation strength (manual mode)
  --rmax <RMAX>       Maximum r in Å  [default: π / q_min]
  --npoints <N>       r grid points  [default: 100]
  --lambda-count <N>  Grid size for automatic search  [default: 60]
  --lambda-min <F>    Lower bound of λ search grid  [default: 1e-6]
  --lambda-max <F>    Upper bound of λ search grid  [default: 1e3]
  -o, --output <FILE>     Write P(r) here  [default: stdout]
  --fit-output <FILE>     Write back-calculated I(q) here
  -v, --verbose           Print diagnostics to stderr
  -h, --help              Print help
  -V, --version           Print version
```

### Examples

```bash
# Automatic λ selection via GCV (default)
unfourier data.dat --rmax 150 --output pr.dat --fit-output fit.dat --verbose

# L-curve selection
unfourier data.dat --rmax 150 --method lcurve --output pr.dat

# Manual λ (M2-style)
unfourier data.dat --rmax 150 --lambda 0.01 --output pr.dat

# Wider λ search range
unfourier data.dat --rmax 150 --lambda-min 1e-8 --lambda-max 1e5
```

### Input format

Whitespace-delimited, three columns, lines starting with `#` are ignored:

```
# q(1/A)         I(q)          sigma(q)
1.00000000e-02   9.71e-01      9.71e-04
1.25000000e-02   9.55e-01      9.55e-04
...
```

### Output format

Two-column `r  P(r)`, with a `#` header line. If `--fit-output` is given, a four-column file `q  I_obs  I_calc  sigma` is also written.

---

## Project structure

```
UnFourier/
├── src/
│   ├── main.rs           # CLI entry point
│   ├── lib.rs            # Module exports
│   ├── data.rs           # .dat parser, SaxsData struct
│   ├── basis.rs          # BasisSet trait + UniformGrid implementation
│   ├── kernel.rs         # Weighted kernel matrix construction
│   ├── solver.rs         # Solver trait, LeastSquaresSvd, TikhonovSolver
│   ├── regularise.rs     # Regulariser trait + SecondDerivative
│   ├── nonneg.rs         # NonNegativityStrategy trait + IterativeClipping
│   ├── lambda_select.rs  # LambdaSelector trait, GCV, L-curve, grid evaluation
│   ├── preprocess.rs     # Preprocessor trait + Identity (pipeline stub)
│   └── output.rs         # P(r) and fit file writers
├── Dev/
│   ├── gen_sphere.py     # Generate synthetic sphere SAXS data
│   ├── gen_debye.py      # Generate synthetic Debye/Gaussian-chain data
│   ├── plot_pr.py        # Plot P(r) output vs analytic reference
│   └── sweep_noise.py    # M3 validation: sweep noise levels, compare GCV / L-curve
├── MILESTONES.md         # Detailed milestone plan with rationale
├── saxs_ift_postmortem.md # Mathematical analysis of regularisation washout on sphere data
├── Cargo.toml
└── LICENSE               # GPLv3
```

---

## Validation strategy

Every milestone validates against two synthetic fixtures. See [MILESTONES.md](MILESTONES.md) for the full rationale and [saxs_ift_postmortem.md](saxs_ift_postmortem.md) for the mathematics behind why the sphere is unsuitable as a noisy benchmark.

**Sphere (noiseless only):** The solid sphere has a fully analytic I(q) and P(r), making it a sharp test of kernel correctness. It is used without noise because the proportional noise model σ = I/k diverges at the sphere's exact intensity zeros, making any regulariser degenerate.

**Debye/Gaussian chain (noisy):** The Debye form factor decays monotonically with no zeros, so σ stays well-conditioned at all q. This is the primary benchmark for noisy data from M2 onwards. Generate test data with:

```bash
python Dev/gen_debye.py --rg 30 --k 5 --output Dev/debye_k5.dat \
    --pr-reference Dev/debye_pr_ref.dat
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| [clap](https://docs.rs/clap) | CLI argument parsing |
| [nalgebra](https://nalgebra.org) | Linear algebra (matrices, SVD, Cholesky) |
| [anyhow](https://docs.rs/anyhow) | Ergonomic error handling in the binary |
| [thiserror](https://docs.rs/thiserror) | Typed errors in the library |

---

## Acknowledgements

- **GNOM** (D. I. Svergun, 1992) and the ATSAS package for establishing the mathematical framework for IFT of SAXS data.
- **BayesApp** (S. Hansen, 2012) for the Bayesian evidence-maximisation approach that inspired M4.
- **Claude** (Anthropic) as a coding and mathematics partner throughout development.

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
