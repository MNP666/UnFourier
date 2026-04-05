# unFourier — Milestone Plan

Each milestone is a **vertical slice**: a working CLI that reads a `.dat` file and
produces a P(r) output. No milestone is "just infrastructure" — every one delivers
an end-to-end result you can run and inspect.

## Design philosophy

**Inspired by GNOM, not compatible with it.** The goal is a fast, modern
Rust implementation of indirect Fourier transformation for SAXS data. We draw on
the mathematical ideas behind GNOM (Tikhonov regularisation, perceptual criteria)
and BayesApp (evidence maximisation), but ATSAS file-format compatibility and
workflow compatibility are explicit non-goals.

**Interfaces first, optimise later.** Each milestone should define clean trait
boundaries so that components (basis functions, solvers, regularisation strategies,
preprocessors) can be swapped or improved without rewiring the pipeline. Concrete
implementations can be naive at first — the interfaces are what matter for future
iteration.

**Python helpers in `Dev/`.** Use Python scripts freely for generating synthetic
test fixtures, plotting results, and prototyping math. These live in `Dev/` and
are not part of the Rust build. If a calculation is easier to validate in
scipy/numpy first, do it there, then port to Rust.

**Parallelism-ready.** Design solver and grid-search interfaces so that
independent evaluations (e.g., fitting at many λ values, cross-validation folds)
can be trivially dispatched to a thread pool in a future milestone. This means:
pure functions where possible, shared-nothing evaluation of each λ candidate,
and collecting results into a Vec rather than mutating shared state. We do not
add `rayon` or async until the serial version is correct and profiled, but the
code should not fight parallelism when we get there.

**Preprocessing as a composable pipeline.** Data preprocessing steps (rebinning,
negative-value handling, q-range selection, Guinier-based q_min estimation) are
each a separate transform that takes a dataset and returns a dataset. In M1–M4
we only implement the bare minimum (parse + pass through), but the trait/struct
boundary is in place from M1 so that preprocessing steps can be slotted in later
without restructuring.

## Validation strategy

Every milestone validates against two complementary synthetic fixtures.

**Sphere — noiseless only.** A solid sphere of radius R has fully analytic answers:

- I(q) = [3(sin(qR) − qR·cos(qR)) / (qR)³]²
- P(r) = piecewise polynomial on [0, 2R]

This is the sharpest test of kernel correctness and solver numerics. It is used
*without noise* only. With a proportional noise model σ = I/k, the sphere becomes
brittle: I(q) has true zeros where σ → 0 and the weight matrix W = diag(1/σ²)
diverges. This causes the dominant eigenvalues of KᵀWK to blow up, making any
finite λ irrelevant (λ_eff = λh/(μ + λh) → 0 as μ → ∞). The solver then
"nails" the near-zero minima at the expense of the rest of the fit.
See `saxs_ift_postmortem.md` for the full mathematical analysis.

**Debye/Gaussian chain — primary noisy fixture.** A polymer chain with radius of
gyration Rg has:

- I(q) = 2(e^{−x} − 1 + x) / x²,  x = (qRg)²
- P(r) ∝ r² exp(−3r²/4Rg²)   (Gaussian, no zeros)

I(q) decays monotonically with no zeros, so σ = I/k is bounded away from zero
for all q and the weight matrix stays well-conditioned. This is the standard
noisy benchmark from M2 onwards. Scripts for both fixtures live in `Dev/`.

---

## M1 — Naive least-squares, end-to-end ✅

**Goal:** A CLI that reads `q, I(q), σ(q)` from a `.dat` file, solves for P(r)
using unregularised least squares, and writes P(r) to stdout or a file.

**What was built:**

Rust:
- `SaxsData` struct (`src/data.rs`): holds q, I(q), σ(q) vectors; parses 3-column `.dat` files
- `Preprocessor` trait (`src/preprocess.rs`) with `Identity` no-op implementation
- `BasisSet` trait (`src/basis.rs`) with `UniformGrid` (rectangular/histogram) implementation
- Kernel matrix construction (`src/kernel.rs`): K_ij = sin(q_i · r_j) / (q_i · r_j) · Δr
- `Solver` trait (`src/solver.rs`) with SVD-based implementation
- `--rmax` and `--output` CLI flags in `src/main.rs`
- `output.rs` writes r, P(r) to file or stdout

Python (`Dev/`):
- `gen_sphere.py`: synthetic sphere I(q) with optional noise
- `plot_pr.py`: plots P(r) with optional analytic overlay

**Key traits introduced:**

```
trait Preprocessor { fn process(&self, data: SaxsData) -> SaxsData; }
trait BasisSet     { fn design_matrix(&self, q: &[f64]) -> Matrix; }
trait Solver       { fn solve(&self, K: &Matrix, data: &SaxsData) -> Solution; }
```

**Definition of done:** ✅ `unfourier input.dat` produces a two-column P(r) output.

---

## M2 — Tikhonov regularisation with manual λ ✅

**Goal:** Add L₂ regularisation so the output is physically meaningful.

**What was built:**

Rust:
- `Regulariser` trait (`src/regularise.rs`) with `SecondDerivative` implementation
- Second-derivative operator L as the regularisation matrix
- Solve: minimise ‖Kc − I‖² + λ‖Lc‖²  →  (KᵀK + λLᵀL)c = KᵀI
- `--lambda` CLI flag for manual λ supply
- `IterativeClipping` non-negativity strategy (`src/nonneg.rs`): re-solves with zero-clamped bins
- Boundary constraint: P(0) = P(r_max) = 0
- `--rmax` CLI flag

**Key trait introduced:**

```
trait Regulariser {
    fn regularise(&self, KtK: &Matrix, KtI: &Vector, lambda: f64) -> Solution;
}
```

Python (`Dev/`):
- `gen_sphere.py`: updated with noise-level CLI args
- `gen_debye.py`: Debye chain fixture

**Definition of done:** ✅ `unfourier --lambda 0.01 --rmax 180 Dev/debye_k5.dat`
produces a smooth, non-negative P(r) that visually matches the Debye reference.

---

## M3 — Automatic regularisation (L-curve / GCV) ✅

**Goal:** Remove the need to hand-tune λ.

**What was built:**

Rust (`src/lambda_select.rs`):
- `LambdaSelector` trait with `LambdaEvaluation` struct (lambda, residual_norm, solution_norm, gcv, log_evidence)
- `GcvSelector`: minimises generalised cross-validation score (default method)
- `LCurveSelector`: finds corner of maximum curvature in (log ‖residual‖, log ‖solution‖) plane
- Log-spaced λ grid; each candidate evaluated independently into `Vec<LambdaEvaluation>` (parallelism-ready)
- `--method lcurve|gcv|manual|bayes` CLI flag; `--lambda` still works as manual override
- Automatic r_max estimation (π / q_min) with `--rmax` override

Python (`Dev/`):
- `sweep_noise.py`: sweeps Debye data at multiple noise levels, collects auto-selected λ and quality metrics

**Definition of done:** ✅ `unfourier Dev/debye_k5.dat` (no manual λ) produces a
good P(r) across the noise sweep.

---

## M4 — Bayesian IFT ✅

**Goal:** Implement evidence-based regularisation (BayesApp-style).

**What was built:**

Rust (`src/lambda_select.rs`):
- `log_evidence` field added to `LambdaEvaluation` — computed from the Cholesky
  factor already live during grid evaluation (log det for free):
  log P(I|λ) = -½[RSS_w + λ_eff‖Lc‖² + log det(A + λ_eff H) − N_r·log λ_eff]
- `BayesianEvidence` implements `LambdaSelector`: picks the λ with highest log-evidence
- `posterior_sigma(m, lambda_eff)`: solves (A + λH)X = I via Cholesky to get
  Σ = (KᵀWK + λLᵀL)⁻¹, returns sqrt(diag(Σ)) as P(r) error bars
- `--method bayes` flag added to the CLI
- Output gains a third column σ_P(r) when method is bayes

Python (`Dev/`):
- `monte_carlo_coverage.py`: generates N noise realisations of Debye data, runs
  `unfourier --method bayes` on each, and checks coverage of the analytic P(r)
  within ±1σ. Plots coverage fraction vs r and mean P(r) ± mean σ vs truth.

**Validation:** Debye/Gaussian chain data only (not sphere — see `saxs_ift_postmortem.md`).
With N=50, mean coverage ~60–68% at ±1σ. Slight under-coverage is expected: the
posterior Σ = (A + λH)⁻¹ captures noise variance but not regularisation bias.

**Definition of done:** ✅ `unfourier --method bayes Dev/debye_k5.dat` produces
3-column P(r) output with error bars. Monte Carlo coverage script confirms calibration.

---

## M5 — Spline basis functions

**Goal:** Replace the simple rectangular basis with cubic B-splines for a more
compact and accurate representation.

**What you build:**

Rust:
- `CubicBSpline` implementation of the `BasisSet` trait (leverage your Python
  prototype in `Dev/spline_pr_tests.py`)
- Compact support: each spline only contributes to a few columns of K → sparse
  kernel (though we can stay dense for now and optimise later)
- Knot placement: uniform spacing first, with the interface allowing adaptive
  placement in the future
- All existing methods (Tikhonov, GCV, Bayes) work with the new basis — they
  only depend on the `BasisSet` trait

Python (`Dev/`):
- Port/validate spline construction against scipy to ensure the Rust
  implementation matches

**Validation:** Same synthetic cases. Results should be at least as good as M2–M4
with fewer parameters. Convergence as knot count increases.

**Definition of done:** All methods produce good P(r) on synthetic data using the
spline basis with fewer coefficients than the rectangular basis.

---

## M6 — Preprocessing pipeline and real-world data

**Goal:** Handle messy experimental SAXS data robustly, and make the tool
pleasant to use.

**What you build:**

Rust — preprocessing (implementations of the `Preprocessor` trait, composable):
- `LogRebin` / `LinearRebin`: reduce data density in high-q region
- `ClipNegative`: handle negative intensities (set to zero, or to small positive)
- `QRangeSelector`: automatic or manual q_min / q_max selection
  - Guinier-based q_min estimation (linear region in ln I vs q²)
  - Signal-to-noise-based q_max cutoff
- `PreprocessingPipeline`: chains multiple preprocessors in sequence

Rust — robust parsing:
- Handle varied `.dat` formats: header lines (auto-detect by failing to parse
  as float), tab or space delimiters, optional 2-column (missing σ → auto-estimate)
- Helpful error messages with line numbers for malformed input

Rust — output and diagnostics:
- Summary statistics: I(0), R_g, D_max, quality-of-fit metrics
- `--output` flag for file output, `--format` for output style
- `--quiet` / `--verbose` flags

Python (`Dev/`):
- Collect 2–3 real experimental `.dat` files (e.g., BSA, lysozyme from public
  SASBDB entries) as test fixtures

**Validation:** Run on real experimental datasets with known published P(r) results.
Compare R_g and D_max to published values.

**Definition of done:** `unfourier lysozyme.dat` produces physically reasonable
results that agree with published analyses.

---

## Future work (beyond M6)

These are explicitly **not** in the milestone plan, but the interfaces are designed
to accommodate them:

- **Parallelism:** add `rayon` to parallelise the λ grid search in M3/M4 and
  cross-validation in GCV. The `LambdaEvaluation` design is already shared-nothing.
- **Adaptive knot placement:** data-driven knot selection for the spline basis.
- **Multiple regularisation terms:** combine smoothness + non-negativity + other
  priors.
- **Alternative output formats:** XML, JSON, or custom formats for downstream tools.
- **Desmearing:** slit-length correction for slit-collimated instruments.
- **WASM/Python bindings:** expose the core library for use from other languages.
- **GPU acceleration:** for very large datasets or batch processing.

---

## Ordering rationale

| Milestone | Why this order |
|-----------|---------------|
| M1 | Proves the entire pipeline works; establishes trait boundaries |
| M2 | Makes the output physically meaningful — first "useful" result |
| M3 | Removes user guesswork; parallelism-ready λ grid design |
| M4 | Added Bayesian evidence λ selection + posterior error bars |
| M5 | Improves accuracy of all methods at once via a new BasisSet |
| M6 | Polish and preprocessing only after the core math is solid |

Each milestone can be shipped, tested, and validated independently. If you get
stuck on M4 (Bayesian), M2–M3 already give you a working, useful tool.
