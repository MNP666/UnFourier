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

Every milestone includes validation against a **synthetic test case** with a known
analytical answer. The simplest is a solid sphere of radius R, which has:

- Analytic I(q) = [3(sin(qR) - qR·cos(qR)) / (qR)³]²
- Analytic P(r) = a piecewise polynomial on [0, 2R]

Generate synthetic I(q) with optional Gaussian noise using a Python script in
`Dev/` (easier to iterate on noise models and plot results). This costs almost
nothing to build and catches numerical bugs immediately — before they compound
across milestones.

---

## M1 — Naive least-squares, end-to-end

**Goal:** A CLI that reads `q, I(q), σ(q)` from a `.dat` file, solves for P(r)
using unregularised least squares, and writes P(r) to stdout or a file.

**What you build:**

Rust:
- `SaxsData` struct: holds q, I(q), σ(q) vectors
- `.dat` parser (3-column, whitespace-delimited, skip `#` comments) — simple for
  now, but behind a `parse()` function that can grow
- `Preprocessor` trait with a no-op `Identity` implementation — placeholder
  interface for future rebinning, clipping, q-range selection
- `BasisSet` trait with a `Grid` (rectangular/histogram) implementation
- Kernel matrix construction: K_ij = sin(q_i · r_j) / (q_i · r_j) · Δr, behind
  a function that takes a `BasisSet` and a q-vector
- `Solver` trait with an unregularised least-squares implementation (SVD)
- r-grid setup (hard-coded r_max and number of points for now)
- Write r, P(r) to stdout

Python (`Dev/`):
- `gen_sphere.py`: generates synthetic I(q) for a solid sphere with optional
  Gaussian noise, writes a 3-column `.dat` file
- `plot_pr.py`: reads two-column P(r) output and plots it (optionally overlay
  the analytic P(r) for comparison)

**Key traits introduced:**

```
trait Preprocessor { fn process(&self, data: SaxsData) -> SaxsData; }
trait BasisSet     { fn design_matrix(&self, q: &[f64]) -> Matrix; }
trait Solver       { fn solve(&self, K: &Matrix, data: &SaxsData) -> Solution; }
```

These are intentionally simple. They will gain associated types and configuration
as needed, but the key is that the pipeline is: parse → preprocess → build basis
→ build kernel → solve → output.

**Key crate dependencies:** `clap` (CLI), `nalgebra` (linear algebra)

**Validation:** Run on sphere data with no noise. The output P(r) will be noisy
and possibly oscillatory (no regularisation), but should broadly peak near the
right r values. This confirms the kernel, parser, and linear algebra are wired
together. Use `plot_pr.py` to visualise.

**Definition of done:** `unfourier input.dat` produces a two-column output that,
when plotted, vaguely resembles a P(r).

---

## M2 — Tikhonov regularisation with manual λ

**Goal:** Add L₂ regularisation so the output is physically meaningful.

**What you build:**

Rust:
- `Regulariser` trait with a `Tikhonov` implementation
- Second-derivative operator L as the regularisation matrix
- Solve: minimise ‖Kc − I‖² + λ‖Lc‖²  →  (KᵀK + λLᵀL)c = KᵀI
- `--lambda` CLI flag (user supplies λ manually)
- Non-negativity constraint on P(r) (iterative clipping or NNLS)
- Boundary constraint: P(0) = P(r_max) = 0
- `--rmax` CLI flag to override the default

**Key trait introduced:**

```
trait Regulariser {
    fn regularise(&self, KtK: &Matrix, KtI: &Vector, lambda: f64) -> Solution;
}
```

The solver now takes a `Regulariser` and a λ. This interface is designed so that
each λ evaluation is self-contained — no shared mutable state — making it trivial
to parallelise the λ grid search in M3.

Python (`Dev/`):
- Update `gen_sphere.py` to support multiple noise levels via CLI args
- `compare_pr.py`: overlay computed P(r) against analytic, compute χ² and
  integrated squared error

**Validation:** Run on sphere data with moderate noise. With a well-chosen λ,
P(r) should closely match the analytic pair-distance distribution. Compare
visually and by metrics using `compare_pr.py`.

**Definition of done:** `unfourier --lambda 0.01 --rmax 100 sphere.dat` produces
a smooth, non-negative P(r) that matches the known sphere result.

---

## M3 — Automatic regularisation (L-curve / GCV)

**Goal:** Remove the need to hand-tune λ.

**What you build:**

Rust:
- `LambdaSelector` trait:
  ```
  trait LambdaSelector {
      fn select(&self, candidates: &[LambdaEvaluation]) -> f64;
  }
  ```
  where `LambdaEvaluation { lambda, residual_norm, solution_norm, ... }` is a
  struct holding everything computed for one λ candidate.
- `LCurve` implementation: find corner of maximum curvature in the
  (log ‖residual‖, log ‖solution‖) plane
- `GCV` implementation: generalised cross-validation criterion
- λ grid generation (log-spaced) — each candidate is evaluated independently,
  results collected into a `Vec<LambdaEvaluation>`. This is the natural
  parallelisation point for a future `rayon::par_iter()`.
- `--auto` flag (default) with `--method lcurve|gcv`
- `--lambda` still works as a manual override
- Automatic r_max estimation (π / q_min as starting guess, with `--rmax` override)

Python (`Dev/`):
- `sweep_noise.py`: run unFourier at multiple noise levels, collect and plot
  the selected λ and resulting P(r) quality metrics
- Add a second fixture: hollow sphere or core-shell particle

**Validation:** Run on sphere data at multiple noise levels. The automatic λ
should produce results comparable to the best hand-tuned λ. Test on the second
synthetic case to check generalisation.

**Definition of done:** `unfourier sphere.dat` (no manual λ) produces a good P(r)
across a range of noise levels.

---

## M4 — Bayesian IFT

**Goal:** Implement evidence-based regularisation (BayesApp-style).

**What you build:**

Rust:
- `BayesianEvidence` as a `LambdaSelector`:
  log P(I|λ) = -½[χ² + λ‖Lc‖² + log det(KᵀK + λLᵀL) − N·log λ + ...]
- Optimise λ by maximising evidence over the same candidate grid (or via
  iterative refinement)
- Posterior covariance: (KᵀK + λLᵀL)⁻¹ → uncertainty estimates on P(r)
- Error bars in output (third column: σ_P(r))
- `--method bayes` flag

Python (`Dev/`):
- `monte_carlo_coverage.py`: generate many noise realisations, run Bayesian IFT
  on each, check that the true P(r) falls within the error bars ~68% of the time

**Validation:** Same synthetic cases as M3. The Bayesian λ should be comparable to
L-curve/GCV. Error bars should be statistically calibrated (Monte Carlo check).

**Definition of done:** `unfourier --method bayes sphere.dat` produces P(r) with
error bars. The error bars are calibrated on synthetic data.

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
| M4 | Adds the second major method; reuses the same interfaces |
| M5 | Improves accuracy of all methods at once via a new BasisSet |
| M6 | Polish and preprocessing only after the core math is solid |

Each milestone can be shipped, tested, and validated independently. If you get
stuck on M4 (Bayesian), M2–M3 already give you a working, useful tool.
