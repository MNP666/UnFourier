# SASDYU3 Validation Failure — Root Cause and Proposed Fixes

## Summary of symptoms

| Method | Basis | ISE vs GNOM | Non-zero bins / 100 |
|--------|-------|-------------|---------------------|
| GCV    | rect  | 12.14 ✗ | 28 |
| L-curve | rect | 0.010 ✓ | 100 |
| Bayes  | rect  | 0.049 ✓ | 98 |
| GCV    | spline | 2.28 ✗ | — |

Rg is correct for every method (within 0.3% of GNOM 21.83 Å).  
The shape is only wrong under GCV — and it is severely wrong: 28 isolated
non-zero islands separated by long runs of exact zeros, rather than the
smooth unimodal curve GNOM recovers.

---

## Root cause 1 — GCV landscape is nearly flat

With **N = 1696 data points** and **M = 100 basis bins** the system is
highly overdetermined (N/M ≈ 17).  In this regime the GCV score barely
changes across the λ grid:

```
GCV min = 1.1386    GCV max = 1.2119    total variation = 6.4%
```

The minimum sits at **λ = 8.2 × 10⁻⁶** (λ_eff ≈ 2100), which is the
smallest λ that dips marginally below the plateau.  The true, physically
meaningful minimum is far to the right of where GCV points.

**Why does N/M ratio hurt GCV?**  GCV estimates the leave-one-out prediction
error.  When the system is heavily overdetermined each data point is already
well-predicted without regularisation, so adding more (increasing λ) barely
changes the score.  The landscape is too flat to locate a reliable minimum.

---

## Root cause 2 — iterative clipping amplifies the under-regularisation

At the selected λ the unconstrained Tikhonov solution has large-amplitude
oscillations (alternating positive/negative lobes).  The iterative
non-negativity clipping zeros out every negative bin and resolves, but when
the oscillations are large the cascade destroys most of the solution, leaving
only 28 isolated non-zero bins.  The resulting P(r) is an unphysical "island"
pattern.

Rg survives intact because the surviving islands happen to be centred near
r ≈ 22 Å, so the second-moment integral is approximately preserved even
though the shape is wrong.

---

## Auto-rebin investigation (ruled out as a primary fix)

The hypothesis was: log-rebinning reduces N, lowers N/M, steepens the GCV
landscape, and produces a more stable result.  A stability check comparing
the full-data and rebinned P(r) could then detect instability and favour the
rebinned solution.

**What we tried:**

| Run | N pts | ISE vs GNOM | Notes |
|-----|-------|-------------|-------|
| GCV, full data | 1696 | 12.14 | Spiky, 28 non-zero bins |
| GCV, rebin=100 | ~100 | 16.14 | Worse — N/M ≈ 1, underdetermined |
| GCV, rebin=200 | ~196 | 13.79 | N/M ≈ 2, still unreliable |

The stability diagnostic itself works correctly: full-vs-rebin ISE = 2.0 ≫
threshold, so the switch fires.  But the rebinned GCV result (13.79) is
*also* wrong — it is not a reliable fallback.

**Why rebinning alone does not fix GCV:**  With N=196 and M=100, the system
is barely overdetermined (N/M ≈ 2).  GCV is now unreliable in the opposite
direction: too few data points to distinguish between solutions.  The sweet
spot (N/M ≈ 5–10) would require very small M, reducing P(r) resolution.
There is no rebinning factor that simultaneously gives stable GCV **and**
adequate resolution for SASDYU3.

**Conclusion:** log-rebin is not preferred over the algorithmic fixes below.
Rebinning remains a valid user-configurable option for runtime reduction on
large datasets, but it should not be presented as a correctness fix.

---

## Proposed fixes

### Fix A — Auto-method selection based on N/M ratio
*(λ selector, no new maths)*

Detect when N/M > threshold (e.g. 10) and automatically use L-curve instead
of GCV, or emit a prominent warning.  L-curve selects λ = 7.0 × 10²
(λ_eff ≈ 9 × 10¹⁰) giving ISE = 0.010 with all 100 bins filled.

**Implementation:** 5 lines in `main.rs` before calling `run_solve`.

---

### Fix B — Flat-landscape detector in GCV (zero extra cost)
*(λ selector, no new maths)*

After evaluating the GCV grid, compute the score variation.  If it is less
than a threshold (e.g. 10% of the minimum), fall back to L-curve on the
**same pre-computed** evaluations — zero extra kernel solves.

```rust
// In GcvSelector::select (lambda_select.rs)
let gcv_vals: Vec<f64> = evaluations.iter().map(|e| e.gcv).collect();
let gcv_min = gcv_vals.iter().cloned().fold(f64::INFINITY, f64::min);
let gcv_max = gcv_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
if (gcv_max - gcv_min) / gcv_min < 0.10 {
    eprintln!("warning: GCV landscape flat ({:.1}% variation); \
               falling back to L-curve", 100.0*(gcv_max-gcv_min)/gcv_min);
    return LCurveSelector.select(evaluations);
}
```

This is transparent to users whose data has a well-defined GCV minimum.
Fixes SASDYU3 (ISE 12.14 → 0.010) with no regression risk.

---

### Fix C — First-derivative (total-variation) regularisation term
*(regulariser, medium effort)*

The current L is a second-difference matrix (curvature penalty).  It allows
slowly-varying oscillations through.  Adding a first-difference term
penalises total variation directly:

```
Φ(p) = ‖W(Kp − I)‖² + λ [ α ‖L₂ p‖² + (1−α) ‖L₁ p‖² ]
```

At α < 1 the penalty disfavours slope changes, naturally smoothing
oscillations even when λ is slightly too small.  A secondary benefit: the
GCV landscape becomes steeper near the oscillation onset (the first-derivative
penalty is more sensitive there), making the GCV minimum easier to locate.

**Implementation:** `src/regularise.rs` — add `MixedDerivative { alpha: f64 }`;
`src/main.rs` — add `--reg-alpha` flag (default 1.0 = current behaviour).

The λ grid evaluations remain independent → **fully parallelisable with
`rayon::par_iter()`**.

---

### Fix D — Non-negativity via constrained solve (NNLS / projected gradient)
*(non-negativity strategy, medium–high effort)*

The iterative clipping cascade is the mechanism that converts an oscillating
solution into the "island" pattern.  Replacing it with a proper constrained
optimisation solves

```
min_p  ‖W(Kp − I)‖² + λ ‖Lp‖²   s.t.  p ≥ 0
```

directly.  Even with a slightly-too-small λ the constrained solution
distributes weight smoothly rather than creating islands.

Approaches ranked by implementation effort:
1. **Projected gradient descent** — after each gradient step, clamp
   negative components to zero; iterate until KKT conditions hold.  Warm-
   starts from the SVD already computed.
2. **Lawson-Hanson NNLS** — active-set algorithm; direct solve; can use
   existing SVD decomposition.  A Rust crate (`nnls`) is available.

Each λ evaluation remains independent → **fully parallelisable**.

**Implementation touch-point:** `src/nonneg.rs` — replace `IterativeClipping`
with `ProjectedGradient` or `NNLSSolver`.

---

### Fix E — SNR-based tail trimming
*(raw data preprocessing, low standalone impact)*

GNOM used only 1482 of 1793 points (up to q = 0.300 Å⁻¹).  `--snr-cutoff`
or `--qmax` is already implemented and reduces N, mildly steepening the GCV
landscape.  Insufficient on its own but good practice regardless.

---

## Top-2 recommended fixes

Given the requirement for robustness and parallelisability:

### 1 — Fix B: flat-landscape GCV fallback (implement first)

**Why first:**
- Zero runtime cost — reuses the already-evaluated grid.
- No API or behaviour change for well-behaved datasets.
- Directly proven to work on SASDYU3: ISE 12.14 → 0.010 in one step.
- ~20 lines confined to `src/lambda_select.rs`.
- The fallback (L-curve) is already implemented and correct.

### 2 — Fix D: constrained NNLS solve (implement second)

**Why second:**
- Addresses root cause 2 (clipping cascade) independently of the λ selector.
  If GCV ever sneaks through with a too-small λ on a future dataset, NNLS
  will still produce a physically reasonable (smooth, non-negative) solution
  rather than an island pattern.
- Each λ evaluation is independent → the full grid search is embarrassingly
  parallel; add `rayon::par_iter()` to `evaluate_lambda_grid` and get the
  entire compute cost down for free.
- The projected-gradient variant can warm-start from the existing SVD
  decomposition (`src/solver.rs`), keeping the per-evaluation overhead small.
- Fix C (first-derivative mixing) is a useful complement but adds a tuning
  parameter; Fix D is always-on with no knobs.

---

## User-configurable bypasses via a TOML config

For cases where the algorithmic fixes are insufficient or where an expert
user knows the right λ range a priori, a project-level config file
(`unfourier.toml` or `.unfourier.toml`) would let users hard-wire settings:

```toml
[regularisation]
lambda_min = 1e-2          # prevents GCV from reaching the under-regularised flat zone
lambda_max = 1e6
method     = "lcurve"      # override default GCV for all runs in this project

[preprocessing]
qmax           = 0.30      # match GNOM's q-range for SASDYU3
negative_handling = "clip"

[basis]
type    = "rect"
npoints = 100
```

Settings in the TOML would be overridden by explicit CLI flags, preserving
full command-line control.  This gives a clean escape hatch for edge cases
without complicating the default code path.

**Key bypass for SASDYU3:** setting `lambda_min = 1.0` in the TOML would
exclude the entire under-regularised region of the GCV landscape (where λ_eff
< 10⁵), preventing the flat-minimum failure entirely.
