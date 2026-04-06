# M7 — Robustness and Configuration: Implementation Todo

## Overview

Three targeted improvements following the SASDYU3 validation failure analysis
(see `SASDYU3_val.md`):

1. **Fix B** — Flat-landscape GCV fallback: detect when GCV cannot reliably
   select λ and automatically use L-curve instead.
2. **Fix D** — Constrained NNLS solve: replace the iterative clipping cascade
   with a proper non-negative least-squares solve that produces smooth P(r)
   even when λ is slightly under-regularised.
3. **TOML config** — User-configurable bypass file for edge cases, with at
   minimum `qmin`, `qmax`, and `lambda_min`/`lambda_max` settings.

---

## Task 1 — Fix B: Flat-landscape GCV fallback (`src/lambda_select.rs`)

**Problem:** When N/M ≫ 1 (e.g. SASDYU3: 1696 pts / 100 bins ≈ 17), the GCV
score varies by only ~6% across 9 decades of λ. GCV selects the wrong (too-
small) λ, producing an oscillating P(r) that the iterative clipper turns into
28 isolated non-zero islands.

**Fix:** In `GcvSelector::select`, after collecting the finite GCV values,
compute the relative variation `(gcv_max − gcv_min) / gcv_min`. If it is less
than `GCV_FLAT_THRESHOLD` (10%), emit a warning to stderr and delegate to
`LCurveSelector.select(candidates)` — which reuses the already-computed
evaluations at zero extra cost.

- [x] Add `const GCV_FLAT_THRESHOLD: f64 = 0.10;` near the top of
      `src/lambda_select.rs`.
- [x] In `GcvSelector::select`, after filtering finite GCV values, compute
      `gcv_min` and `gcv_max`; if `(gcv_max - gcv_min) / gcv_min < GCV_FLAT_THRESHOLD`,
      print a warning to stderr and `return LCurveSelector.select(candidates);`.
- [x] Unit test: a synthetic `candidates` slice where all GCV values differ by
      < 10% triggers the fallback; assert the returned index matches the L-curve
      winner (not the GCV minimum).
- [x] Unit test: a synthetic `candidates` slice with > 10% GCV variation returns
      the GCV minimum as before (no regression for well-behaved data).

**Expected outcome on SASDYU3:** ISE drops from 12.14 → ~0.010. Well-behaved
datasets (SASDME2, SASDF42) are unaffected.

Files: `src/lambda_select.rs` only (~20 lines + 2 tests).

---

## Task 2 — Fix D: Constrained NNLS solve (`src/nonneg.rs`, `src/solver.rs`)

**Problem:** When λ is slightly too small, the unconstrained Tikhonov solution
oscillates. The iterative clipping loop zeros out negative bins and re-solves,
but with large oscillations most bins are eventually zeroed, leaving an unphysical
"island" pattern.

**Fix:** Replace the iterative re-solve loop with a proper non-negative least-
squares (NNLS) solve. Even with a too-small λ, NNLS distributes weight smoothly
rather than creating islands. This fix is independent of Fix B: if GCV ever
sneaks through with a bad λ on a future dataset, NNLS will still give a
physically reasonable result.

Recommended approach: **projected gradient descent** warm-started from the
unconstrained SVD solution. Each λ evaluation is independent, so the full grid
search is ready for `rayon::par_iter()`.

### 2a — `ProjectedGradient` in `src/nonneg.rs`

- [x] Add `pub struct ProjectedGradient { pub max_iter: usize, pub tol: f64 }`
      with a sensible default (`max_iter = 500`, `tol = 1e-8`).
- [x] Implement `NonNegativityStrategy`:
  - `name()` → `"projected-gradient"`
  - `find_violations`: sign check (negative indices) for diagnostic use.
  - `is_constraining()` → `true` (new trait method; `NoConstraint` returns `false`).
- [x] Add `pub fn projected_gradient_nnls(a, b, warm_start, max_iter, tol)` —
      the shared PG loop used by both `solver.rs` and `lambda_select.rs`.
      Step size: `1/trace(A)` (guaranteed descent, no eigendecomposition needed).
- [x] Unit test: `find_violations` on `[-1, 2, -0.5]` returns `[0, 2]`.
- [x] Unit test: `projected_gradient_nnls` on a 3×3 identity system recovers
      the correct NNLS solution in 1 step.

### 2b — Projected-gradient loop in `src/solver.rs` and `src/lambda_select.rs`

- [x] `TikhonovSolver::new` now uses `ProjectedGradient::default()` and
      `max_nonneg_iter = 500`.
- [x] Replaced the active-set clipping loop in `TikhonovSolver::solve` with a
      call to `projected_gradient_nnls` (warm-started from unconstrained solve).
      `NoConstraint` path unchanged (bypasses PG loop via `is_constraining()`).
- [x] Replaced `apply_nonneg_tikhonov` + `solve_active` in `lambda_select.rs`
      with a direct call to `projected_gradient_nnls` (reuses the unconstrained
      solution already computed for GCV/L-curve metrics).
- [x] Added `// rayon::par_iter() ready` comment at the `evaluate_lambda_grid`
      loop.
- [x] Unit test in `solver.rs`: 3×3 identity system recovers `[0.1, 2.0, 0.0]`.

Files: `src/nonneg.rs`, `src/solver.rs`, `src/lambda_select.rs`, `Dev/validate_real_data.py`.

---

## Task 3 — TOML config (`src/config.rs`, `src/main.rs`, `Cargo.toml`)

**Purpose:** Allow expert users (and test scripts) to hard-wire settings that
bypass edge-case behaviour without touching CLI defaults. CLI flags always
override TOML values.

Key bypass for SASDYU3: `[regularisation] lambda_min = 1.0` prevents GCV from
reaching the under-regularised flat zone entirely.

### 3a — `Cargo.toml` additions

- [x] Add dependencies:
  ```toml
  serde = { version = "1", features = ["derive"] }
  toml  = "0.8"
  ```

### 3b — `src/config.rs` (new file)

- [x] Define `UnfourierConfig` with serde-derived `Deserialize`:
  ```toml
  [regularisation]
  method     = "gcv"     # "gcv" | "lcurve" | "bayes"
  lambda_min = 1e-6
  lambda_max = 1e3

  [preprocessing]
  qmin              = 0.0   # 0.0 = no lower bound
  qmax              = 0.0   # 0.0 = no upper bound
  negative_handling = "clip" # "clip" | "omit" | "keep"

  [basis]
  type    = "rect"   # "rect" | "spline"
  npoints = 100
  ```
- [x] All fields optional (via `Option<T>`) so a partial TOML only overrides
      the specified keys.
- [x] `UnfourierConfig::load() -> Result<Option<Self>>`: looks for
      `unfourier.toml` in the current working directory; returns `None` if
      absent, `Err` if present but unparseable.

### 3c — Merge logic in `src/main.rs`

- [x] After `Args::parse()`, call `UnfourierConfig::load()?`.
- [x] For each CLI `Option<T>` arg (`lambda_min`, `lambda_max`, `method`,
      `qmin`, `qmax`, `negative_handling`, `npoints`): if the CLI arg is
      `None`, fill it from the TOML value (if present).
- [x] CLI args that are already `Some(...)` are left unchanged (CLI wins).
- [x] In verbose mode, print `"  [config] loaded from unfourier.toml"` and
      list each key that was filled from the file.
- [x] Unit test (in `src/config.rs`): parse a minimal TOML string and verify
      the correct fields are populated.

Files: `src/config.rs` (new), `src/main.rs`, `Cargo.toml`.

---

## Ordering rationale

| Task | Priority | Why |
|------|----------|-----|
| 1 — GCV fallback | highest | ~20 lines, zero cost, proven ISE 12→0.01 on SASDYU3 |
| 2 — NNLS | high | Independent safety net; parallelisable; fixes root cause 2 |
| 3 — TOML config | medium | User escape hatch; also enables reproducible test configs |

---

## Definition of done

```bash
cargo test
cargo build --release

# Task 1: GCV fallback fires on SASDYU3
./target/release/unfourier data/dat_ref/SASDYU3.dat \
    --rmax 80 --negative-handling clip --method gcv --verbose
# expect stderr: "GCV landscape flat (6.4% variation); falling back to L-curve"

# Task 2: NNLS produces smooth P(r)
./target/release/unfourier data/dat_ref/SASDYU3.dat \
    --rmax 80 --negative-handling clip --method gcv --verbose
# expect: all 100 bins non-zero

# Task 3: TOML bypass for SASDYU3
printf '[preprocessing]\nqmax = 0.30\n' > unfourier.toml
./target/release/unfourier data/dat_ref/SASDYU3.dat --rmax 80 --verbose
rm unfourier.toml

# Full validation: all three datasets (SASDYU3 ISE should now pass)
python Dev/validate_real_data.py
```

Expected outcome: all three datasets pass ISE < 0.15 and Rg within 15%.
