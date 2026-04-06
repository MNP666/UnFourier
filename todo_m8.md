# M8 — Perceptual Constraints: Implementation Todo

## Background and motivation

The M7 validation exposed two residual failure modes that survive even after the
GCV-fallback and NNLS fixes:

1. **Open right boundary.** For the rect basis, no constraint forces P(r) → 0 as
   r → D_max. The last bin sits at D_max − Δr/2 and can take any positive value,
   producing a P(r) that is still elevated at the edge of the distance range. This
   is physically wrong: by definition, no pair distance can exceed D_max, so P(r)
   must be exactly zero there. The spline basis already enforces this structurally
   (the clamped endpoint basis functions are dropped), but the rect basis does not.

2. **Discontinuous P(r) at small λ.** Even with NNLS, if λ is slightly
   under-regularised the unconstrained solution oscillates enough that many
   consecutive bins end up with very similar small values while their differences
   are large. The second-derivative (curvature) penalty damps oscillations of
   period ≥ 3 bins but is relatively blind to single-step jumps between adjacent
   bins. Adding a first-derivative (slope) penalty fills this gap.

Both corrections are **perceptual** in the sense that they encode physics we know
to be true before seeing the data:
- P(r) is compactly supported on [0, D_max]
- P(r) is a smooth, unimodal-ish function with no sharp jumps

The implementation keeps both corrections as soft, tunable weights so that expert
users can disable or strengthen them via an `unfourier.toml` config file.

---

## Mathematical framework

### Existing Tikhonov problem

The solver currently minimises:

```
min_c  ‖K_w c − I_w‖²  +  λ_eff ‖L c‖²
```

where:
- `K_w = diag(1/σ) · K`  is the error-weighted kernel
- `I_w = I/σ`             is the error-weighted intensity
- `L = D₂`               is the second finite-difference matrix (n−2 × n)
- `λ_eff = λ · tr(KᵀK) / tr(LᵀL)` is the internally scaled regularisation strength

### Task 1 extension — boundary constraints

Add two virtual rows to the weighted system:

```
[ K_w    ]       [ I_w ]
[ w·e₁ᵀ ]  c  ≈ [  0  ]   ← P(r = 0)    = 0
[ w·eₙᵀ ]       [  0  ]   ← P(r = D_max) = 0
```

where `e₁` and `eₙ` are the first and last standard basis vectors. This is the
standard *augmented-system* approach for soft linear equality constraints in
least-squares. The weight `w` controls enforcement strength: large `w` makes the
constraint nearly hard; small `w` makes it advisory only.

Because the constraint rows look exactly like two additional data points, they flow
through GCV, L-curve, and Bayesian evidence without any special-casing. The
effective data count increases by 2, which is negligible for the datasets we use
(N ≥ 200). The Cholesky solve and all downstream diagnostics are unchanged.

**Choosing the default weight.**
We want the boundary constraint to be as influential as roughly √N data points
(enough to be decisive without dominating the fit). A natural scale is:

```
w_default = sqrt(N_data) × rms(1/σᵢ)
```

This is dimensionally consistent: `w²` has units of `1/σ²`, matching the data
rows. When `boundary_weight = 0` in the TOML (meaning "use default"), this formula
is used. When `boundary_weight = k > 0`, the weight is `k × w_default` (a
multiplier). When `boundary_weight = -1`, no constraint rows are added.

**Spline basis.** The spline polynomial IS zero at 0 and D_max by construction:
`build_kernel_matrix` drops the two endpoint B-spline columns (B₀ and B_{n-1}),
so no coefficient ever multiplies the basis functions that are nonzero at the
boundaries. However, `r_values()` returns only the *interior* Greville abscissae,
so the output file never contains an explicit (0, 0) or (D_max, 0) row — the
curve appears to end before D_max with a non-zero value. The fix (implemented
before M8, in `main.rs`) is to append explicit boundary rows to `solution.r`
and `solution.p_r` after the solve when the spline basis is used. Constraint
augmentation rows are therefore not needed for the spline basis.

### Task 2 extension — combined regulariser

Replace `L = D₂` with:

```
L_combined = [ sqrt(d₁) · D₁ ]
             [ sqrt(d₂) · D₂ ]
```

so that:

```
‖L_combined c‖² = d₁ ‖D₁ c‖²  +  d₂ ‖D₂ c‖²
```

where:
- `D₁` is the first finite-difference matrix ((n−1) × n): penalises slope
- `D₂` is the second finite-difference matrix ((n−2) × n): penalises curvature
- `d₁`, `d₂` are dimensionless relative weights

Crucially, `λ_eff` still scales the whole combined penalty, so the λ grid search
and all selectors are unchanged. The `trace_ratio` recomputation in
`GridMatrices::build` automatically adjusts for the different trace of
`L_combined^T L_combined`, keeping the user-facing λ in the same intuitive range.

When `d₁ = 0` and `d₂ = 1` (the default), the combined regulariser reduces
exactly to the existing `SecondDerivative` behaviour — no regression.

When `d₁ > 0`, each adjacent pair of bins is penalised for differing by more than
the regularisation allows. This specifically targets the single-step discontinuities
that the curvature penalty misses.

### Task 3 — TOML configuration

```toml
[constraints]
# Boundary enforcement: P(r=0) = P(r=D_max) = 0
# -1 = disabled (off)
#  0 = automatic default weight (sqrt(N) * rms(1/sigma))
# >0 = explicit multiplier on the default weight
boundary_weight = 0.0

# First-derivative slope penalty (added on top of curvature regularisation)
# -1 = disabled (off); existing second-derivative-only behaviour
#  0 = default relative weight 1.0
# >0 = explicit relative weight
d1_smoothness = -1.0
```

The `-1 = off, 0 = default, positive = custom` convention is consistent with
the M7 TOML design for `lambda_min` / `lambda_max`.

---

## Task 1 — Boundary constraints (`src/kernel.rs`, `src/config.rs`, `src/main.rs`)

### 1a — `src/config.rs`

- [x] Add `ConstraintsConfig` struct:
  ```rust
  #[derive(Debug, Deserialize, Default)]
  #[serde(deny_unknown_fields)]
  pub struct ConstraintsConfig {
      pub boundary_weight: Option<f64>,  // -1=off, 0=auto, >0=multiplier
      pub d1_smoothness:   Option<f64>,  // -1=off, 0=default 1.0, >0=explicit
  }
  ```
- [x] Add `pub constraints: ConstraintsConfig` field to `UnfourierConfig`.
- [x] Update the `parse_minimal_toml` and add a `parse_constraints_toml` unit test
      in the existing `tests` block.

### 1b — `src/kernel.rs`

- [x] Add `pub fn append_boundary_constraints`:
  ```rust
  pub fn append_boundary_constraints(
      k_w: &mut DMatrix<f64>,
      i_w: &mut Vec<f64>,
      weight: f64,
  )
  ```
  - Appends a row `[weight, 0, …, 0]` and target `0.0`
    (constrains the first basis coefficient = 0, i.e. P(r=0) = 0).
  - Appends a row `[0, …, 0, weight]` and target `0.0`
    (constrains the last basis coefficient = 0, i.e. P(r=D_max) = 0).
  - Both `k_w` and `i_w` grow by 2 rows/elements.
- [x] Unit test: call on a 3×2 matrix; verify shape becomes 5×2 and last two
      rows match the expected pattern.

### 1c — `src/main.rs`

- [x] After `Args::parse()` + TOML merge, read `cfg.constraints.boundary_weight`.
  - If `None` or `Some(0.0)`: use automatic default weight.
  - If `Some(-1.0)`: skip.
  - If `Some(k)` where k > 0: use `k × default_weight`.
- [x] After `build_weighted_system`, compute `default_weight`:
  ```rust
  let rms_inv_sigma = (data.error.iter().map(|s| 1.0/s/s).sum::<f64>()
                       / data.error.len() as f64).sqrt();
  let w_bc = (data.len() as f64).sqrt() * rms_inv_sigma * multiplier;
  ```
- [x] Call `append_boundary_constraints(&mut k_w, &mut i_w, w_bc)` for rect basis
      only (`BasisChoice::Rect`).
- [x] In verbose mode, print:
  ```
    [constraints] boundary P(0)=P(Dmax)=0  weight = {:.3e}
  ```
- [x] Unit test (in `kernel.rs`): a 5-point rect solve with a large boundary
      weight forces `p_r[0]` and `p_r[n-1]` to be < 1e-3 × peak.

---

## Task 2 — Combined regulariser (`src/regularise.rs`, `src/lambda_select.rs`, `src/solver.rs`, `src/main.rs`)

### 2a — `src/regularise.rs`

- [x] Add `pub struct FirstDerivative;` implementing `Regulariser`:
  - `name()` → `"first-derivative"`
  - `matrix(n)`: returns the (n−1) × n first finite-difference matrix:
    `D₁[i,i] = -1`, `D₁[i,i+1] = +1`.
- [x] Add `pub struct CombinedDerivative { pub d1_weight: f64, pub d2_weight: f64 }`
      implementing `Regulariser`:
  - `name()` → `"combined-derivative"`
  - `matrix(n)`: returns the stacked matrix
    `[ sqrt(d1_weight) * D1; sqrt(d2_weight) * D2 ]`
    with shape `((n-1) + (n-2)) × n = (2n-3) × n`.
  - When `d1_weight = 0.0` and `d2_weight = 1.0`, the Gram matrix must
    equal `SecondDerivative.gram_matrix(n)` exactly (no regression).
- [x] Unit tests:
  - `first_derivative_shape`: `FirstDerivative.matrix(4)` has shape 3×4.
  - `combined_matches_second_when_d1_zero`: for n=5,
    `CombinedDerivative { d1_weight: 0.0, d2_weight: 1.0 }.gram_matrix(5)`
    equals `SecondDerivative.gram_matrix(5)` entry-wise (tolerance 1e-12).
  - `combined_d1_increases_trace`: for n=10,
    `CombinedDerivative { d1_weight: 1.0, d2_weight: 1.0 }.gram_matrix(10).trace()`
    is strictly greater than `SecondDerivative.gram_matrix(10).trace()`.

### 2b — `src/lambda_select.rs` — decouple `GridMatrices` from `SecondDerivative`

Currently `GridMatrices::build` hardcodes `SecondDerivative` on line 375-376.
This is the only place in the λ-selection path that touches the regulariser.

- [x] Change the signature of `GridMatrices::build` to accept a precomputed
      `ltl: DMatrix<f64>` parameter:
  ```rust
  pub fn build(
      k_weighted:   &DMatrix<f64>,
      i_weighted:   &[f64],
      k_unweighted: &DMatrix<f64>,
      i_observed:   &[f64],
      sigma:        &[f64],
      r:            &[f64],
      ltl:          DMatrix<f64>,   // ← new parameter
  ) -> Self
  ```
  Remove the internal `let reg = SecondDerivative; let ltl = reg.gram_matrix(n_r);`
  lines and use the passed-in `ltl` instead.
- [x] Update the one call site in `src/main.rs` to pass `ltl` (see Task 2c).
- [x] No other callers exist; confirm with `grep -r "GridMatrices::build"`.

### 2c — `src/main.rs` — wire up combined regulariser

- [x] Read `cfg.constraints.d1_smoothness` after TOML merge:
  - `None` or `Some(-1.0)`: d₁ = 0.0 (SecondDerivative only, no regression).
  - `Some(0.0)`: d₁ = 1.0 (default relative weight).
  - `Some(k)` where k > 0: d₁ = k.
- [x] Build the Gram matrix once, before the solver:
  ```rust
  let reg: Box<dyn Regulariser> = if d1_weight > 0.0 {
      Box::new(CombinedDerivative { d1_weight, d2_weight: 1.0 })
  } else {
      Box::new(SecondDerivative)
  };
  let ltl = reg.gram_matrix(n_basis);
  ```
- [x] Pass `ltl.clone()` to `GridMatrices::build` (for λ-auto methods).
- [x] Set `solver.regulariser = reg` on `TikhonovSolver` (for manual λ method).
- [x] In verbose mode, print:
  ```
    [constraints] regulariser: combined  d1={:.2}  d2=1.00
  ```
  (or `second-derivative` when d₁ = 0).

---

## Task 3 — Validation and TOML example

### 3a — Smoke test via TOML

- [x] Verify boundary constraint fires on rect basis:
  ```bash
  printf '[constraints]\nboundary_weight = 0\n' > unfourier.toml
  ./target/release/unfourier data/dat_ref/SASDME2.dat \
      --rmax 55 --basis rect --verbose
  # expect: [constraints] boundary P(0)=P(Dmax)=0  weight = ...
  # expect: p_r[0] and p_r[99] both ≈ 0
  rm unfourier.toml
  ```
- [x] Verify d1 smoothness fires:
  ```bash
  printf '[constraints]\nd1_smoothness = 0\n' > unfourier.toml
  ./target/release/unfourier data/dat_ref/SASDME2.dat \
      --rmax 55 --basis rect --verbose
  # expect: [constraints] regulariser: combined  d1=1.00  d2=1.00
  rm unfourier.toml
  ```
- [x] Verify default behaviour is unchanged (no TOML, no flags):
  ```bash
  ./target/release/unfourier data/dat_ref/SASDME2.dat --rmax 55 --basis rect
  # no [constraints] lines in output; P(r) identical to pre-M8
  ```

### 3b — Validation script

- [x] Run `python Dev/validate_real_data.py` with default settings; all existing
      pass/fail counts must be unchanged (no regression).
- [x] Run with boundary + d1 enabled for SASDF42 (the problem child):
  ```bash
  printf '[constraints]\nboundary_weight = 0\nd1_smoothness = 0\n' > unfourier.toml
  python Dev/validate_real_data.py
  rm unfourier.toml
  ```
  Expected: SASDF42 ISE improves; SASDME2 and SASDYU3 unchanged or better.

---

## Ordering rationale

| Task | Priority | Why |
|------|----------|-----|
| 1 — Boundary constraints | highest | ~40 lines; fixes the open right boundary on rect; zero risk of regression on spline |
| 2 — Combined regulariser | high | Decouples GridMatrices from SecondDerivative (a latent design debt); enables d1 smoothing |
| 3 — Validation | required | Confirms no regression before merging |

---

## Definition of done

```bash
cargo test          # all existing + new tests pass
cargo build --release

# Boundary constraint smoke test
printf '[constraints]\nboundary_weight = 0\n' > unfourier.toml
./target/release/unfourier data/dat_ref/SASDME2.dat \
    --rmax 55 --basis rect --verbose 2>&1 | grep constraints
rm unfourier.toml

# Regression check: default output must not change
./target/release/unfourier data/dat_ref/SASDME2.dat --rmax 55 --basis rect \
    > /tmp/after.dat
# compare with a pre-M8 reference if available

# Full validation
python Dev/validate_real_data.py
```

Expected outcome:
- `cargo test` passes with ≥ 5 new tests (2 in regularise.rs, 1 in kernel.rs,
  1 in config.rs, 1 integration).
- Default runs (no TOML) produce identical output to pre-M8.
- `boundary_weight = 0` makes last rect bin ≈ 0 on all three datasets.
- `d1_smoothness = 0` reduces jump amplitude between consecutive bins,
  visible in the validation plot's rect-basis row.
