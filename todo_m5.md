# M5 — Cubic B-spline Basis: Implementation Todo

## Design decision: which spline parameterisation?

The Python prototype (`Dev/spline_pr_tests.py`) uses `scipy.interpolate.CubicSpline`
as an **interpolating spline**: coefficients are P(r) values at knot positions, and
the boundary P(0) = P(r_max) = 0 is enforced by fixing the endpoint y-values to zero
with clamped first-derivative BCs.

**This file recommends a proper B-spline basis instead.** Reasons:

- B-spline basis functions are non-negative and form a partition of unity — this
  makes the non-negativity constraint and the regulariser more physically meaningful.
- Coefficients are control-point ordinates, not function values: the solver only sees
  a standard linear system `Kc ≈ I`.
- Boundary conditions are enforced structurally (by dropping two columns of the design
  matrix) rather than by fixing specific coefficient values post-solve.
- Standard in the IFT literature (GNOM uses B-splines).

**Boundary condition enforcement** — clamped knot vector + drop endpoint basis
functions:

For cubic B-splines with a clamped (open) knot vector:

```
knots = [0, 0, 0, 0,  t_1, t_2, ..., t_m,  r_max, r_max, r_max, r_max]
```

this gives `n = m + 4` total basis functions. The clamped knot vector guarantees
`B_0(0) = 1` and `B_{n-1}(r_max) = 1`, with all other basis functions zero at those
points. So if we simply **exclude columns 0 and n-1 from the design matrix**, the
reconstructed P(r) is exactly zero at both endpoints for any coefficient vector — no
post-hoc clamping needed. The free parameters are `c_1 .. c_{n-2}`.

---

## Todo list

### 1 — Update the Python prototype

File: `Dev/spline_pr_tests.py`

- [ ] Replace the `CubicSpline` interpolating approach with a proper B-spline design
      matrix using `scipy.interpolate.BSpline` and `make_interp_spline` / manual knot
      construction via `splev`/`BSpline.design_matrix`.
- [ ] Implement `make_bspline_basis(r_max, n_interior_knots, degree=3)` that returns:
  - the clamped knot vector
  - the design matrix `B` of shape `(n_r_eval, n_free)` excluding the two endpoint
    columns (n_free = n_total - 2)
  - Greville abscissae for the free basis functions
- [ ] Verify P(0) = 0 and P(r_max) = 0 for arbitrary coefficient vectors by
      evaluating `B @ c` at r=0 and r=r_max.
- [ ] Compute the kernel matrix `K[i,j] = 4π ∫ B_j(r) sin(q_i r)/(q_i r) dr`
      numerically (Gauss-Legendre over each knot span) and compare to the rectangular
      basis kernel on the same q grid.
- [ ] Recover P(r) on clean Debye data using `scipy.optimize.nnls` with the B-spline
      K matrix. Confirm the result is at least as good as the rectangular basis with
      the same number of free parameters.
- [ ] Add a convergence check: increase n_interior_knots from 10 → 50 and plot
      residual and P(r) quality vs knot count.

### 2 — Rust: Cox–de Boor B-spline evaluation

File: new `src/bspline.rs` (internal utility, not pub in the trait hierarchy)

- [ ] Implement `fn clamped_knots(r_max: f64, n_interior: usize) -> Vec<f64>` that
      produces `[0,0,0,0, t_1,...,t_n_interior, r_max,r_max,r_max,r_max]` with
      uniformly spaced interior knots.
- [ ] Implement `fn basis_matrix(knots: &[f64], degree: usize, r: &[f64]) -> DMatrix<f64>`
      using the Cox–de Boor recursion. Returns the full `(n_r, n_basis)` matrix
      including all basis functions (including the two endpoint ones).
- [ ] Implement `fn greville(knots: &[f64], degree: usize) -> Vec<f64>` returning the
      Greville abscissae `ξ_j = (t_{j+1} + ... + t_{j+degree}) / degree`.
- [ ] Unit test: on a 5-knot-span clamped grid, verify that the basis functions sum
      to 1 at every interior point (partition of unity).
- [ ] Unit test: verify `B[0, 0] == 1.0` and `B[0, 1..] == 0`, and similarly at
      `r = r_max` for the last column.

### 3 — Rust: kernel integral via Gauss–Legendre quadrature

File: `src/bspline.rs` or `src/kernel.rs`

- [ ] Implement 5-point Gauss–Legendre quadrature nodes and weights as a `const`
      array (or compute once at startup).
- [ ] Implement `fn integrate_basis_sinc(knots: &[f64], j: usize, q: f64) -> f64`
      that integrates `B_j(r) · sinc(q·r)` over the support `[t_j, t_{j+4}]` by
      summing Gauss–Legendre over each of the (up to 4) knot spans.
  - Handle the `q → 0` limit: `sinc(0) = 1`, so the integral becomes
    `∫ B_j(r) dr`, which can also be computed by the same quadrature.
- [ ] Verify the quadrature against the Python kernel matrix (same knots, same q
      values — values should match to < 1e-6 relative error).

### 4 — Rust: `CubicBSpline` struct implementing `BasisSet`

File: `src/basis.rs`

- [ ] Add `pub struct CubicBSpline { knots: Vec<f64>, r_free: Vec<f64>, r_max: f64 }`
      where `r_free` holds the Greville abscissae of the **free** basis functions
      (indices 1..n-2).
- [ ] `CubicBSpline::new(r_max: f64, n_basis: usize) -> Self` — `n_basis` is the
      number of free parameters (= n_interior_knots + 2). Suggest a default of 20.
- [ ] Implement `BasisSet`:
  - `r_values()` → Greville abscissae of the free basis functions
  - `r_max()` → r_max
  - `build_kernel_matrix(q)` → `(n_q × n_basis)` matrix using the quadrature
    integral, one column per free basis function (columns 1..n-2 of the full basis)
- [ ] The existing `SecondDerivative` regulariser works unchanged on the control
      points of the free basis functions — no modifications needed.

### 5 — Rust: CLI integration

File: `src/main.rs`

- [ ] Add `--basis rect|spline` flag (default: `rect` to preserve existing behaviour).
- [ ] Add `--n-basis N` flag (default: 20 for spline, existing n_points logic for
      rect). The flag name `--n-basis` is more general than `--n-points`.
- [ ] Wire the flag into the pipeline: construct either `UniformGrid` or `CubicBSpline`
      and pass as `&dyn BasisSet`. No other pipeline changes needed.
- [ ] Verbose output (`--verbose`): print basis type and n_basis alongside the
      existing λ and r_max diagnostics.

### 6 — Validation

- [ ] **Numerical match**: run the Rust B-spline kernel on a test q-grid and compare
      to the Python prototype. Max absolute difference < 1e-5.
- [ ] **Boundary conditions**: assert P(0) ≈ 0 and P(r_max) ≈ 0 in the output for
      all methods (GCV, L-curve, Bayes).
- [ ] **Debye recovery**: `unfourier --basis spline Dev/debye_k5.dat` should produce
      P(r) quality equal to or better than `--basis rect` with the same n_basis.
- [ ] **Fewer parameters**: confirm that 20 spline basis functions gives comparable or
      better P(r) quality than 100 rectangular bins on Debye data.
- [ ] **All methods work**: verify GCV, L-curve, and Bayes all run to completion with
      `--basis spline` and produce physically reasonable output.
- [ ] **Sweep**: run `Dev/sweep_noise.py` with both bases and overlay the results.

---

## Notes

**Why not use the interpolating spline from the Python prototype?**
The `CubicSpline` approach ties each coefficient to a function *value* at a knot.
This makes the regulariser act on point values rather than on the smoothness of the
spline representation. B-spline control points give a cleaner separation between the
basis geometry (knot placement) and the coefficients (free variables).

**Regulariser: finite-difference vs analytic**
The existing `SecondDerivative` regulariser applies finite differences to the
B-spline control points. This is an approximation of the true analytic smoothness
penalty `∫ [P''(r)]² dr = cᵀ H c` where `H_jk = ∫ B''_j(r) B''_k(r) dr`. The
analytic version would require implementing B-spline derivative evaluation and a
Gram matrix integral. It is a worthwhile future improvement (M5+ or M6) but the
finite-difference approximation is adequate and requires no changes to the existing
`Regulariser` trait.

**Knot count guidance**
A good default is `n_basis = 20` free parameters (i.e., 22 total B-spline functions,
22 + 4 - 1 = 25 knots including the 3 repeated ones at each end). For the rectangular
basis the current default is typically 50–100 bins. The spline achieves comparable
accuracy with fewer parameters because the basis functions have wider, overlapping
support.
