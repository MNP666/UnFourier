# CODEX.md

This note is a strategy for fixing the M8 boundary problem in unFourier: fitted
I(q) can look acceptable while the reported P(r) is not physical at r = 0 and
r = D_max. The target is not just "make the first and last plotted values zero";
the target is that the fitted real-space function itself goes to zero in a
smooth, model-consistent way.

## Diagnosis

The current failure comes from mixing three different objects:

1. The coefficient vector `c` solved by the optimizer.
2. The real-space function `P(r) = sum_j c_j phi_j(r)`.
3. The sampled output table written as `r, P(r)`.

The rectangular basis used to hide this distinction because coefficients are bin
heights. That was useful for the first vertical slice, but it is not a physical
production basis. For the B-spline basis, coefficients are spline control
weights, not exact P(r) samples. Writing spline coefficients directly as P(r) can
make the output look non-physical even when the spline expansion is better
behaved than the table suggests.

There is also a more direct bug in the current M8 path: `main.rs` inserts
boundary rows and then hard-sets the first and last interior `solution.p_r`
values to zero after the solve. That creates a visible step in the output and
also makes `solution.p_r` inconsistent with `solution.i_calc`, because the
back-calculated fit was computed from the pre-clamped coefficients.

The important rule for M8 is therefore:

```text
Do not change P(r) after solving unless I(q) is recomputed from the same
coefficients. Prefer constraints in the basis/linear system over post-hoc edits.
```

## Physical Model

P(r) should satisfy at least:

```text
P(0) = 0
P(D_max) = 0
P(r) >= 0
```

For a smooth particle-like distribution, a stronger spline boundary model is
often useful:

```text
P'(0) = 0
P'(D_max) = 0
```

This stronger condition should be built into the spline basis. The rectangular
basis should be removed rather than patched further.

## Preferred Solution

Separate "solve coefficients" from "output samples".

Make the spline basis evaluate the real-space expansion:

```rust
trait BasisSet {
    fn r_values(&self) -> &[f64];
    fn r_max(&self) -> f64;
    fn build_kernel_matrix(&self, q: &[f64]) -> DMatrix<f64>;

    fn output_grid(&self) -> Vec<f64>;
    fn evaluate_pr(&self, coeffs: &[f64], r: &[f64]) -> Vec<f64>;
}
```

Then the solver should return coefficients, and `main.rs` should convert them to
sampled P(r) exactly once:

```text
coefficients -> basis.evaluate_pr(coefficients, output_grid) -> output P(r)
coefficients -> K * coefficients -> I_calc
```

For `CubicBSpline`, `evaluate_pr` must call `bspline::basis_matrix(...)` and
multiply by the solved coefficients. The output grid should include `0.0` and
`D_max`, plus a dense uniform grid or the Greville abscissae. A dense grid is
better for plotting because it shows the actual spline curve rather than the
control polygon.

Because this is now spline-only, keeping `BasisSet` is optional. It is reasonable
to keep it for one pass to reduce churn, then collapse it later if it no longer
earns its keep.

## Boundary Enforcement

Use hard constraints by construction where possible. Avoid post-hoc clamping.

### Spline Basis

Implement boundary conditions in the spline parameterisation.

For value-zero endpoints:

```text
full spline coefficients: [0, c0, c1, ..., cN, 0]
```

This is equivalent to dropping the endpoint B-spline columns from the kernel.
It makes P(0) and P(D_max) exactly zero.

For smooth value-and-slope-zero endpoints:

```text
full spline coefficients: [0, 0, c0, c1, ..., cN, 0, 0]
```

For a clamped cubic B-spline, this also makes the first derivative zero at both
ends. This is the cleanest way to avoid vertical cliffs in spline output.

The implementation should build the full B-spline basis internally, then map
free coefficients into the full coefficient vector with fixed zeros at the
boundary. That same mapping should be used for:

1. Kernel columns used by the solver.
2. Regularisation matrix.
3. P(r) evaluation for output.

This keeps the math consistent.

## Regularisation Strategy

Do not hand-code boundary regularisers that can drift away from the actual basis
constraints. Instead, build regularisation in full coefficient space and project
it onto the free coefficients.

For example:

```text
c_full = B * c_free
K_free = K_full * B
L_free = L_full * B
```

Here `B` is a sparse/free-column mapping that inserts fixed zero boundary
coefficients. For the spline-only code path it needs to handle:

1. Spline value-zero: `[0, c..., 0]`
2. Spline value-and-slope-zero: `[0, 0, c..., 0, 0]`

The existing first- and second-derivative penalties can then be applied to
`c_full`, while the solver only sees `c_free`.

Use the combined penalty as the default smoothness model:

```text
||L c||^2 = d1 * ||D1 c||^2 + d2 * ||D2 c||^2
```

Recommended defaults:

```toml
[constraints]
d1_smoothness = 0.1
spline_boundary = "value_zero"
```

Keep `d1_smoothness = -1.0` as an escape hatch for second-derivative-only
behaviour. Keep `spline_boundary = "value_slope_zero"` as an expert option for
datasets where endpoint slopes should be forced flatter.

## Knot Density Strategy

The number of spline knots/basis functions affects endpoint behaviour and the
whole bias-variance tradeoff.

Too few basis functions can make the curve too stiff near r = 0 and D_max. Too
many basis functions let the solution express local wiggles unless regularisation
is strong enough. The right control is often not an absolute count, but a target
real-space resolution, such as "one spline control region every 5-8 Angstrom".

Add an optional knot-spacing control before attempting any two-pass Dmax
refinement:

```toml
[basis]
# Use exactly this many free spline coefficients.
n_basis = 20

# Alternative: choose n_basis from Dmax. Ignored if n_basis is set.
knot_spacing = 7.5
min_basis = 12
max_basis = 48
```

CLI equivalents:

```bash
unfourier data.dat --n-basis 24
unfourier data.dat --knot-spacing 7.5 --min-basis 12 --max-basis 48
```

Resolution rule:

```text
if n_basis is set:
    use n_basis
else if knot_spacing is set:
    n_basis = ceil(D_max / knot_spacing), clamped to [min_basis, max_basis]
else:
    use default n_basis = 20
```

This gives Dmax-dependent basis complexity without adding a second solve. A
two-pass strategy can come later:

```text
initial Dmax -> first spline solve -> infer effective Dmax -> rebuild basis -> resolve
```

That second pass should wait until the simpler knot-spacing option has been
validated, because Dmax inference needs careful thresholds and can be fooled by
regularisation tails or noise.

## Implementation Plan

1. Remove the rectangular basis from the active code path.

   Delete `UniformGrid` from `src/basis.rs`. Remove `BasisChoice`, the `--basis`
   flag, and the `--npoints` flag from `src/main.rs`. Construct `CubicBSpline`
   unconditionally. Keep `--n-basis` as the exact resolution knob.

2. Add knot-density controls.

   Add `--knot-spacing`, `--min-basis`, and `--max-basis` as optional CLI
   controls. Add matching `[basis]` TOML fields. `--n-basis` wins when supplied.

3. Add an evaluated-output path.

   Add `output_grid` and `evaluate_pr` methods to `BasisSet`. Update
   `CubicBSpline`. The spline implementation should evaluate the actual
   B-spline expansion, not return raw coefficients.

4. Stop post-hoc P(r) mutation.

   Remove the boundary insertion and hard interior zeroing block in `main.rs`.
   Any displayed boundary zeros must come from `basis.evaluate_pr`, not from
   editing `solution.p_r` after `I_calc` has already been computed.

5. Make `Solution` explicit about coefficients.

   Either add a `coeffs: Vec<f64>` field or rename the internal solved vector so
   it is not confused with sampled P(r). The output writer should receive the
   evaluated P(r), while fit calculation should always use coefficients.

6. Implement spline boundary modes.

   Start with two modes:

   ```text
   value_zero        -> full coefficients [0, c..., 0]
   value_slope_zero  -> full coefficients [0, 0, c..., 0, 0]
   ```

   `value_zero` should be the default. `value_slope_zero` should be available as
   a toggle because it can suppress legitimate small-r signal when the basis is
   too coarse or regularisation is too strong.

7. Build regularisation through the same free-to-full mapping.

   Replace ad hoc boundary anchoring with `L_free = L_full * B`. This prevents
   the regulariser, kernel, and output evaluator from disagreeing about which
   coefficients exist at the boundary.

8. Update config and docs.

   Add spline boundary and knot-density options:

   ```toml
   [basis]
   n_basis = 20
   knot_spacing = 7.5
   min_basis = 12
   max_basis = 48

   [constraints]
   spline_boundary = "value_zero"  # value_zero | value_slope_zero
   d1_smoothness = 0.1
   ```

   Remove or de-emphasise `boundary_weight` if hard coefficient elimination is
   used. A hard basis constraint is easier to reason about than pseudo-data rows.

9. Remove rectangular validation comparisons.

   Update the validation scripts so spline is the only active run. Historical
   notes can keep mentioning rectangles, but active scripts should not require a
   removed CLI option.

## Rectangular Basis Removal Plan

Breaking changes are acceptable, so prefer a clean deletion over compatibility
shims.

### Rust Code

1. `src/main.rs`

   Remove:

   ```text
   BasisChoice
   --basis
   --npoints
   match args.basis { ... }
   UniformGrid import
   rect-specific verbose text
   ```

   Keep:

   ```text
   --n-basis <N>  default 20
   --knot-spacing <A>
   --min-basis <N>
   --max-basis <N>
   CubicBSpline::new(r_max, n_basis)
   ```

   The verbose line becomes simply:

   ```text
   basis: cubic-b-spline  n_basis=...
   ```

2. `src/basis.rs`

   Delete `UniformGrid` and its tests. Keep `CubicBSpline`; optionally rename it
   later to `SplineBasis` once the boundary work settles. Do not rename it during
   the first deletion pass unless the compiler errors are already small.

3. `src/kernel.rs`

   Replace tests that instantiate `UniformGrid` with a tiny `CubicBSpline`.
   Delete `append_boundary_constraints` if hard spline coefficient elimination is
   used. If any soft constraint helper remains, it should be documented as a
   generic linear constraint helper, not as a rect boundary mechanism.

4. `src/config.rs`

   Remove `basis.type` and `basis.npoints`. Add spline-specific fields:

   ```toml
   [basis]
   n_basis = 20
   knot_spacing = 7.5
   min_basis = 12
   max_basis = 48
   ```

   If preserving the old `npoints` spelling is not useful, do not support it.
   Failing fast on old config is fine for this project.

5. `src/regularise.rs`

   Keep only regularisers that are meaningful for splines. The safest interim
   path is to leave `FirstDerivative`, `SecondDerivative`, and
   `CombinedDerivative`, then remove `BoundaryAnchoredCombined` once
   `L_free = L_full * B` is implemented.

6. `src/solver.rs` and `src/lambda_select.rs`

   These should not need rect-specific logic. They should operate on whatever
   kernel and regularisation matrices the spline basis supplies.

### Validation Scripts

1. `Dev/validate_real_data.py`

   Remove rect runs, rect plots, and pass/fail summaries labelled "rect". Keep a
   single spline result per dataset, plus optional rebin variants where useful.

2. `Dev/validate_m5.py`

   Retire or rewrite this as a spline regression script. The old purpose was to
   prove splines beat rectangles; that question is closed. New checks should be:

   ```text
   spline output endpoints are zero
   evaluated P(r) is not raw coefficients
   all lambda methods run
   Debye ISE is reasonable
   ```

3. `Dev/sweep_noise.py`

   Make spline the default and remove rect labels from plots.

4. `Dev/monte_carlo_coverage.py`

   This script currently assumes the rectangular kernel/grid. Either rewrite its
   reference grid around spline evaluation or temporarily mark it as retired
   until Bayesian coverage is revalidated for splines.

### Docs

Update active docs:

```text
README.md
CLAUDE.md
pipeline.md
CODEX.md
```

Remove active mentions of:

```text
--basis rect|spline
--npoints
UniformGrid
boundary_weight as a rect-only mechanism
rect-vs-spline validation
```

Historical milestone files in `docs/` can remain as archaeological notes, but
active docs should describe only the spline implementation.

### Search Checklist

Before calling the removal done, these commands should return no active source
or README hits except historical docs or generated plots:

```bash
rg -e "UniformGrid|BasisChoice|--basis|--npoints|rect" src README.md CLAUDE.md pipeline.md Dev
rg -e "append_boundary_constraints|boundary_weight" src README.md CLAUDE.md pipeline.md
```

### Recommended Order

1. Remove CLI branching and instantiate `CubicBSpline` unconditionally.
2. Delete `UniformGrid` and fix compiler errors.
3. Update kernel/config tests away from rectangles.
4. Add knot-spacing controls.
5. Remove post-hoc boundary mutation.
6. Add spline `evaluate_pr` output.
7. Implement spline boundary modes and coefficient mapping.
8. Update validation scripts.
9. Update active docs.
10. Run `cargo test`, `cargo build --release`, and spline validation.

## Validation Plan

Use Debye chain data as the primary noisy benchmark. Use sphere data only for
kernel/evaluation sanity checks.

Minimum checks:

1. `cargo test`
2. `cargo build --release`
3. Generate Debye data with `python Dev/gen_debye.py`.
4. Run spline GCV:

   ```bash
   ./target/release/unfourier Dev/debye_k5.dat \
       --n-basis 20 --rmax 150 \
       --output /tmp/pr_spline.dat --fit-output /tmp/fit_spline.dat --verbose
   ```

5. Confirm from `/tmp/pr_spline.dat`:

   ```text
   first P(r) ~= 0
   last P(r)  ~= 0
   no hard jump between endpoint and next plotted sample
   P(r) remains non-negative
   ```

6. Confirm `I_calc` was computed from the same coefficients used to evaluate
   P(r). No post-solve mutation is allowed.

7. Run `python Dev/validate_real_data.py` and compare SASDME2, SASDF42, and
   SASDYU3 against the current baseline.

8. Sweep knot density on at least Debye and one real dataset:

   ```text
   n_basis: 12, 16, 20, 28, 36
   knot_spacing: 5.0, 7.5, 10.0
   boundary: value_zero, value_slope_zero
   d1_smoothness: 0.0, 0.03, 0.1, 0.3
   ```

## Success Criteria

M8 is done when:

1. Spline output contains exact zero endpoint values.
2. Spline output approaches both endpoints smoothly without post-hoc clamping.
3. The plotted P(r) is the evaluated spline function, not the raw coefficient
   vector.
4. `I_calc` and output P(r) are derived from the same solved coefficients.
5. The active CLI and docs no longer expose or recommend rectangular bins.
6. `rg -e "UniformGrid|BasisChoice|--basis|--npoints|rect" src README.md CLAUDE.md pipeline.md Dev`
   has no active-code hits.
7. `--knot-spacing` and `[basis].knot_spacing` choose a sensible clamped
   `n_basis` from Dmax, while explicit `--n-basis` still wins.
8. Validation does not materially worsen chi^2 or the existing real-data checks.
