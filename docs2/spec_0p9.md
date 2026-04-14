# unFourier 0.9 Spec

This spec turns the strategy in `CODEX.md` into implementation epics and issues.
Version 0.9 is allowed to break compatibility. The goal is to simplify the
project around the spline basis and fix P(r) boundary behaviour without carrying
the old rectangular-basis scaffolding forward.

## Goals

1. Make cubic B-splines the only active basis.
2. Remove rectangular-bin CLI/config/docs/validation paths.
3. Stop treating spline coefficients as sampled P(r).
4. Remove post-solve P(r) mutation.
5. Enforce boundary behaviour through spline parameterisation.
6. Add a soft neighboring-coefficient smoothness penalty.
7. Add knot-density controls based on Dmax.
8. Keep validation exploratory and transparent.

## Non-Goals

1. No two-pass automatic Dmax refinement in 0.9.
2. No adaptive/nonuniform knot placement in 0.9.
3. No public compatibility shim for `--basis rect` or `--npoints`.
4. No promise that `value_slope_zero` is the default boundary mode.
5. No rewrite of the whole solver architecture unless needed for consistency.

## Proposed User Surface

CLI:

```bash
unfourier data.dat --rmax 150 --n-basis 24
unfourier data.dat --rmax 150 --knot-spacing 7.5 --min-basis 12 --max-basis 48
```

TOML:

```toml
[basis]
n_basis = 20
knot_spacing = 7.5
min_basis = 12
max_basis = 48

[constraints]
spline_boundary = "value_zero"       # value_zero | value_slope_zero
d1_smoothness = 0.1                  # -1 off, 0 default, >0 explicit
d2_smoothness = 1.0
```

Resolution rule:

```text
if n_basis is set by CLI or TOML:
    use n_basis
else if knot_spacing is set:
    n_basis = ceil(Dmax / knot_spacing), clamped to [min_basis, max_basis]
else:
    use n_basis = 20
```

## Epic 1: Spline-Only Product Surface

Remove the rectangular basis from the active source, command-line interface,
configuration, and docs.

### Issue 1.1: Remove basis selection from CLI

Tasks:

1. Delete `BasisChoice` from `src/main.rs`.
2. Remove the `--basis` argument.
3. Remove the `--npoints` argument.
4. Instantiate `CubicBSpline` unconditionally.
5. Keep `--n-basis` as the exact basis-size option.
6. Update verbose output to `basis: cubic-b-spline n_basis=...`.

Acceptance:

1. `unfourier data.dat --basis rect` fails as an unknown argument.
2. `unfourier data.dat --npoints 100` fails as an unknown argument.
3. `unfourier data.dat --n-basis 20` runs with a spline basis.
4. No `BasisChoice` references remain in `src/`.

### Issue 1.2: Delete `UniformGrid`

Tasks:

1. Remove `UniformGrid` from `src/basis.rs`.
2. Remove `UniformGrid` tests.
3. Replace any test fixture that used `UniformGrid` with `CubicBSpline`.
4. Update module-level docs in `src/basis.rs`.

Acceptance:

1. `rg -e "UniformGrid|rectangular|histogram" src` has no active hits.
2. `cargo test` compiles past the basis module.

### Issue 1.3: Clean TOML basis config

Tasks:

1. Remove `basis.type`.
2. Remove `basis.npoints`.
3. Add `basis.n_basis`.
4. Add `basis.knot_spacing`.
5. Add `basis.min_basis`.
6. Add `basis.max_basis`.
7. Update config parsing tests.

Acceptance:

1. Old `[basis] type = "rect"` config fails fast.
2. `[basis] n_basis = 24` parses.
3. `[basis] knot_spacing = 7.5` parses.
4. Unknown fields still fail because of `deny_unknown_fields`.

### Issue 1.4: Update active docs

Tasks:

1. Update `README.md`.
2. Update `CLAUDE.md`.
3. Update `pipeline.md`.
4. Keep historical docs in `docs/` unless they are actively misleading.

Acceptance:

1. Active docs do not mention `--basis rect|spline`.
2. Active docs do not recommend rectangular bins.
3. Active docs describe cubic B-splines as the only basis.

## Epic 2: Separate Coefficients From Evaluated P(r)

Make it impossible to accidentally publish spline control coefficients as if
they were sampled P(r).

### Issue 2.1: Make `Solution` explicit about coefficients

Tasks:

1. Add `coeffs: Vec<f64>` to `Solution`, or rename `p_r` internally to
   `coeffs`.
2. Keep `i_calc` computed from `coeffs`.
3. Decide whether `Solution` stores sampled P(r) or whether output evaluation
   happens outside `Solution`.
4. Update solver tests.

Acceptance:

1. It is clear in type/field names which vector is solved coefficients.
2. `i_calc` is always derived from the same coefficient vector.
3. No post-solve coefficient mutation is needed for output.

### Issue 2.2: Add spline output evaluation

Tasks:

1. Add `output_grid()` to the basis layer.
2. Add `evaluate_pr(coeffs, r)` to the basis layer.
3. Implement spline evaluation using `bspline::basis_matrix`.
4. Choose output-grid default: include `0`, `Dmax`, and enough interior points
   for a smooth plotted curve.

Acceptance:

1. Spline output table is evaluated P(r), not raw coefficients.
2. The output table includes `r = 0` and `r = Dmax`.
3. The first and last output P(r) values are zero for `value_zero`.

### Issue 2.3: Remove post-hoc boundary mutation

Tasks:

1. Delete the block in `main.rs` that inserts boundary rows into `solution.r`.
2. Delete hard zeroing of first/last interior P(r) values.
3. Delete matching posterior-error zeroing.
4. Recompute output via basis evaluation instead.

Acceptance:

1. `main.rs` does not mutate solved P(r)/coefficients after solving.
2. `solution.i_calc` and written P(r) are consistent with the same coefficients.
3. The old visible boundary cliff does not come from output editing.

### Issue 2.4: Add consistency tests

Tasks:

1. Add a test that deliberately checks coefficient count vs output-grid count.
2. Add a test that output endpoints are zero without mutating coefficients.
3. Add a test that `i_calc` is unchanged by output evaluation.

Acceptance:

1. Tests fail if raw coefficients are written as P(r).
2. Tests fail if post-hoc mutation returns.

## Epic 3: Spline Boundary Modes

Enforce endpoint constraints through the spline parameterisation.

### Issue 3.1: Add boundary mode config

Tasks:

1. Add `SplineBoundaryMode` enum.
2. Support `value_zero`.
3. Support `value_slope_zero`.
4. Parse `[constraints] spline_boundary`.
5. Default to `value_zero`.

Acceptance:

1. `value_zero` fixes only the endpoint coefficients.
2. `value_slope_zero` fixes the first two and last two full coefficients.
3. Invalid strings fail with a clear error.

### Issue 3.2: Implement free-to-full coefficient mapping

Tasks:

1. Build a mapping from free coefficients to full spline coefficients.
2. Use the same mapping for kernel construction.
3. Use the same mapping for output evaluation.
4. Expose helper methods only if tests need them.

Acceptance:

1. `value_zero` maps `[c...]` to `[0, c..., 0]`.
2. `value_slope_zero` maps `[c...]` to `[0, 0, c..., 0, 0]`.
3. Mapping logic is not duplicated in kernel and output code.

### Issue 3.3: Project kernel through mapping

Tasks:

1. Build full spline kernel with all B-spline columns.
2. Project to free columns through the coefficient mapping.
3. Remove pseudo-data boundary rows from the normal solve path.

Acceptance:

1. Kernel columns match the free coefficients.
2. Boundary coefficients cannot accidentally contribute to I(q).
3. No `boundary_weight` is needed for hard boundary values.

### Issue 3.4: Project regularisation through mapping

Tasks:

1. Build derivative penalty in full coefficient space.
2. Project with `L_free = L_full * B`.
3. Use projected `L_free^T L_free` in manual and automatic lambda paths.
4. Retire `BoundaryAnchoredCombined` after projected regularisation works.

Acceptance:

1. Kernel, regulariser, and output all agree on free coefficients.
2. The regulariser sees the fixed zero boundaries.
3. No ad hoc post-hoc boundary anchoring remains.

### Issue 3.5: Boundary validation

Tasks:

1. Add unit tests for endpoint values.
2. Add a small numerical derivative check for `value_slope_zero`.
3. Add regression output checks on Debye data.

Acceptance:

1. `value_zero`: P(0) and P(Dmax) are zero.
2. `value_slope_zero`: endpoint slopes are near zero on a dense grid.
3. `value_zero` does not create a small-r zero plateau by construction.

## Epic 4: Smoothness Regularisation

Add a neighboring-coefficient penalty and keep curvature regularisation.

### Issue 4.1: Combined D1 and D2 regulariser

Tasks:

1. Keep or refactor `FirstDerivative`.
2. Keep or refactor `SecondDerivative`.
3. Use a combined penalty:

   ```text
   d1 * ||D1 c||^2 + d2 * ||D2 c||^2
   ```

4. Apply it in the projected full-coefficient space from Epic 3.

Acceptance:

1. `d1_smoothness = -1` disables the D1 contribution.
2. `d1_smoothness = 0` uses the default D1 weight.
3. Positive `d1_smoothness` uses the supplied weight.

### Issue 4.2: Add D2 config

Tasks:

1. Add `d2_smoothness`.
2. Default it to `1.0`.
3. Allow advanced users to tune it.
4. Keep lambda selection scaling stable.

Acceptance:

1. Existing curvature behaviour is recoverable.
2. D1 and D2 weights are visible in verbose output.
3. Changing D1 does not silently disable D2.

### Issue 4.3: Explore defaults

Tasks:

1. Sweep `d1_smoothness` on Debye and real datasets.
2. Compare small-r behaviour.
3. Compare Dmax boundary taper.
4. Compare chi-squared and P(r) stability.

Acceptance:

1. Proposed default D1 weight is documented.
2. `value_zero` remains the default unless validation strongly supports
   `value_slope_zero`.

## Epic 5: Knot Density Controls

Expose basis complexity in a way that scales with Dmax.

### Issue 5.1: Add CLI options

Tasks:

1. Add `--knot-spacing`.
2. Add `--min-basis`.
3. Add `--max-basis`.
4. Validate positive values.
5. Validate `min_basis <= max_basis`.

Acceptance:

1. `--n-basis` wins when supplied.
2. `--knot-spacing 7.5` computes `ceil(Dmax / 7.5)` and clamps it.
3. Verbose output shows whether `n_basis` was explicit or derived.

### Issue 5.2: Add TOML options

Tasks:

1. Add `[basis] knot_spacing`.
2. Add `[basis] min_basis`.
3. Add `[basis] max_basis`.
4. Apply CLI precedence over TOML.

Acceptance:

1. TOML-only knot spacing works.
2. CLI `--n-basis` overrides TOML knot spacing.
3. Config tests cover explicit and derived basis counts.

### Issue 5.3: Knot-density sweep

Tasks:

1. Add or update a validation script to sweep `n_basis`.
2. Add or update a validation script to sweep `knot_spacing`.
3. Record chi-squared, endpoint behaviour, and P(r) roughness.

Acceptance:

1. There is at least one command that compares `n_basis = 12, 16, 20, 28, 36`.
2. There is at least one command that compares `knot_spacing = 5.0, 7.5, 10.0`.
3. Results are easy to inspect visually.

### Issue 5.4: Defer two-pass Dmax refinement

Tasks:

1. Document two-pass Dmax refinement as future work.
2. Do not implement it in 0.9.
3. Note open questions: tail threshold, noise sensitivity, lambda reselection,
   and whether Dmax may grow after the first pass.

Acceptance:

1. 0.9 has no hidden second solve.
2. Future work is described clearly enough to revisit later.

## Epic 6: Validation and Developer Scripts

Make validation match the spline-only implementation.

### Issue 6.1: Rewrite `Dev/validate_real_data.py`

Tasks:

1. Remove rect runs.
2. Remove rect basis-function row from plots.
3. Keep spline comparison against GNOM/reference where available.
4. Add endpoint diagnostics.
5. Add optional rebin variants where useful.

Acceptance:

1. Script runs without `--basis`.
2. Output table no longer reports rect pass counts.
3. Endpoint diagnostics are visible.

### Issue 6.2: Retire or rewrite `Dev/validate_m5.py`

Tasks:

1. Remove spline-vs-rect comparison checks.
2. Keep kernel sanity checks if useful.
3. Add spline-only regression checks.

Acceptance:

1. No check depends on a removed rectangular basis.
2. Script name or header no longer implies M5 comparison work.

### Issue 6.3: Update `Dev/sweep_noise.py`

Tasks:

1. Make spline the only active basis.
2. Remove rect labels from plots.
3. Include `n_basis` and `d1_smoothness` sweeps.

Acceptance:

1. Noise sweep runs with the 0.9 CLI.
2. Plots make spline parameter choices clear.

### Issue 6.4: Decide fate of `Dev/monte_carlo_coverage.py`

Tasks:

1. Identify rectangular-kernel assumptions.
2. Either rewrite for spline output or mark as temporarily retired.
3. Document the decision.

Acceptance:

1. The script does not silently validate the wrong model.
2. Bayesian coverage status is clear in docs.

## Epic 7: 0.9 Documentation and Cleanup

Make active docs match the implementation.

### Issue 7.1: Update active docs

Tasks:

1. `README.md`: spline-only usage, knot spacing, boundary modes.
2. `CLAUDE.md`: current architecture and commands.
3. `pipeline.md`: remove rect stage and boundary pseudo-data stage.
4. `CODEX.md`: keep strategy in sync if implementation diverges.

Acceptance:

1. A new reader sees only the spline workflow.
2. Commands in docs run against the 0.9 CLI.

### Issue 7.2: Add cleanup search checks

Tasks:

1. Add final manual cleanup commands to the spec or README.
2. Run:

   ```bash
   rg -e "UniformGrid|BasisChoice|--basis|--npoints|\\brect\\b|rectangular" src README.md CLAUDE.md pipeline.md Dev
   rg -e "append_boundary_constraints|boundary_weight" src README.md CLAUDE.md pipeline.md
   ```

3. Triage any remaining hits as active or historical.

Acceptance:

1. No active source references removed concepts.
2. Historical references are clearly historical.

### Issue 7.3: Final verification

Tasks:

1. Run `cargo test`.
2. Run `cargo build --release`.
3. Generate or reuse Debye data.
4. Run spline validation on Debye.
5. Run real-data validation.
6. Save representative plots if desired.

Acceptance:

1. Tests pass.
2. Release build succeeds.
3. Spline P(r) reaches zero endpoints without hard output clamps.
4. Small-r P(r) is not artificially zero unless the data and settings imply it.
5. I(q) fit remains reasonable.

## Suggested Implementation Order

1. Epic 1: remove the rectangular product surface.
2. Epic 5, issues 5.1 and 5.2: add knot-density controls early.
3. Epic 2: separate coefficients from evaluated P(r).
4. Epic 3: implement boundary modes and coefficient mapping.
5. Epic 4: wire projected D1/D2 regularisation.
6. Epic 6: update validation scripts.
7. Epic 7: update docs and run final checks.

## Open Questions

1. Should `value_slope_zero` remain an expert-only option forever, or become the
   default if validation shows no small-r suppression?
2. What dense output-grid size is enough for smooth plots without huge files?
3. Should output include both sampled P(r) and raw coefficients in a debug mode?
4. Should knot spacing become the default over fixed `n_basis` after validation?
5. How should Dmax be inferred for a future two-pass solve without chasing noise
   tails?
