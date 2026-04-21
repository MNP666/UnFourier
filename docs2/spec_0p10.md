# unFourier 0.10 Spec

This spec starts the next iteration after the spline-focused 0.9 work. Version
0.10 is allowed to expose experimental controls. Nothing in this spec promises
that automatic preprocessing choices are final or generally reliable before 1.0.

The main new feature is a Guinier preflight scan for low-q data quality. The
scanner should repeatedly fit the Guinier relation while discarding increasing
numbers of initial points, then report where `Rg` and `I0` become stable. It
should support both a reporting-only mode and an opt-in mutation mode that uses
the suggested low-q cutoff as the effective `qmin`.

## Implementation Status

Epic 1, Epic 2, Epic 3, Epic 4, and Epic 5 are implemented. The remaining 0.10
work starts at Epic 6: user-facing and developer documentation.

## Goals

1. Add a small, testable Guinier scan routine for low-q quality checks.
2. Report a table of candidate Guinier fits and a suggested `qmin`.
3. Add an opt-in mode that applies the suggested `qmin` to preprocessing.
4. Keep explicit user settings higher priority than automatic suggestions.
5. Make real-data validation show the effect of report-only and applied modes.
6. Keep the feature transparent, experimental, and easy to disable.

## Non-Goals

1. No claim that automatic Guinier trimming is universally correct.
2. No default automatic mutation of user data in 0.10.
3. No two-pass Dmax refinement beyond the existing `rmax = pi / qmin` behavior.
4. No replacement for manual Guinier inspection.
5. No support for signed or contrast-variation P(r) in this iteration.
6. No broad ATSAS/GNOM parser compatibility work.

## Reporting vs Mutation

Most of the work is in reporting, not mutation. The scanner, fit statistics,
stability rule, tests, and human-readable output are required for both modes.
Once the scanner returns a recommendation, mutation is mostly pipeline plumbing:
if auto mode is enabled and the user did not set `qmin`, set the effective
`qmin` before the existing `QRangeSelector` runs.

The extra work for mutation is not many lines of math, but it does need clear
precedence rules and careful verbose output so users can see exactly what was
discarded.

## Proposed User Surface

CLI:

```bash
# Report only. Does not change the fit.
unfourier data.dat --guinier-report --rmax 150

# Report and apply suggested qmin, unless --qmin is also supplied.
unfourier data.dat --auto-qmin guinier --guinier-report --rmax 150

# Explicit qmin wins over auto-qmin. The report may still be printed.
unfourier data.dat --qmin 0.006 --auto-qmin guinier --guinier-report --rmax 150
```

TOML:

```toml
[preprocessing]
auto_qmin = "off"          # off | guinier

[guinier]
report = false
min_points = 8
max_points = 25
max_skip = 8
max_qrg = 1.3
stability_windows = 3
rg_tolerance = 0.02
i0_tolerance = 0.03
max_chi2 = 3.0
```

Resolution rules:

```text
if --qmin is supplied:
    use --qmin
    if auto_qmin is enabled, still allow report but do not mutate qmin
else if auto_qmin = guinier:
    run Guinier scan
    if recommendation exists:
        use recommended qmin
    else:
        leave qmin unset and warn/report
else:
    leave qmin unset
```

The `--guinier-report` flag should print the report even when `--verbose` is not
set. In verbose mode, the report should be included automatically when
`--auto-qmin guinier` is active.

## Mathematical Model

Use the standard Guinier approximation:

```text
I(q) = I0 * exp(-(Rg^2 * q^2) / 3)
```

Linearized:

```text
y = a + b*x
x = q^2
y = ln(I)
a = ln(I0)
b = -Rg^2 / 3
```

Therefore:

```text
Rg = sqrt(-3*b)
I0 = exp(a)
```

Use weighted least squares with:

```text
sigma_y ~= sigma_I / I
weight = 1 / sigma_y^2
```

Reject candidate windows with non-positive intensities, non-finite values,
positive fitted slope, too few points, `qmax * Rg > max_qrg`, or poor reduced
chi-squared.

## Epic 1: Guinier Scan Core - Done

Implement the analysis as a small reusable routine with no side effects.

### Issue 1.1: Add result types

Tasks:

1. Add a `guinier` module or a focused section in `preprocess.rs`.
2. Add `GuinierScanConfig`.
3. Add `GuinierWindowFit`.
4. Add `GuinierRecommendation`.
5. Add `GuinierScanReport`.
6. Keep types independent of CLI parsing.

Suggested fields:

```text
GuinierWindowFit:
    skip
    n_points
    q_min
    q_max
    qrg_max
    rg
    i0
    slope
    intercept
    chi2_red
    valid
    reject_reason

GuinierRecommendation:
    skip
    q_min
    rg
    i0
    chi2_red
    confidence_label
```

Acceptance:

1. A scan can be run from tests without invoking the CLI.
2. Rejected windows keep enough information for reporting.
3. The recommendation can be absent without causing the solve to fail.

### Issue 1.2: Weighted linear Guinier fit

Tasks:

1. Implement weighted linear regression for `ln(I)` vs `q^2`.
2. Propagate input sigma to log-space sigma.
3. Compute `Rg`, `I0`, `qmax * Rg`, and reduced chi-squared.
4. Return explicit rejection reasons for invalid windows.

Acceptance:

1. Synthetic Guinier data recovers known `Rg` and `I0`.
2. A positive slope is rejected.
3. Non-positive intensities are rejected for the affected window.
4. Zero or invalid sigma values do not panic.

### Issue 1.3: Generate candidate windows

Tasks:

1. For `skip = 0..max_skip`, try prefix windows after skipping initial points.
2. For each skip, try `n_points = min_points..max_points` where possible.
3. Prefer the largest valid Guinier-range window for each skip.
4. Keep all attempted windows for reporting.

Acceptance:

1. The scanner explores increasing initial truncation.
2. A noisy first point can be skipped while later windows remain valid.
3. The scanner never uses high-q points that violate `qmax * Rg <= max_qrg`.

### Issue 1.4: Stability recommendation

Tasks:

1. Look for the earliest skip where the next `stability_windows` valid fits are
   stable.
2. Treat `Rg` as stable when relative changes are below `rg_tolerance`.
3. Treat `I0` as stable when relative changes are below `i0_tolerance`.
4. Prefer lower skip counts when several plateaus satisfy the rule.
5. Return no recommendation if no stable plateau is found.

Acceptance:

1. A stable synthetic series recommends the first point.
2. A corrupted first point recommends a later `qmin`.
3. A chaotic low-q series reports no recommendation rather than guessing.

## Epic 2: CLI and TOML Integration - Done

Expose the scanner without making it a default hidden preprocessing step.

### Issue 2.1: Add CLI flags

Tasks:

1. Add `--guinier-report`.
2. Add `--auto-qmin <off|guinier>`.
3. Add advanced tuning flags only if they are needed during validation.
4. Document that `--qmin` takes precedence over `--auto-qmin`.

Acceptance:

1. `--guinier-report` prints a report and leaves the fit unchanged.
2. `--auto-qmin guinier` applies the recommendation when available.
3. `--qmin` prevents auto mutation and prints a clear note if a report exists.
4. Invalid `--auto-qmin` values fail through clap.

### Issue 2.2: Add TOML config

Tasks:

1. Add `preprocessing.auto_qmin`.
2. Add `[guinier]` config.
3. Preserve CLI-over-TOML precedence.
4. Keep `deny_unknown_fields` behavior.

Acceptance:

1. TOML-only report/apply settings work.
2. CLI `--auto-qmin off` can disable TOML auto mode.
3. Unknown Guinier fields fail fast.

### Issue 2.3: Place mutation in the preprocessing pipeline

Tasks:

1. Run the scan after negative-intensity handling.
2. Run the scan before `QRangeSelector`.
3. If mutation is enabled, set the effective `qmin` before `QRangeSelector`.
4. Keep explicit `qmin` untouched.
5. Ensure automatic `rmax = pi / qmin` sees the effective post-Guinier q range.

Acceptance:

1. Applied auto-qmin removes the intended low-q points.
2. Report-only mode leaves point counts unchanged.
3. Explicit `--rmax` is not changed by the recommendation.
4. Automatic `rmax` follows the actual processed data after q filtering.

## Epic 3: Reporting - Done

Make the scan inspectable enough that users can decide whether to trust it.

### Issue 3.1: Human-readable report

Tasks:

1. Print a compact table of accepted and rejected windows.
2. Include `skip`, `qmin`, `n`, `qmax*Rg`, `Rg`, `I0`, `chi2_red`, and status.
3. Print the final recommendation or "no stable recommendation".
4. State whether the recommendation was applied or report-only.

Example:

```text
Guinier scan:
skip  qmin       n   qmax*Rg   Rg      I0       chi2   status
0     0.0030    18     1.24    42.8    101.2    6.8    reject: chi2
1     0.0038    17     1.21    41.3     99.5    2.1    valid
2     0.0046    16     1.18    41.1     99.1    1.3    stable

Guinier recommendation: qmin = 0.0046 (skip 2), report-only
```

Acceptance:

1. The report is understandable without reading source code.
2. Rejection reasons are visible.
3. Applied mode names the old and new effective q range.

### Issue 3.2: Optional machine-readable report

Tasks:

1. Decide whether a JSON report is worth adding in 0.10.
2. If yes, add `--guinier-report-json <path>`.
3. If no, document it as future work.

Current decision: defer JSON output for now. Epic 3 is complete with the text
report as the supported 0.10 reporting surface; validation scripts may parse the
human-readable report during the 0.10 iteration. Add
`--guinier-report-json <path>` later if text parsing becomes fragile or if
external callers need a supported machine-readable format.

Acceptance:

1. Validation scripts can still parse text if JSON is deferred.
2. No unstable file format is advertised as permanent.

## Epic 4: Validation Script Support - Done

Use the real-data validation workflow to exercise both report-only and applied
auto-qmin.

Current implementation: `Dev/validate_real_data.py` accepts repeatable
`--guinier-mode off|report|apply` switches. Report/apply modes write captured
Guinier report snippets to the validation output directory. When both `off` and
`report` are requested, the script verifies that report-only mode leaves P(r)
unchanged. When both `report` and `apply` are requested, it compares point
counts, ISE, Rg agreement, chi-squared, and classifies the applied mode as
helped, hurt, mixed, or did nothing.

### Issue 4.1: Add validation switches

Tasks:

1. Add a `Dev/validate_real_data.py` option for Guinier mode:

   ```text
   off | report | apply
   ```

2. Pass `--guinier-report` for report/apply modes.
3. Pass `--auto-qmin guinier` for apply mode.
4. Store stdout/stderr snippets per dataset if useful.

Acceptance:

1. The validation script can run with no Guinier scan.
2. The validation script can run with report-only scans.
3. The validation script can run with applied auto-qmin.

### Issue 4.2: Compare applied vs report-only outcomes

Tasks:

1. Record the suggested/applied `qmin`.
2. Record point counts before and after preprocessing when available.
3. Compare ISE, Rg agreement, chi-squared, and I(q) fit.
4. Highlight datasets where auto-qmin changes the result materially.

Acceptance:

1. The validation table shows whether auto-qmin helped, hurt, or did nothing.
2. The validation plot remains readable with five datasets.
3. Report-only mode has identical P(r) output to off mode.

## Epic 5: Tests and Guardrails

Keep the experimental feature from silently doing surprising things.

Current implementation: scanner unit tests cover synthetic recovery, corrupted
low-q points, rejected windows, and no-recommendation behavior. Binary unit
tests cover report formatting and qmin precedence. CLI integration tests in
`tests/guinier_cli.rs` cover report-only equivalence, applied auto-qmin,
explicit qmin precedence, and CLI override of TOML `auto_qmin`. Real-data
validation results are recorded in `docs2/epic5_validation.md`.

### Issue 5.1: Unit tests

Tasks:

1. Test weighted linear fit on noiseless synthetic Guinier data.
2. Test noisy first-point detection.
3. Test no-recommendation behavior.
4. Test explicit `qmin` precedence.
5. Test TOML parsing for the new sections.

Acceptance:

1. `cargo test` covers the recommendation logic.
2. Tests do not depend on real-data fixtures.
3. Failure messages identify the relevant window or rule.

### Issue 5.2: Integration checks

Tasks:

1. Run real-data validation with Guinier off.
2. Run real-data validation with report-only.
3. Run real-data validation with applied auto-qmin.
4. Compare generated fit plots.

Acceptance:

1. Report-only and off modes produce the same P(r) files.
2. Applied mode changes only datasets with nonzero recommended skip.
3. Any improvement/regression is documented in the iteration notes.

## Epic 6: Documentation

Document the feature as an experimental assistant, not an oracle.

### Issue 6.1: README and pipeline docs

Tasks:

1. Add a short "Guinier preflight" section to `README.md`.
2. Explain report-only vs applied mode.
3. Explain that explicit `qmin` wins.
4. Add the pipeline placement to `pipeline.md`.
5. Note that the feature is experimental until 1.0.

Acceptance:

1. Users can discover the report mode without reading source.
2. Users are warned not to trust auto-qmin blindly.
3. The command examples run.

### Issue 6.2: Developer docs

Tasks:

1. Update `CLAUDE.md` / `AGENTS.md` if the workflow changes.
2. Add validation commands for report and apply modes.
3. Record known failure modes.

Acceptance:

1. Future development agents know where the scan lives.
2. Validation expectations are clear.

## Suggested Implementation Order

1. Done: Epic 1, implement scanner and tests without CLI integration.
2. Done: Epic 3.1, add text report formatting.
3. Done: Epic 2.1, add CLI report-only mode.
4. Done: Epic 2.3, wire opt-in mutation through existing `QRangeSelector`.
5. Done: Epic 2.2, add TOML config.
6. Done: Epic 4, update validation scripts.
7. Done: Epic 5, add guardrail tests and run off/report/apply validation.
8. Epic 6: update active docs.

## Open Questions

1. Should `max_chi2` be a hard rejection or only part of confidence labeling?
2. Should the stabilization rule use consecutive skip windows or any local
   plateau in sorted skip order?
3. Should report-only mode print by default when `--verbose` is set, or only when
   explicitly requested?
4. Should applied auto-qmin be available from TOML, or should mutation require an
   explicit CLI flag during 0.10?
5. Should validation compare auto-qmin against GNOM's chosen low-q range where
   that metadata is available?
