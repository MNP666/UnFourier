# Epic 5 Validation Notes

Date: 2026-04-21

Scope: Guinier preflight tests and guardrails for `docs2/spec_0p10.md`.

## Unit and CLI Guardrails

Command:

```bash
cargo test
```

Result:

```text
library tests:      56 passed
binary unit tests:  31 passed
CLI integration:     4 passed
doctests:            0 passed, 2 ignored
```

The CLI integration tests in `tests/guinier_cli.rs` cover:

1. `--guinier-report` leaves `P(r)` output unchanged.
2. `--auto-qmin guinier` applies the skip-1 recommendation for a synthetic
   corrupted first point.
3. Explicit `--qmin` prevents the auto recommendation from mutating qmin.
4. CLI `--auto-qmin off` overrides `preprocessing.auto_qmin = "guinier"` from
   `unfourier.toml`.

## Real-Data Integration

Command:

```bash
python3 Dev/validate_real_data.py --guinier-mode off --guinier-mode report --guinier-mode apply
```

Result: PASS.

Primary spline criteria:

```text
ISE: 15/15 primary runs passed
Rg:  PASS
Overall: PASS
```

Report-only equivalence:

```text
All off/report comparisons had max |dP| = 0.
```

Applied auto-qmin effects:

| Dataset | Variant | Recommended skip | Applied qmin | Fit point change | Impact |
|---------|---------|------------------|--------------|------------------|--------|
| SASDF42 | spline | 0 | 8.0434e-03 | 0 | did nothing |
| SASDYT6 | spline | 5 | 3.3718e-02 | -5 | helped |
| SASDYT6 | spline-rebin | 5 | 3.3718e-02 | +9 | hurt |

All other dataset/variant pairs had no stable recommendation and were unchanged.
The SASDYT6 rebin regression is small in absolute ISE (`+0.0002491`) but is a
useful reminder that applied mode is experimental and should remain opt-in.
