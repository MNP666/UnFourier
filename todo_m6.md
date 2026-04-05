# M6 — Preprocessing Pipeline and Real-World Data: Implementation Todo

## Overview

M6 brings unFourier from synthetic-only to real experimental SAXS data.
The three core blockers for real data are:

1. **Fragile parsing** — real files have varied headers and may not have strictly
   monotone q values.
2. **Negative intensities** — background subtraction routinely produces I ≤ 0 at
   high q; these must not be naively down-weighted via 1/σ², which can diverge.
3. **No q-range control** — real data needs trimming of the noisy high-q tail.

Reference data for validation lives in:
- `data/dat_ref/` — SASDME2.dat, SASDF42.dat, SASDYU3.dat (1696 pts)
- `data/prs_ref/` — SASDME2.out, SASDF42.out, SASDYU3.out (GNOM reference P(r))

---

## Design decisions

### ClipNegative
- Replace I ≤ 0 with `min_positive_I` (smallest positive I in the dataset).
- Inflate the corresponding σ by a large factor (default 1000 × max(σ)),
  making the weight 1/σ² negligible in the fit.
- Do **not** use NaN — causes downstream arithmetic failures.
- Alternative: omit the point entirely (`OmitNonPositive`).
- Emit a verbose warning counting clipped points.

### `.dat` parser
- Expect three columns that parse to floats; skip non-float lines silently
  (handles all header styles seen: bare text, `#`-prefixed, `###`-prefixed).
- **New:** after collecting all points, verify first column is strictly
  increasing; error with the first offending line number if not.
- Leading whitespace and any whitespace delimiter already handled via
  `split_whitespace`. CRLF handled by `std::str::lines()`.

### `.out` parser (GNOM reference files)
- Find the line containing `"Distance distribution"` — this marks the P(r)
  section in both GNOM v4.5 and v5.0.
- After it, collect lines that parse as three floats (R, P(R), ERROR) using
  the same logic as the `.dat` parser; stop at the first non-numeric line.
- Also extract Rg, I(0), D_max from the Results section (before the data
  sections). Two format variants observed:
  - GNOM v5.0: `Real space Rg: 0.XXXXE+02 +- ...`
  - GNOM v4.5: `Real space: Rg = XX.XX +- ...`

---

## Todo list

### 1 — Improve `.dat` parser (`src/data.rs`)

- [ ] Track the source line number for each accepted data row during parsing.
- [ ] After collecting all numeric rows, check `q[i] < q[i+1]` for all i;
      emit an error that names the first offending line number if violated.
- [ ] Unit test: a `.dat` string with non-monotone q returns `Err` containing
      the offending line number.
- [ ] Unit test: leading whitespace and tab delimiters parse correctly.
- [ ] Unit test: CRLF line endings (`\r\n`) parse correctly.

Design note: retain the existing "silently skip non-float lines" behaviour —
it cleanly handles all three header styles in the reference `.dat` files.

---

### 2 — `ClipNegative` preprocessor (`src/preprocess.rs`)

- [ ] Add `pub struct ClipNegative { pub sigma_inflate_factor: f64 }` with
      `ClipNegative::default()` → `sigma_inflate_factor = 1000.0`.
- [ ] Implement `Preprocessor`:
  - Compute `i_floor = min(I[i] for I[i] > 0)`.
    Return `Err` if no positive intensities exist.
  - Compute `sigma_ceil = sigma_inflate_factor × max(σ[i])`.
  - For each point where `I[i] ≤ 0`: set `I[i] = i_floor`, `σ[i] = sigma_ceil`.
  - `name()` → `"clip-negative"`.
- [ ] Unit test: a 5-point dataset with 2 negative intensities produces correct
      `i_floor` replacement and inflated sigmas.
- [ ] The inflated sigma ensures the fit weight `1/σ²` is a factor
      `sigma_inflate_factor²` below the maximum — effectively zero contribution.

Omit alternative:
- [ ] Add `pub struct OmitNonPositive;` implementing `Preprocessor` — filters
      out all rows where `I ≤ 0`. Simpler but loses q-coverage.
      `name()` → `"omit-non-positive"`.
- [ ] Unit test: filtered dataset contains only positive-intensity points.

---

### 3 — `QRangeSelector` preprocessor (`src/preprocess.rs`)

- [ ] Add:
  ```rust
  pub struct QRangeSelector {
      pub q_min: Option<f64>,
      pub q_max: Option<f64>,
      pub snr_threshold: Option<f64>,
  }
  ```
- [ ] Implement `Preprocessor`:
  - If `q_min = Some(v)`: discard points with `q < v`.
  - If `q_max = Some(v)`: discard points with `q > v`.
  - If `snr_threshold = Some(t)`: additionally discard points where
    `I/σ < t`, working from the high-q end inward (trim the noisy tail).
  - Return `Err` if no points remain after filtering.
  - `name()` → `"q-range-selector"`.
- [ ] Unit test: manual q-range returns the expected subset.
- [ ] Unit test: SNR threshold trims the correct tail points.

Guinier q_min (stretch goal — defer to M6+ if time is short):
- [ ] `GuinierQmin` struct: fits `ln I` vs `q²` over the lowest-q region,
      finds the upper limit of linearity satisfying `q·Rg < 1.3`.
      Return the estimated `q_min`.

---

### 4 — `LogRebin` preprocessor (`src/preprocess.rs`)

Priority: **lower** — the solver runs on 1696 points (SASDYU3) without
rebinning, just more slowly. Needed for practical performance on large datasets.

- [ ] Add `pub struct LogRebin { pub n_bins: usize }`.
- [ ] Implement `Preprocessor`:
  - Create `n_bins` log-spaced q bin edges from `q[0]` to `q[-1]`.
  - For each non-empty bin: `I_new = mean(I[j])`,
    `σ_new = sqrt(Σσ[j]²) / n` (standard error of mean).
  - Discard empty bins.
  - `name()` → `"log-rebin"`.
- [ ] Unit test: rebinning a 100-point dataset to 20 bins produces the
      correct weighted averages.

---

### 5 — CLI integration (`src/main.rs`)

- [ ] Add `--negative-handling clip|omit|keep` (default: `clip`).
- [ ] Add `--qmin FLOAT` and `--qmax FLOAT`.
- [ ] Add `--snr-cutoff FLOAT` (default: `0.0` = disabled).
- [ ] Add `--rebin N` (default: `0` = disabled).
- [ ] Build the preprocessing pipeline in `main()` based on flags, in order:

      ```
      [ClipNegative | OmitNonPositive | (skip if keep)]
        → [QRangeSelector if any of --qmin/--qmax/--snr-cutoff is set]
        → [LogRebin if --rebin > 0]
      ```

      Replacing the current `Identity`-only pipeline.

- [ ] Verbose output (`--verbose`): for each preprocessor step, print to
      stderr the number of points modified and the resulting q range and
      point count.

---

### 6 — GNOM `.out` reference parser (`Dev/parse_gnom.py`)

- [ ] Implement `parse_gnom_out(path: str) -> dict` returning:
  - `"pr"`:    `np.ndarray` shape `(N, 3)` — columns R, P(R), ERROR
  - `"rg"`:    `float`
  - `"i0"`:    `float`
  - `"d_max"`: `float`

- [ ] **P(r) section**:
  1. Scan lines for the substring `"Distance distribution"`.
  2. Collect subsequent lines that parse as three floats (R, P(R), ERROR).
  3. Stop at the first non-parseable line once data has started.

- [ ] **Metadata** (Results section, appears before the data tables):
  - D_max: match `"Real space range"` → extract the upper bound, or
    match `"Maximum characteristic size"`.
  - Rg: match `"Real space Rg:"` (v5.0) or `"Real space: Rg ="` (v4.5).
  - I(0): match `"Real space I(0):"` (v5.0) or `"Real space: ... I(0) ="` (v4.5).

- [ ] Validate on all three `.out` files; print a summary table showing
      the parsed Rg, I(0), D_max, and number of P(r) points.

---

### 7 — Validation (`Dev/validate_real_data.py`)

- [ ] For each dataset (SASDME2, SASDF42, SASDYU3):
  - Parse `d_max` and `rg` from the GNOM `.out` file via `parse_gnom_out`.
  - Run:
    ```
    unfourier data/dat_ref/{name}.dat \
        --rmax {d_max} \
        --negative-handling clip \
        --method gcv \
        --basis rect \
        --output /tmp/{name}_pr.dat \
        --verbose
    ```
  - Load the GNOM reference P(r).
  - Compute ISE between unfourier P(r) and GNOM P(r) (normalised to peak).
  - Estimate Rg from the unfourier P(r):
    `Rg² = ∫ r² P(r) dr / (2 ∫ P(r) dr)`.
  - Report: ISE, `|Rg_unfourier − Rg_gnom| / Rg_gnom`.

- [ ] Pass criteria (suggested):
  - ISE < 0.15 (shapes broadly agree; exact match not expected given different
    regularisation strategies)
  - Rg within 15% of GNOM reference

- [ ] Also run with `--basis spline` and display both results side by side.

- [ ] For SASDYU3 (1696 pts), also run with `--rebin 200` and compare runtime
      and ISE.

---

## Ordering rationale

| Task | Priority | Why |
|------|----------|-----|
| 1 — parser | high | Blocks loading real data if q is non-monotone |
| 2 — ClipNegative | high | Critical for any background-subtracted SAXS file |
| 3 — QRangeSelector | high | Needed to match GNOM q-range for fair comparison |
| 4 — LogRebin | medium | Quality-of-life for large datasets; solver works without |
| 5 — CLI | high | Exposes tasks 1–4 to the user |
| 6 — GNOM parser | medium | Required for validation |
| 7 — Validation | medium | Definition of done |

---

## Definition of done

```bash
# Build
cargo build --release

# Parser and preprocessor unit tests
cargo test

# Run on all three real datasets
./target/release/unfourier data/dat_ref/SASDME2.dat \
    --rmax 45.19 --negative-handling clip --verbose
./target/release/unfourier data/dat_ref/SASDF42.dat \
    --rmax 139.20 --negative-handling clip --verbose
./target/release/unfourier data/dat_ref/SASDYU3.dat \
    --rmax 80.0 --negative-handling clip --verbose

# Full comparison vs GNOM reference
python Dev/validate_real_data.py
```

Expected outcomes:
- All three runs complete without error.
- Rg within 15% of GNOM for all three datasets.
- ISE < 0.15 for at least two of three datasets.
