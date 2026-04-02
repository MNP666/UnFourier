# M4 — Bayesian IFT: Task Breakdown

> **Test fixture:** All noisy validation uses the Debye/Gaussian chain form factor
> (`Dev/debye_k5.dat` etc.), not sphere data. The sphere form factor has true zeros
> where σ = I/k → 0, causing the weight matrix to diverge and making any finite λ
> ineffective. See `saxs_ift_postmortem.md` for the full analysis.
> Noiseless sphere is acceptable only for kernel-correctness checks.

## Issue 1: Bayesian evidence function

Implement log-evidence calculation for a given λ:

```
log P(I|λ) = -½[ χ² + λ‖Lc‖² + log det(KᵀK + λLᵀL) − N·log λ ]
```

- Reuse existing SVD decomposition from `TikhonovSolver` — singular values give `log det` cheaply
- Add a `BayesianEvidence` struct that wraps the evidence calculation for a single `LambdaEvaluation`
- Unit-test against a known analytic case or a Python/scipy reference

---

## Issue 2: `BayesianEvidence` as a `LambdaSelector`

Implement `LambdaSelector` for `BayesianEvidence`:

- Evaluate over the same log-spaced λ grid used by GCV/L-curve (reuse existing grid generation)
- Select λ that maximises log-evidence
- Add `--method bayes` to the CLI (alongside `gcv`, `lcurve`, `manual`)
- Validate that selected λ is comparable to GCV/L-curve on Debye data

---

## Issue 3: Posterior covariance and P(r) error bars

Compute uncertainty on the solution coefficients:

- Posterior covariance: `Σ = (KᵀK + λLᵀL)⁻¹`
- Marginal standard deviations: `σ_c = sqrt(diag(Σ))`
- Propagate to P(r) error bars (one σ per r-grid point)
- Add a third column to the output: `r  P(r)  σ_P(r)`
- Update `output.rs` to handle the optional error-bar column

---

## Issue 4: Monte Carlo calibration script (`Dev/monte_carlo_coverage.py`)

Python validation for statistical correctness of error bars:

- Generate many noise realisations of Debye data at a fixed noise level
- Run `unfourier --method bayes` on each
- Check that the true P(r) falls within ±1σ ~68% of the time at each r point
- Plot coverage fraction vs r; flag r regions where bars are under- or over-confident
