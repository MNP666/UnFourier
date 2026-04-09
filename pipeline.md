# unFourier Pipeline

This document describes every stage of the IFT pipeline — what it computes, why,
and the mathematics behind it.  The implementation follows the same order:

```
parse → preprocess → basis → build system → [constrain] → λ selection → solve → non-negativity → output
```

---

## 1  Parse

**Module:** `src/data.rs`  
**Struct:** `SaxsData { q: Vec<f64>, intensity: Vec<f64>, error: Vec<f64> }`

The input is a whitespace-delimited `.dat` file with three columns:

```
q_i   I(q_i)   σ(q_i)
```

Lines beginning with `#` are comments. The parser validates that q is strictly
increasing and all σ > 0 (after any preprocessing).

---

## 2  Preprocess

**Module:** `src/preprocess.rs`  
**Trait:** `Preprocessor`

Each step consumes a `SaxsData` and returns a transformed one.  Steps run in
order; each is a no-op if disabled.

### 2a  Negative-intensity handling

Background subtraction can produce I(q) ≤ 0. Three strategies are available:

| Mode | Action |
|------|--------|
| `clip` (default) | Replace I ≤ 0 with `i_floor = min(I > 0)` and set `σ = 1000 × max(σ)`, making the point effectively invisible to the solver. |
| `omit` | Discard points with I ≤ 0 entirely. |
| `keep` | Leave values unchanged. Only safe when all I > 0 is guaranteed. |

### 2b  q-range and SNR selection

Three optional cuts, applied in sequence:

1. **Lower bound:** discard points with `q < q_min`.
2. **Upper bound:** discard points with `q > q_max`.
3. **SNR tail trim:** scan from the highest-q point inward; discard leading
   points where `I(q)/σ(q) < snr_threshold`.  Removes the noisy high-q tail
   without a manual `q_max`.

### 2c  Logarithmic rebinning

For dense datasets (e.g. 1696 points) the solver is overconstrained relative to
the number of basis functions, and the grid search can be slow.  Log-rebinning
averages N input points into M log-spaced bins:

```
q_new   = mean(q_j)
I_new   = mean(I_j)
σ_new   = sqrt(Σ σ_j²) / n_j        (standard error of the mean)
```

Empty bins are silently discarded.  This preserves the relative q-spacing that
matters for the Fourier kernel and reduces N to a manageable size.

---

## 3  Basis

**Module:** `src/basis.rs`, `src/bspline.rs`  
**Trait:** `BasisSet`

P(r) is represented as a finite linear combination of basis functions φ_j(r):

```
P(r) ≈ Σ_j  c_j · φ_j(r)
```

The choice of basis determines smoothness properties and boundary behaviour.

### 3a  Rectangular bins (`--basis rect`)

The simplest basis: n equal-width bins of width Δr = D_max / n,

```
φ_j(r) = 1   if  (j-½)Δr ≤ r < (j+½)Δr
        = 0   otherwise
```

The kernel integral

```
K_ij = ∫ φ_j(r) · sinc(q_i r) · r² dr
     = ∫_{r_j - Δr/2}^{r_j + Δr/2}  4π r² sinc(q_i r) dr
```

is evaluated analytically (closed-form antiderivative of r² sinc(qr)).

The bin centres r_j = (j + ½)Δr are the output r-grid.  By construction,
the last bin centre sits at D_max − Δr/2; P(r) is not structurally zero at
D_max. Boundary constraints (Stage 5) correct this.

### 3b  Cubic B-splines (`--basis spline`)

P(r) is represented on a clamped knot vector.  For n free parameters, the code
builds n + 2 B-splines on the knot vector

```
[0, 0, 0, 0,  t_1, …, t_{n-3},  D_max, D_max, D_max, D_max]
```

and then **drops the two endpoint basis functions** B_0 and B_{n+1} (which are
the only ones non-zero at r = 0 and r = D_max respectively).  The remaining n
interior splines satisfy

```
φ_j(0) = 0   and   φ_j(D_max) = 0   for all j ∈ {1, …, n}
```

so P(r) is structurally zero at both boundaries without any additional
constraint rows.

The kernel matrix entry

```
K_ij = 4π ∫_0^{D_max}  φ_j(r) · sinc(q_i r) · r² dr
```

is computed by 5-point Gauss–Legendre quadrature on each knot span.

---

## 4  Build weighted system

**Module:** `src/kernel.rs`  
**Function:** `build_weighted_system`

The forward model is a Fredholm integral equation of the first kind:

```
I(q) = 4π ∫_0^{D_max}  P(r) · sinc(qr) · r² dr
```

After basis expansion this becomes the linear system **Kc ≈ I**, where

```
K_ij  =  4π ∫ φ_j(r) · sinc(q_i r) · r² dr
```

Measurement errors σ_i enter via an error-weighted reformulation.  Define the
diagonal weight matrix **W = diag(1/σ_i)**.  The weighted system

```
K_w c  ≈  I_w
```

where **K_w = W K** and **I_w = W I**, has the property that minimising
‖K_w c − I_w‖² is equivalent to minimising the chi-squared objective:

```
χ²(c) = Σ_i  [(I(q_i) − (Kc)_i) / σ_i]²
```

The weighted matrices are passed to the solver unchanged, keeping error
weighting entirely separate from regularisation.

---

## 5  Boundary constraints (optional, rect basis only)

**Module:** `src/kernel.rs`  
**Function:** `append_boundary_constraints`

For the rect basis, P(r) is not structurally zero at r = 0 or r = D_max.
Two soft equality constraints are added by augmenting the weighted system with
virtual data rows:

```
[ K_w    ]       [ I_w ]
[ w·e₁ᵀ ]  c  ≈ [  0  ]   ← P(r = 0)    = 0
[ w·eₙᵀ ]       [  0  ]   ← P(r = D_max) = 0
```

where e₁ and eₙ are the first and last standard basis vectors. These rows look
exactly like two additional data points with target value zero, so they flow
through GCV, L-curve, and Bayesian evidence without special-casing.

**Choosing the weight.** The default weight is:

```
w = sqrt(N) × rms(1/σ)     where  rms(1/σ) = sqrt(mean_i(1/σ_i²))
```

This makes the constraint as influential as roughly √N data points — decisive
without dominating the fit. The TOML `boundary_weight` field scales this:
`0.0` = automatic, `k > 0` = k × default, `-1` = disabled.

Enabled via `[constraints] boundary_weight = 0` in `unfourier.toml`.

---

## 6  Regulariser

**Module:** `src/regularise.rs`  
**Trait:** `Regulariser`

Solving the weighted system directly (without regularisation) fits noise.
Tikhonov regularisation adds a penalty on undesirable features of c:

```
min_c  ‖K_w c − I_w‖²  +  λ_eff · ‖L c‖²
```

The regularisation matrix L encodes the physics we want to enforce.

### 6a  Second-derivative penalty (default)

`SecondDerivative` produces the (n−2) × n finite-difference matrix D₂:

```
D₂[i, i]   =  1
D₂[i, i+1] = −2
D₂[i, i+2] =  1
```

so that `(D₂ c)[i] ≈ c[i] − 2c[i+1] + c[i+2]`, the discrete second derivative.
Minimising ‖D₂ c‖² penalises curvature — it encourages P(r) to be smooth.

### 6b  First-derivative penalty

`FirstDerivative` produces the (n−1) × n matrix D₁:

```
D₁[i, i]   = −1
D₁[i, i+1] = +1
```

Minimising ‖D₁ c‖² penalises slope — it penalises large step changes between
adjacent bins, targeting single-step discontinuities that the curvature penalty
misses.

### 6c  Combined penalty (M8)

`CombinedDerivative { d1_weight, d2_weight }` stacks both matrices with
per-component weights:

```
L_combined = [ sqrt(d₁) · D₁ ]      shape: (2n−3) × n
             [ sqrt(d₂) · D₂ ]
```

so that:

```
‖L_combined c‖² = d₁ ‖D₁ c‖²  +  d₂ ‖D₂ c‖²
```

When d₁ = 0, d₂ = 1 the combined matrix reduces exactly to `SecondDerivative`
— no regression.  Enabled via `[constraints] d1_smoothness = 0` in
`unfourier.toml`.

---

## 7  λ selection

**Module:** `src/lambda_select.rs`  
**Trait:** `LambdaSelector`

The regularisation strength λ controls the bias–variance trade-off:

- λ too small → P(r) fits noise and oscillates wildly.
- λ too large → P(r) is over-smoothed and loses genuine structure.

All auto-selectors evaluate the same log-spaced grid of λ candidates, computing
the quantities below for each one, then pick the best by their respective
criterion.

### Internal λ scaling

The raw user-facing λ is rescaled before use:

```
λ_eff = λ × tr(KᵀK) / tr(LᵀL)
```

This makes λ = 1 correspond to equal weight for data and regularisation terms
(in the trace-norm sense) regardless of intensity scale or error magnitudes.
The `GridMatrices` struct precomputes the trace ratio once and reuses it across
the entire λ grid.

### 7a  Generalised Cross-Validation (GCV)

For the weighted system K_w c ≈ I_w the GCV score is:

```
GCV(λ) = RSS_w(λ) / n / (1 − df(λ)/n)²
```

where:

```
RSS_w(λ) = ‖K_w c(λ) − I_w‖²           (weighted residual sum of squares)
df(λ)    = tr(H(λ))                      (effective degrees of freedom)
H(λ)     = K_w (KᵀK + λ_eff LᵀL)⁻¹ Kᵀ  (hat / influence matrix)
```

Using the cyclic trace identity:

```
df(λ) = tr(H) = n_r − λ_eff · tr((KᵀK + λ_eff LᵀL)⁻¹ LᵀL)
```

The inner trace is computed via Cholesky: if A = C Cᵀ, then
tr(A⁻¹ LᵀL) = ‖X‖_F² where C X = LᵀL column by column.

GCV minimisation is the default. When the GCV landscape is very flat (relative
variation < 10%) the selector falls back to L-curve to avoid picking an
under-regularised λ.

### 7b  L-curve

Plot the points

```
(ρ_i, η_i) = (log RSS_w(λ_i),  log ‖L c(λ_i)‖²)
```

over the λ grid. The "corner" of the resulting L-shaped curve corresponds to
the optimal balance between residual and solution norm. The corner is found by
discrete curvature of the parametric curve (ρ(λ), η(λ)):

```
κ_i = |ρ′_i η″_i − η′_i ρ″_i| / (ρ′_i² + η′_i²)^(3/2)
```

where primes are first and second centred finite differences with respect to the
log-λ index. The λ with maximum κ is chosen.

### 7c  Bayesian evidence

The Bayesian log-evidence for the Gaussian model is:

```
log P(I|λ) = −½ [ RSS_w  +  λ_eff ‖Lc‖²
                 + log det(KᵀK + λ_eff LᵀL)
                 − N_r log λ_eff ]
             + const
```

Maximising over λ balances goodness-of-fit (RSS_w), solution complexity (‖Lc‖²),
and the evidence-counting terms. This method also produces the posterior
covariance:

```
Σ_post = (KᵀK + λ_eff LᵀL)⁻¹
```

from which per-r uncertainty bars σ_P(r) = sqrt(Σ_post[j,j]) are extracted.

### 7d  Manual

λ is supplied directly via `--lambda`. No grid search is performed.
`λ_eff = λ × tr(KᵀK) / tr(LᵀL)` is still applied.

---

## 8  Solve

**Module:** `src/solver.rs`  
**Struct:** `TikhonovSolver`

Given the selected λ_eff, form the normal equations for the augmented system:

```
(KᵀK + λ_eff LᵀL) c = Kᵀ I_w
```

where K = K_w here (the hat-matrix derivation uses the weighted kernel). The
system matrix A = KᵀK + λ_eff LᵀL is symmetric positive-definite for any
λ_eff > 0.  It is solved by:

1. **Cholesky factorisation** (preferred — A is SPD, Cholesky is the fastest
   and most numerically stable approach).
2. **LU factorisation** (fallback if Cholesky fails, e.g. near-singular A at
   very small λ).

The result is the unconstrained Tikhonov solution c_unc.

---

## 9  Non-negativity enforcement

**Module:** `src/nonneg.rs`  
**Trait:** `NonNegativityStrategy`

P(r) is a probability-related function and must be non-negative: c_j ≥ 0 for
all j. The unconstrained Tikhonov solution can violate this, especially near the
boundaries.

The solver uses **Projected Gradient NNLS** (`ProjectedGradient`): given the
quadratic objective

```
min_c  ½ cᵀ A c − bᵀ c    s.t.  c ≥ 0
```

(where A = KᵀK + λ_eff LᵀL and b = Kᵀ I_w), the projected gradient iteration
is:

```
c ← max(c − α · (Ac − b),  0)
```

with a backtracking step size α chosen to satisfy the Armijo condition.  The
algorithm converges to the true NNLS minimum without the cascade-zeroing
pathology of iterative clipping. Up to 500 iterations are allowed; convergence
is declared when ‖gradient restricted to active set‖ < 1e-8.

---

## 10  Back-calculation and output

**Module:** `src/kernel.rs`, `src/output.rs`

The fitted intensity is back-calculated from the final coefficient vector:

```
I_calc(q_i) = (K · c)[i]     (unweighted kernel, final constrained c)
```

The reduced chi-squared is:

```
χ²_red = (1/N) Σ_i [(I_obs(q_i) − I_calc(q_i)) / σ_i]²
```

A value close to 1.0 indicates the fit is consistent with the measurement
uncertainties.  Values significantly below 1 suggest over-fitting (λ too
small); values significantly above 1 suggest over-smoothing (λ too large).

**Spline boundary rows.** The spline basis structurally enforces P(0) = P(D_max)
= 0, but the interior Greville abscissae returned by `r_values()` do not include
the boundary points.  The output stage appends explicit rows `(0, 0)` and
`(D_max, 0)` so the written curve visibly reaches zero at both ends.

### Output files

**P(r) output** (`--output`, default stdout):

```
# r(Å)    P(r)          [σ_P(r)  — Bayesian mode only]
1.234e+00  4.567e-03    [8.9e-04]
...
```

**Fit output** (`--fit-output`):

```
# q(1/Å)   I_obs   I_calc   sigma
1.00e-02   9.71e-01  9.68e-01  9.71e-04
...
```

---

## Pipeline diagram

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  INPUT: q, I(q), σ(q)                                           │
 └─────────────────────────┬────────────────────────────────────────┘
                            │
              ┌─────────────▼───────────────┐
              │  PREPROCESS                  │
              │  1. Clip / omit negatives    │
              │  2. q-range / SNR filter     │
              │  3. Log-rebin (optional)     │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
              │  BASIS                       │
              │  rect: uniform bins          │
              │  spline: clamped B-splines   │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
              │  BUILD WEIGHTED SYSTEM       │
              │  K_w = diag(1/σ) · K         │
              │  I_w = I / σ                 │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
              │  BOUNDARY CONSTRAINTS        │  ← rect only, optional
              │  Augment K_w, I_w with       │
              │  soft P(0)=P(Dmax)=0 rows    │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
              │  REGULARISER                 │
              │  L = D₂ (curvature, default) │
              │    or sqrt(d₁)D₁ ⊕ sqrt(d₂)D₂│
              │  LᵀL = gram_matrix(n)        │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
              │  λ SELECTION                 │
              │  gcv / lcurve / bayes /      │
              │  manual                      │
              │  λ_eff = λ · tr(KᵀK)/tr(LᵀL)│
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
              │  TIKHONOV SOLVE              │
              │  (KᵀK + λ_eff LᵀL) c = KᵀI │
              │  Cholesky → c_unc            │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
              │  NON-NEGATIVITY (NNLS)       │
              │  Projected gradient          │
              │  c_j ← max(c_j, 0)          │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
              │  OUTPUT                      │
              │  P(r) file  +  fit file      │
              │  χ²_red  +  [σ_P(r) if Bayes]│
              └──────────────────────────────┘
```
