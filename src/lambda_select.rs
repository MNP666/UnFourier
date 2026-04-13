//! Automatic λ selection for Tikhonov regularisation.
//!
//! # Overview
//!
//! Choosing the regularisation strength λ is the central practical challenge
//! in indirect Fourier transformation. Too small: P(r) fits noise and
//! oscillates. Too large: P(r) is over-smoothed and loses structure.
//!
//! This module implements two classical automatic selectors:
//!
//! - **GCV** (Generalised Cross-Validation): minimise the leave-one-out
//!   prediction error, estimated analytically from the hat matrix trace.
//!   Works well for moderate noise; can underestimate λ when noise is large.
//!
//! - **L-curve**: plot log(residual norm) vs log(solution norm) over a λ grid
//!   and pick the "corner" of maximum curvature. More robust to non-Gaussian
//!   noise than GCV; can be ambiguous if the curve is not clearly L-shaped.
//!
//! Both methods evaluate all λ candidates independently, making the grid
//! evaluation a natural target for `rayon::par_iter()` in a future milestone.
//!
//! # Usage
//!
//! ```ignore
//! let evals = evaluate_lambda_grid(&lambdas, &matrices, ...);
//! let best  = GcvSelector.select(&evals);     // or LCurveSelector
//! let soln  = apply_nonneg(best);
//! ```
//!
//! # Mathematical details
//!
//! See `saxs_ift_postmortem.md` for why proportional noise on sphere data
//! makes any λ selector degenerate. The selectors here assume a well-conditioned
//! weight matrix (i.e., Debye-like data without near-zero I(q)).
//!
//! ## GCV formula
//!
//! For the weighted system K_w·c ≈ I_w (K_w = diag(1/σ)·K), the GCV score is:
//!
//! ```text
//! GCV(λ) = RSS_w(λ) / n / (1 − df(λ)/n)²
//! ```
//!
//! where:
//! - `RSS_w = ‖K_w·c(λ) − I_w‖²`  (residual in the weighted space)
//! - `df(λ) = tr(H(λ))`            (effective degrees of freedom, hat matrix trace)
//! - `H(λ)  = K_w (A + λ_eff H_reg)⁻¹ K_wᵀ`  (hat / influence matrix)
//!
//! Using the cyclic trace identity:
//! ```text
//! df(λ) = tr(H) = tr((A + λ_eff H_reg)⁻¹ A)
//!               = n_r − λ_eff · tr((A + λ_eff H_reg)⁻¹ H_reg)
//! ```
//!
//! The inner trace is computed via a Cholesky solve:
//! if M = C·Cᵀ (Cholesky factor), then tr(M⁻¹ H_reg) = ‖X‖_F² where C·X = H_reg·col_j.
//!
//! ## L-curve corner detection
//!
//! Points (ρ_i, η_i) = (log RSS_w, log ‖Lc‖²) are computed for each λ_i.
//! The corner is found by discrete curvature:
//!
//! ```text
//! κ_i = |ρ'_i · η''_i − η'_i · ρ''_i| / (ρ'_i² + η'_i²)^(3/2)
//! ```
//!
//! where primes denote first/second centred finite differences with respect to
//! the log(λ) index. The candidate with maximum κ is chosen.

use anyhow::{anyhow, Result};
use nalgebra::{DMatrix, DVector};

use crate::nonneg::projected_gradient_nnls;
use crate::solver::Solution;

/// If the relative variation of finite GCV scores across the grid is below
/// this threshold, the GCV landscape is considered flat and the selector
/// falls back to L-curve to avoid picking an under-regularised λ.
const GCV_FLAT_THRESHOLD: f64 = 0.10;

// ---------------------------------------------------------------------------
// LambdaEvaluation
// ---------------------------------------------------------------------------

/// All quantities computed for a single λ candidate.
///
/// Storing everything here (rather than just the score) means the caller can
/// inspect diagnostics and plot the full L-curve without re-running solves.
/// The `solution` field contains the non-negativity-enforced coefficient
/// vector — this is what gets used if this λ is selected.
#[derive(Debug, Clone)]
pub struct LambdaEvaluation {
    /// User-facing λ (before internal trace-ratio scaling).
    pub lambda: f64,

    /// Effective λ actually used: `lambda * tr(KᵀK) / tr(H_reg)`.
    pub lambda_eff: f64,

    /// Weighted residual sum of squares: ‖K_w·c − I_w‖² (unconstrained solve).
    /// This is what enters the GCV numerator and the L-curve x-axis.
    pub rss_weighted: f64,

    /// Regularisation term: ‖L·c‖² = cᵀ LᵀL c (unconstrained solve).
    /// This is the L-curve y-axis.
    pub solution_norm: f64,

    /// Effective degrees of freedom: df = tr(H(λ)) = tr((A + λ_eff H_reg)⁻¹ A).
    pub df: f64,

    /// GCV score: RSS_w / n / (1 − df/n)².
    /// Lower is better. Minimise over λ to select.
    pub gcv: f64,

    /// Bayesian log-evidence: log P(I|λ) = -½[RSS_w + λ_eff‖Lc‖² + log det(A + λ_eff H) − N_r log λ_eff]
    /// Higher is better. Maximise over λ to select (used by `BayesianEvidence` in M4).
    pub log_evidence: f64,

    /// Reduced χ² from the non-negativity-enforced solution (for display).
    pub chi_squared: f64,

    /// The non-negativity-enforced coefficient solution.
    pub solution: Solution,
}

// ---------------------------------------------------------------------------
// LambdaSelector trait
// ---------------------------------------------------------------------------

/// Select the best λ from a grid of pre-evaluated candidates.
///
/// # Design
///
/// The trait operates on a slice of [`LambdaEvaluation`] rather than raw
/// data so that all expensive computation (matrix solves) is done once in
/// [`evaluate_lambda_grid`], and multiple selectors can be compared cheaply
/// by re-running `select` on the same evaluations.
pub trait LambdaSelector: Send + Sync {
    fn name(&self) -> &str;

    /// Return the index of the selected candidate in `candidates`.
    ///
    /// Panics if `candidates` is empty.
    fn select(&self, candidates: &[LambdaEvaluation]) -> usize;
}

// ---------------------------------------------------------------------------
// GcvSelector
// ---------------------------------------------------------------------------

/// Select λ by minimising the Generalised Cross-Validation score.
///
/// GCV approximates the leave-one-out cross-validation error analytically.
/// It tends to work well for moderate noise with a well-conditioned weight
/// matrix. Can underestimate λ when noise is large or weights vary greatly.
pub struct GcvSelector;

impl LambdaSelector for GcvSelector {
    fn name(&self) -> &str {
        "gcv"
    }

    fn select(&self, candidates: &[LambdaEvaluation]) -> usize {
        assert!(!candidates.is_empty(), "LambdaEvaluation slice is empty");

        // Guard against degenerate GCV values (NaN / Inf) which arise when
        // df ≈ n (the hat matrix trace equals n_q, giving a zero denominator).
        let finite: Vec<f64> = candidates
            .iter()
            .filter(|e| e.gcv.is_finite())
            .map(|e| e.gcv)
            .collect();

        if finite.is_empty() {
            return 0;
        }

        let gcv_min = finite.iter().cloned().fold(f64::INFINITY, f64::min);
        let gcv_max = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variation = (gcv_max - gcv_min) / gcv_min;

        if variation < GCV_FLAT_THRESHOLD {
            eprintln!(
                "warning: GCV landscape flat ({:.1}% variation across grid); \
                 falling back to L-curve",
                variation * 100.0
            );
            return LCurveSelector.select(candidates);
        }

        candidates
            .iter()
            .enumerate()
            .filter(|(_, e)| e.gcv.is_finite())
            .min_by(|(_, a), (_, b)| a.gcv.partial_cmp(&b.gcv).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// LCurveSelector
// ---------------------------------------------------------------------------

/// Select λ by finding the corner of maximum curvature on the L-curve.
///
/// The L-curve plots log(RSS_w) against log(solution_norm) parametrised by λ.
/// The "corner" at maximum curvature represents the best trade-off between
/// data fit (small RSS_w) and smoothness (small ‖Lc‖²).
///
/// Corner detection uses second-order finite differences of the discrete curve
/// with respect to the log-λ index. Boundary points (first and last candidates)
/// are excluded since centred differences require a neighbour on each side.
pub struct LCurveSelector;

impl LambdaSelector for LCurveSelector {
    fn name(&self) -> &str {
        "lcurve"
    }

    fn select(&self, candidates: &[LambdaEvaluation]) -> usize {
        assert!(!candidates.is_empty(), "LambdaEvaluation slice is empty");

        let n = candidates.len();
        if n <= 2 {
            return 0;
        }

        // Build (x, y) = (log RSS_w, log solution_norm) for each candidate.
        // Replace non-positive values with a small floor to avoid −∞ in log.
        let floor = 1e-300_f64;
        let x: Vec<f64> = candidates
            .iter()
            .map(|e| e.rss_weighted.max(floor).ln())
            .collect();
        let y: Vec<f64> = candidates
            .iter()
            .map(|e| e.solution_norm.max(floor).ln())
            .collect();

        // Discrete curvature at interior points using centred finite differences.
        // dx_i  ≈ (x_{i+1} − x_{i-1}) / 2
        // ddx_i ≈  x_{i+1} − 2·x_i + x_{i-1}   (and same for y)
        // κ_i   = |dx·ddy − dy·ddx| / (dx² + dy²)^(3/2)
        let mut best_idx = 1usize;
        let mut best_kappa = f64::NEG_INFINITY;

        for i in 1..(n - 1) {
            let dx = (x[i + 1] - x[i - 1]) / 2.0;
            let dy = (y[i + 1] - y[i - 1]) / 2.0;
            let ddx = x[i + 1] - 2.0 * x[i] + x[i - 1];
            let ddy = y[i + 1] - 2.0 * y[i] + y[i - 1];

            let denom = (dx * dx + dy * dy).powf(1.5);
            if denom < 1e-30 {
                continue; // degenerate segment (flat curve)
            }

            let kappa = (dx * ddy - dy * ddx).abs() / denom;
            if kappa > best_kappa {
                best_kappa = kappa;
                best_idx = i;
            }
        }

        best_idx
    }
}

// ---------------------------------------------------------------------------
// BayesianEvidence
// ---------------------------------------------------------------------------

/// Selects λ by maximising the Bayesian log-evidence log P(I|λ).
///
/// The evidence marginalises over the P(r) coefficients, yielding an objective
/// that balances data fit and prior smoothness without requiring a separate
/// criterion (GCV, L-curve). The formula evaluated per candidate is:
///
/// ```text
/// log P(I|λ) = -½ [ RSS_w + λ_eff‖Lc‖² + log det(A + λ_eff H) − N_r log λ_eff ]
/// ```
///
/// where A = KᵀWK, H = LᵀL, RSS_w is the weighted residual at the MAP
/// solution c*(λ), and N_r is the number of basis coefficients.
///
/// The log det term is obtained for free from the Cholesky factor already
/// computed during grid evaluation: log det(M) = 2 Σ_i log L_{ii}.
///
pub struct BayesianEvidence;

impl LambdaSelector for BayesianEvidence {
    fn name(&self) -> &str {
        "bayes"
    }

    /// Return the index of the candidate with the highest log-evidence.
    ///
    /// Non-finite values (e.g. from a degenerate λ = 0) are excluded.
    /// Falls back to index 0 only if no finite value is found.
    fn select(&self, candidates: &[LambdaEvaluation]) -> usize {
        assert!(!candidates.is_empty(), "LambdaEvaluation slice is empty");
        candidates
            .iter()
            .enumerate()
            .filter(|(_, e)| e.log_evidence.is_finite())
            .max_by(|(_, a), (_, b)| a.log_evidence.partial_cmp(&b.log_evidence).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Pre-computed matrices for λ grid evaluation
// ---------------------------------------------------------------------------

/// Pre-computed matrices needed to evaluate λ candidates efficiently.
///
/// These are computed once from the weighted kernel and passed to
/// [`evaluate_lambda_grid`]. Building them outside the grid loop avoids
/// redundant `O(n_q · n_r²)` matrix multiplications.
pub struct GridMatrices {
    /// K_wᵀ K_w  (n_r × n_r)
    pub ktk: DMatrix<f64>,

    /// K_wᵀ I_w  (n_r)
    pub kti: DVector<f64>,

    /// Regularisation Gram matrix LᵀL  (n_r × n_r)
    pub ltl: DMatrix<f64>,

    /// Unweighted kernel  (n_q × n_r), for back-calculation and χ²
    pub k_unweighted: DMatrix<f64>,

    /// Original (unweighted) intensities
    pub i_observed: Vec<f64>,

    /// Measurement errors (for χ²)
    pub sigma: Vec<f64>,

    /// ‖I_w‖²  (precomputed for RSS calculation)
    pub i_w_norm_sq: f64,

    /// tr(KᵀK) / tr(LᵀL) — used to scale λ internally
    pub trace_ratio: f64,
}

impl GridMatrices {
    /// Build from the weighted system and basis parameters.
    ///
    /// # Arguments
    /// - `k_weighted`   — K_w = diag(1/σ)·K (n_q × n_r)
    /// - `i_weighted`   — I_w = I/σ          (n_q)
    /// - `k_unweighted` — K                  (n_q × n_r)
    /// - `i_observed`   — raw I(q)            (n_q)
    /// - `sigma`        — measurement errors  (n_q)
    /// - `ltl`          — precomputed LᵀL Gram matrix (n_r × n_r)
    pub fn build(
        k_weighted: &DMatrix<f64>,
        i_weighted: &[f64],
        k_unweighted: &DMatrix<f64>,
        i_observed: &[f64],
        sigma: &[f64],
        ltl: DMatrix<f64>,
    ) -> Self {
        let ktk = k_weighted.transpose() * k_weighted;
        let i_w_vec = DVector::from_column_slice(i_weighted);
        let kti = k_weighted.transpose() * &i_w_vec;
        let i_w_norm_sq = i_weighted.iter().map(|x| x * x).sum::<f64>();

        let trace_ktk = ktk.trace();
        let trace_ltl = ltl.trace().max(1e-10);
        let trace_ratio = trace_ktk / trace_ltl;

        Self {
            ktk,
            kti,
            ltl,
            k_unweighted: k_unweighted.clone(),
            i_observed: i_observed.to_vec(),
            sigma: sigma.to_vec(),
            i_w_norm_sq,
            trace_ratio,
        }
    }
}

// ---------------------------------------------------------------------------
// evaluate_lambda_grid
// ---------------------------------------------------------------------------

/// Evaluate all λ candidates and return a [`LambdaEvaluation`] for each.
///
/// For each λ:
/// 1. Compute λ_eff = λ · trace_ratio.
/// 2. Solve the unconstrained Tikhonov system (for GCV / L-curve quantities).
/// 3. Compute RSS_w, solution_norm, df, and GCV score from the unconstrained c.
/// 4. Apply non-negativity (projected-gradient NNLS) to get the final
///    coefficient vector.
/// 5. Compute χ² from the non-negativity-enforced solution (for display).
///
/// The unconstrained solve is used for λ selection because GCV and the L-curve
/// assume a linear estimator. Non-negativity makes the estimator nonlinear and
/// invalidates the GCV formula. The constrained solution is still used for the
/// final output once the best λ is selected.
///
/// Each evaluation is fully independent — this loop is a natural candidate for
/// `rayon::par_iter()` when parallelism is added in a future milestone.
pub fn evaluate_lambda_grid(lambdas: &[f64], m: &GridMatrices) -> Result<Vec<LambdaEvaluation>> {
    let n_q = m.i_observed.len();
    let n_r = m.ktk.ncols();

    let mut evaluations = Vec::with_capacity(lambdas.len());

    // rayon::par_iter() ready: each lambda evaluation is fully independent.
    for &lambda in lambdas {
        let lambda_eff = lambda * m.trace_ratio;

        // ---- unconstrained solve (for GCV / L-curve) ----
        let mat = &m.ktk + &m.ltl * lambda_eff;

        let chol = mat.clone().cholesky().ok_or_else(|| {
            anyhow!(
                "Cholesky failed for λ = {:.3e} (λ_eff = {:.3e}); system is not positive-definite",
                lambda,
                lambda_eff
            )
        })?;

        let c_unconstrained = chol.solve(&m.kti);

        // RSS_w = ‖I_w‖² − 2·cᵀ·KᵀI_w + cᵀ·KᵀK·c
        let rss_weighted = {
            let kti_dot = c_unconstrained.dot(&m.kti);
            let ktk_c = &m.ktk * &c_unconstrained;
            let ctktk_c = c_unconstrained.dot(&ktk_c);
            (m.i_w_norm_sq - 2.0 * kti_dot + ctktk_c).max(0.0)
        };

        // solution_norm = cᵀ LᵀL c
        let solution_norm = {
            let ltl_c = &m.ltl * &c_unconstrained;
            c_unconstrained.dot(&ltl_c)
        };

        // log_evidence = -½ [ RSS_w + λ_eff‖Lc‖² + log det(A + λ_eff H) − N_r log λ_eff ]
        //
        // log det(A + λ_eff H) = 2 Σ_i log L_{ii}  where  mat = L Lᵀ (Cholesky).
        // All diagonal elements of L are positive by construction.
        let log_evidence = {
            let log_det_a = 2.0 * chol.l().diagonal().iter().map(|&x| x.ln()).sum::<f64>();
            -0.5 * (rss_weighted + lambda_eff * solution_norm + log_det_a
                - n_r as f64 * lambda_eff.ln())
        };

        // df = tr(H) = tr((A + λ_eff H_reg)⁻¹ A)
        //            = n_r − λ_eff · tr(M⁻¹ H_reg)
        //
        // tr(M⁻¹ H_reg) computed via:  if M = CC^T then tr(M⁻¹ H_reg) = ‖X‖_F²
        // where C·X = H_reg  (solve each column of H_reg against the Cholesky factor).
        //
        // For n_r = 100, H_reg (= LᵀL) is 100×100.  One matrix solve.
        let df = {
            let m_inv_h = chol.solve(&m.ltl);
            let trace_m_inv_h = m_inv_h.trace();
            let df_raw = n_r as f64 - lambda_eff * trace_m_inv_h;
            // Clamp to a safe range: df must be in [1, n_q − 1]
            df_raw.clamp(1.0, (n_q - 1) as f64)
        };

        // GCV score: RSS_w / n / (1 − df/n)²
        let gcv = {
            let denom = (1.0 - df / n_q as f64).powi(2);
            if denom < 1e-30 {
                f64::INFINITY
            } else {
                rss_weighted / n_q as f64 / denom
            }
        };

        // ---- constrained solve (non-negativity, for final P(r)) ----
        // Warm-start from the unconstrained solution already computed above.
        let coeffs = projected_gradient_nnls(&mat, &m.kti, &c_unconstrained, 500, 1e-8);

        // Back-calculate I(q) and χ² from constrained solution
        let coeff_vec = DVector::from_column_slice(&coeffs);
        let i_calc: Vec<f64> = (&m.k_unweighted * &coeff_vec).iter().cloned().collect();
        let chi_squared = reduced_chi_squared(&m.i_observed, &i_calc, &m.sigma);

        evaluations.push(LambdaEvaluation {
            lambda,
            lambda_eff,
            rss_weighted,
            solution_norm,
            df,
            gcv,
            log_evidence,
            chi_squared,
            solution: Solution {
                coeffs,
                coeff_err: None,
                i_calc,
                chi_squared,
                lambda_effective: Some(lambda_eff),
            },
        });
    }

    Ok(evaluations)
}

// ---------------------------------------------------------------------------
// Posterior covariance
// ---------------------------------------------------------------------------

/// Compute marginal posterior standard deviations of the solved coefficients.
///
/// For the weighted Tikhonov problem the posterior covariance is:
///
/// ```text
/// Σ = (KᵀWK + λ_eff LᵀL)⁻¹  =  (A + λ_eff H)⁻¹
/// ```
///
/// The marginal σ for each coefficient is `sqrt(Σ_{jj})`.
///
/// # Computation
///
/// The inverse is obtained by solving `(A + λ_eff H) X = I` against the
/// Cholesky factor of `A + λ_eff H` — the same factorisation already computed
/// during grid evaluation.  This costs O(n_r³) but n_r is typically ~100 so
/// it is negligible compared with the grid evaluation.
///
/// Note: `lambda_eff` must be the *effective* (scaled) λ stored in
/// [`LambdaEvaluation::lambda_eff`], not the user-facing λ.
pub fn posterior_coeff_sigma(m: &GridMatrices, lambda_eff: f64) -> Result<Vec<f64>> {
    let n_r = m.ktk.ncols();
    let mat = &m.ktk + &m.ltl * lambda_eff;

    let chol = mat.cholesky().ok_or_else(|| {
        anyhow!(
            "Cholesky failed for posterior covariance at λ_eff = {:.3e}; \
             matrix is not positive-definite",
            lambda_eff
        )
    })?;

    // Solve (A + λ_eff H) X = I  →  X = Σ = (A + λ_eff H)^{-1}
    let identity = DMatrix::<f64>::identity(n_r, n_r);
    let sigma_mat = chol.solve(&identity);

    // Marginal std dev for each coefficient: sqrt(Σ_{jj})
    let coeff_sigma: Vec<f64> = (0..n_r)
        .map(|j| sigma_mat[(j, j)].max(0.0).sqrt())
        .collect();
    Ok(coeff_sigma)
}

// ---------------------------------------------------------------------------
// log-spaced λ grid helpers
// ---------------------------------------------------------------------------

/// Generate `n` log-spaced λ values in `[lambda_min, lambda_max]`.
pub fn log_lambda_grid(lambda_min: f64, lambda_max: f64, n: usize) -> Vec<f64> {
    assert!(lambda_min > 0.0 && lambda_max > lambda_min);
    assert!(n >= 2);
    let log_min = lambda_min.ln();
    let log_max = lambda_max.ln();
    (0..n)
        .map(|i| (log_min + (log_max - log_min) * i as f64 / (n - 1) as f64).exp())
        .collect()
}

/// Estimate a reasonable λ search range from the pre-computed matrices.
///
/// `lambda_min` is set to give an essentially unregularised solution;
/// `lambda_max` is set to give a heavily over-smoothed solution.
///
/// Both are in user-facing λ units (before trace-ratio scaling).
/// In practice, useful values nearly always fall within this range.
pub fn estimate_lambda_range(_m: &GridMatrices) -> (f64, f64) {
    // In scaled units, λ_eff = 1 corresponds to equal trace weights.
    // λ_min: 6 decades below equal-weight  → effectively unregularised
    // λ_max: 3 decades above equal-weight  → heavily over-smoothed
    // Both expressed in user-facing λ (before trace-ratio scaling).
    let lambda_min = 1e-6_f64;
    let lambda_max = 1e3_f64;
    (lambda_min, lambda_max)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Reduced χ² from the non-negativity-enforced solution.
fn reduced_chi_squared(i_observed: &[f64], i_calc: &[f64], sigma: &[f64]) -> f64 {
    let n = i_observed.len();
    if n == 0 {
        return 0.0;
    }
    i_observed
        .iter()
        .zip(i_calc.iter())
        .zip(sigma.iter())
        .map(|((&io, &ic), &s)| {
            let s = if s > 0.0 { s } else { 1.0 };
            ((io - ic) / s).powi(2)
        })
        .sum::<f64>()
        / n as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_grid_endpoints() {
        let grid = log_lambda_grid(1e-4, 1e2, 10);
        assert_eq!(grid.len(), 10);
        assert!((grid[0] - 1e-4).abs() < 1e-12);
        assert!((grid[9] - 1e2).abs() < 1e-8);
    }

    #[test]
    fn gcv_selects_finite_minimum() {
        // Build a trivial L-curve with two evaluations to verify selection logic.
        let make_eval = |gcv: f64| LambdaEvaluation {
            lambda: 1.0,
            lambda_eff: 1.0,
            rss_weighted: 1.0,
            solution_norm: 1.0,
            df: 5.0,
            gcv,
            log_evidence: 0.0,
            chi_squared: 1.0,
            solution: Solution {
                coeffs: vec![],
                coeff_err: None,
                i_calc: vec![],
                chi_squared: 1.0,
                lambda_effective: Some(1.0),
            },
        };

        let evals = vec![make_eval(f64::INFINITY), make_eval(0.5), make_eval(1.0)];
        assert_eq!(GcvSelector.select(&evals), 1); // index 1 has minimum finite GCV
    }

    #[test]
    fn gcv_flat_landscape_falls_back_to_lcurve() {
        // Five candidates whose GCV values vary by < 1% (well below the 10%
        // threshold). The GCV minimum is at index 4 (gcv = 0.995).
        // The L-curve data forms a clear L-shape, so LCurveSelector picks an
        // interior index (not 4). GcvSelector must agree with LCurveSelector.
        let make_eval = |gcv: f64, rss: f64, norm: f64| LambdaEvaluation {
            lambda: 1.0,
            lambda_eff: 1.0,
            rss_weighted: rss,
            solution_norm: norm,
            df: 5.0,
            gcv,
            log_evidence: 0.0,
            chi_squared: 1.0,
            solution: Solution {
                coeffs: vec![],
                coeff_err: None,
                i_calc: vec![],
                chi_squared: 1.0,
                lambda_effective: Some(1.0),
            },
        };

        // GCV flat: variation = (1.003 - 0.995) / 0.995 ≈ 0.8% < 10%.
        // GCV minimum at index 4. L-curve forms a clear elbow — corner away from index 4.
        let evals = vec![
            make_eval(1.000, 1e-4, 1e4),
            make_eval(1.001, 1e-2, 1e3),
            make_eval(1.002, 1e0, 1e1),
            make_eval(1.003, 1e3, 1e0),
            make_eval(0.995, 1e5, 1e-1), // GCV minimum
        ];

        let gcv_result = GcvSelector.select(&evals);
        let lcurve_result = LCurveSelector.select(&evals);

        // Fallback must agree with direct L-curve selection.
        assert_eq!(
            gcv_result, lcurve_result,
            "flat GCV should delegate to L-curve (got index {gcv_result}, lcurve chose {lcurve_result})"
        );
        // And must differ from the raw GCV minimum (index 4).
        assert_ne!(
            gcv_result, 4,
            "flat GCV fallback must not return the GCV minimum (index 4)"
        );
    }

    #[test]
    fn gcv_normal_landscape_uses_gcv_minimum() {
        // GCV values with > 10% variation: no fallback, GCV minimum is returned.
        let make_eval = |gcv: f64| LambdaEvaluation {
            lambda: 1.0,
            lambda_eff: 1.0,
            rss_weighted: 1.0,
            solution_norm: 1.0,
            df: 5.0,
            gcv,
            log_evidence: 0.0,
            chi_squared: 1.0,
            solution: Solution {
                coeffs: vec![],
                coeff_err: None,
                i_calc: vec![],
                chi_squared: 1.0,
                lambda_effective: Some(1.0),
            },
        };

        // Variation = (2.0 - 0.5) / 0.5 = 300% > 10% — no fallback.
        let evals = vec![make_eval(2.0), make_eval(0.5), make_eval(1.0)];
        assert_eq!(GcvSelector.select(&evals), 1); // GCV minimum at index 1
    }

    /// Build the same tiny synthetic GridMatrices used in other tests.
    fn make_test_matrices() -> GridMatrices {
        use nalgebra::DMatrix;
        let n_q = 8_usize;
        let n_r = 5_usize;
        let k_w = DMatrix::from_fn(n_q, n_r, |i, j| {
            let x = (i + 1) as f64 * (j + 1) as f64 * 0.2;
            x.sin() / x
        });
        let i_w: Vec<f64> = (1..=n_q).map(|i| (i as f64) * 0.05).collect();
        use crate::regularise::Regulariser as _;
        let ltl = crate::regularise::SecondDerivative.gram_matrix(n_r);
        GridMatrices::build(&k_w, &i_w, &k_w.clone(), &i_w, &vec![1.0; n_q], ltl)
    }

    #[test]
    fn posterior_coeff_sigma_shape_and_sign() {
        let m = make_test_matrices();
        let lambda_eff = 1e6_f64; // large enough for Cholesky to be well-conditioned
        let sigma = posterior_coeff_sigma(&m, lambda_eff).unwrap();

        assert_eq!(sigma.len(), m.ktk.ncols(), "sigma length must match n_r");
        assert!(
            sigma.iter().all(|&s| s > 0.0 && s.is_finite()),
            "all sigma values must be positive and finite"
        );
    }

    #[test]
    fn posterior_coeff_sigma_shrinks_with_lambda() {
        // Larger λ → prior dominates → tighter posterior → smaller σ.
        let m = make_test_matrices();
        let sigma_small = posterior_coeff_sigma(&m, 1e4).unwrap();
        let sigma_large = posterior_coeff_sigma(&m, 1e8).unwrap();

        let sum_small: f64 = sigma_small.iter().sum();
        let sum_large: f64 = sigma_large.iter().sum();
        assert!(
            sum_large < sum_small,
            "larger λ should give smaller total uncertainty (got {sum_large:.3e} vs {sum_small:.3e})"
        );
    }

    #[test]
    fn bayes_selects_maximum_evidence() {
        let make_eval = |log_evidence: f64| LambdaEvaluation {
            lambda: 1.0,
            lambda_eff: 1.0,
            rss_weighted: 1.0,
            solution_norm: 1.0,
            df: 5.0,
            gcv: 1.0,
            log_evidence,
            chi_squared: 1.0,
            solution: Solution {
                coeffs: vec![],
                coeff_err: None,
                i_calc: vec![],
                chi_squared: 1.0,
                lambda_effective: Some(1.0),
            },
        };

        // Index 2 has the highest finite evidence; index 0 is non-finite.
        let evals = vec![
            make_eval(f64::NEG_INFINITY),
            make_eval(-5.0),
            make_eval(-2.0), // winner
            make_eval(-4.0),
        ];
        assert_eq!(BayesianEvidence.select(&evals), 2);
    }

    /// Verify log_evidence stored in LambdaEvaluation matches the formula:
    ///   -½ [ RSS_w + λ_eff‖Lc‖² + log det(A + λ_eff H) − N_r log λ_eff ]
    ///
    /// Strategy: run evaluate_lambda_grid on a tiny synthetic problem, then
    /// independently recompute log det from the Cholesky of (ktk + λ_eff·ltl)
    /// and compare to the stored value.
    #[test]
    fn log_evidence_matches_formula() {
        let m = make_test_matrices();
        let n_r = m.ktk.ncols();

        let lambda = 0.01_f64;
        let evals = evaluate_lambda_grid(&[lambda], &m).unwrap();
        let ev = &evals[0];

        // Recompute log det independently from the same matrix
        let mat = &m.ktk + &m.ltl * ev.lambda_eff;
        let chol = mat
            .cholesky()
            .expect("Cholesky should succeed for this test matrix");
        let log_det = 2.0 * chol.l().diagonal().iter().map(|&x| x.ln()).sum::<f64>();

        let expected = -0.5
            * (ev.rss_weighted + ev.lambda_eff * ev.solution_norm + log_det
                - n_r as f64 * ev.lambda_eff.ln());

        assert!(ev.log_evidence.is_finite(), "log_evidence should be finite");
        assert!(
            (ev.log_evidence - expected).abs() < 1e-10,
            "log_evidence {:.6e} does not match formula {:.6e}",
            ev.log_evidence,
            expected
        );
    }

    #[test]
    fn lcurve_avoids_boundaries() {
        // A curve of 5 points — corner detection should not return index 0 or 4.
        let make_eval = |rss: f64, snorm: f64| LambdaEvaluation {
            lambda: 1.0,
            lambda_eff: 1.0,
            rss_weighted: rss,
            solution_norm: snorm,
            df: 5.0,
            gcv: 1.0,
            log_evidence: 0.0,
            chi_squared: 1.0,
            solution: Solution {
                coeffs: vec![],
                coeff_err: None,
                i_calc: vec![],
                chi_squared: 1.0,
                lambda_effective: Some(1.0),
            },
        };

        // Synthetic L-shape: residual drops steeply then flattens,
        // solution norm rises slowly then steeply.
        let evals = vec![
            make_eval(1e-8, 1e4),
            make_eval(1e-6, 1e3),
            make_eval(1e-4, 1e1), // corner here
            make_eval(1e-2, 1e-1),
            make_eval(1e0, 1e-3),
        ];
        let idx = LCurveSelector.select(&evals);
        assert!(
            idx > 0 && idx < 4,
            "L-curve corner should be interior, got {idx}"
        );
    }
}
