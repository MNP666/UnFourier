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

use crate::nonneg::{IterativeClipping, NonNegativityStrategy};
use crate::regularise::SecondDerivative;
use crate::regularise::Regulariser;
use crate::solver::Solution;

// ---------------------------------------------------------------------------
// LambdaEvaluation
// ---------------------------------------------------------------------------

/// All quantities computed for a single λ candidate.
///
/// Storing everything here (rather than just the score) means the caller can
/// inspect diagnostics and plot the full L-curve without re-running solves.
/// The `solution` field contains the non-negativity-enforced P(r) — this is
/// what gets used if this λ is selected.
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

    /// Reduced χ² from the non-negativity-enforced solution (for display).
    pub chi_squared: f64,

    /// The non-negativity-enforced P(r) solution (ready to use if λ selected).
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

    /// r-grid values from the BasisSet
    pub r: Vec<f64>,
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
    /// - `r`            — r-grid values        (n_r)
    pub fn build(
        k_weighted: &DMatrix<f64>,
        i_weighted: &[f64],
        k_unweighted: &DMatrix<f64>,
        i_observed: &[f64],
        sigma: &[f64],
        r: &[f64],
    ) -> Self {
        let n_r = k_weighted.ncols();
        let reg = SecondDerivative;
        let ltl = reg.gram_matrix(n_r);

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
            r: r.to_vec(),
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
/// 4. Apply non-negativity (iterative clipping) to get the final P(r).
/// 5. Compute χ² from the non-negativity-enforced solution (for display).
///
/// The unconstrained solve is used for λ selection because GCV and the L-curve
/// assume a linear estimator. Non-negativity makes the estimator nonlinear and
/// invalidates the GCV formula. The constrained solution is still used for the
/// final output once the best λ is selected.
///
/// Each evaluation is fully independent — this loop is a natural candidate for
/// `rayon::par_iter()` when parallelism is added in a future milestone.
pub fn evaluate_lambda_grid(
    lambdas: &[f64],
    m: &GridMatrices,
) -> Result<Vec<LambdaEvaluation>> {
    let n_q = m.i_observed.len();
    let n_r = m.r.len();

    let mut evaluations = Vec::with_capacity(lambdas.len());
    let nonneg = IterativeClipping;

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
        let p_r = apply_nonneg_tikhonov(&m.ktk, &m.kti, &m.ltl, lambda_eff, &nonneg, n_r)?;

        // Back-calculate I(q) and χ² from constrained solution
        let coeffs = DVector::from_column_slice(&p_r);
        let i_calc: Vec<f64> = (&m.k_unweighted * &coeffs).iter().cloned().collect();
        let chi_squared = reduced_chi_squared(&m.i_observed, &i_calc, &m.sigma);

        evaluations.push(LambdaEvaluation {
            lambda,
            lambda_eff,
            rss_weighted,
            solution_norm,
            df,
            gcv,
            chi_squared,
            solution: Solution {
                r: m.r.clone(),
                p_r,
                p_r_err: None,
                i_calc,
                chi_squared,
                lambda_effective: Some(lambda_eff),
            },
        });
    }

    Ok(evaluations)
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

/// Apply iterative non-negativity clipping to the Tikhonov solution.
///
/// Used for the final P(r) once λ is selected. The unconstrained solution used
/// for λ selection is computed separately (see [`evaluate_lambda_grid`]).
fn apply_nonneg_tikhonov(
    ktk: &DMatrix<f64>,
    kti: &DVector<f64>,
    ltl: &DMatrix<f64>,
    lambda_eff: f64,
    nonneg: &dyn NonNegativityStrategy,
    n_r: usize,
) -> Result<Vec<f64>> {
    let max_iter = 30usize;
    let mut fixed: Vec<usize> = vec![];
    let mut p_r = vec![0.0_f64; n_r];

    for _ in 0..=max_iter {
        p_r = solve_active(ktk, kti, ltl, lambda_eff, &fixed, n_r)?;
        let violations = nonneg.find_violations(&p_r);
        if violations.is_empty() {
            break;
        }
        fixed.extend(violations);
        fixed.sort_unstable();
        fixed.dedup();
    }

    Ok(p_r)
}

/// Solve (KᵀK_a + λ LᵀL_a) c_a = KᵀI_a on the active index set.
///
/// Mirrors [`crate::solver::solve_tikhonov_active`] but operates on pre-computed
/// matrices rather than rebuilding them — safe to call in a tight loop.
fn solve_active(
    ktk: &DMatrix<f64>,
    kti: &DVector<f64>,
    ltl: &DMatrix<f64>,
    lambda: f64,
    fixed: &[usize],
    n_basis: usize,
) -> Result<Vec<f64>> {
    let active: Vec<usize> = (0..n_basis).filter(|j| !fixed.contains(j)).collect();
    let n_active = active.len();

    if n_active == 0 {
        return Ok(vec![0.0; n_basis]);
    }

    let ktk_a = DMatrix::from_fn(n_active, n_active, |i, j| ktk[(active[i], active[j])]);
    let ltl_a = DMatrix::from_fn(n_active, n_active, |i, j| ltl[(active[i], active[j])]);
    let kti_a = DVector::from_fn(n_active, |i, _| kti[active[i]]);

    let a = ktk_a + ltl_a * lambda;

    let c_active: DVector<f64> = if let Some(chol) = a.clone().cholesky() {
        chol.solve(&kti_a)
    } else {
        a.lu()
            .solve(&kti_a)
            .ok_or_else(|| anyhow!("Active-set Tikhonov solve is singular"))?
    };

    let mut coeffs = vec![0.0_f64; n_basis];
    for (k, &j) in active.iter().enumerate() {
        coeffs[j] = c_active[k];
    }
    Ok(coeffs)
}

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
            chi_squared: 1.0,
            solution: Solution {
                r: vec![],
                p_r: vec![],
                p_r_err: None,
                i_calc: vec![],
                chi_squared: 1.0,
                lambda_effective: Some(1.0),
            },
        };

        let evals = vec![make_eval(f64::INFINITY), make_eval(0.5), make_eval(1.0)];
        assert_eq!(GcvSelector.select(&evals), 1); // index 1 has minimum finite GCV
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
            chi_squared: 1.0,
            solution: Solution {
                r: vec![],
                p_r: vec![],
                p_r_err: None,
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
        assert!(idx > 0 && idx < 4, "L-curve corner should be interior, got {idx}");
    }
}
