use anyhow::{anyhow, Result};
use nalgebra::{DMatrix, DVector};
use thiserror::Error;

use crate::nonneg::{projected_gradient_nnls, NonNegativityStrategy, ProjectedGradient};
use crate::regularise::{Regulariser, SecondDerivative};

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("SVD decomposition failed to converge")]
    SvdFailed,
    #[error("kernel matrix dimensions ({kernel_rows}×{kernel_cols}) are incompatible with data length ({data_len})")]
    DimensionMismatch {
        kernel_rows: usize,
        kernel_cols: usize,
        data_len: usize,
    },
}

// ---------------------------------------------------------------------------
// Solution
// ---------------------------------------------------------------------------

/// The result of an IFT solve.
#[derive(Debug, Clone)]
pub struct Solution {
    /// r values at which P(r) is evaluated (one per basis function).
    pub r: Vec<f64>,

    /// P(r) coefficients. May contain negative values in M1 (unregularised).
    /// From M2 onwards, a non-negativity constraint is applied.
    pub p_r: Vec<f64>,

    /// Uncertainty on P(r) (one σ per r value).
    /// `None` until M4 adds Bayesian posterior covariance estimates.
    pub p_r_err: Option<Vec<f64>>,

    /// Back-calculated I(q) from the solution: I_calc = K · c.
    pub i_calc: Vec<f64>,

    /// Reduced chi-squared: Σ[(I_obs - I_calc)² / σ²] / N_q.
    pub chi_squared: f64,

    /// The effective λ actually used in the solve, after internal scaling.
    /// `None` for unregularised solvers. Useful for diagnostics and for
    /// comparing λ values across datasets with different intensity scales.
    pub lambda_effective: Option<f64>,
}

// ---------------------------------------------------------------------------
// Solver trait
// ---------------------------------------------------------------------------

/// Solves for the P(r) coefficient vector given a (possibly weighted) kernel
/// matrix and the (possibly weighted) intensity vector.
///
/// # Design notes
///
/// The solver receives the *already-weighted* system `(K_w, I_w)` from
/// [`crate::kernel::build_weighted_system`], so it only needs to handle the
/// unweighted optimisation problem internally. This keeps each component
/// responsible for one concern:
///
/// - `kernel::build_weighted_system` → error weighting
/// - `Solver`                        → linear algebra / optimisation
/// - `Regulariser`                   → regularisation penalty matrix
///
/// Each call to `solve` is intentionally stateless: it takes all inputs by
/// reference and returns an owned `Solution`. This means multiple independent
/// calls (e.g., for a λ grid search in M3) can be trivially parallelised with
/// `rayon::par_iter()` without any changes to this interface.
pub trait Solver: Send + Sync {
    /// Solve for P(r) coefficients.
    ///
    /// # Arguments
    /// - `k_weighted`   — weighted kernel matrix (n_q × n_basis)
    /// - `i_weighted`   — weighted intensity vector (length n_q)
    /// - `k_unweighted` — unweighted kernel (for back-calculation and χ²)
    /// - `i_observed`   — original (unweighted) intensities
    /// - `sigma`        — measurement errors (for χ² calculation)
    /// - `r`            — r-grid values from the BasisSet
    fn solve(
        &self,
        k_weighted: &DMatrix<f64>,
        i_weighted: &[f64],
        k_unweighted: &DMatrix<f64>,
        i_observed: &[f64],
        sigma: &[f64],
        r: &[f64],
    ) -> Result<Solution>;
}

// ---------------------------------------------------------------------------
// LeastSquaresSvd — unregularised SVD solve (M1)
// ---------------------------------------------------------------------------

/// Unregularised least-squares solver via truncated SVD.
///
/// Minimises ‖K_w · c − I_w‖² (with error weighting already applied).
/// Results will be oscillatory — this solver exists to validate the pipeline,
/// not to produce physically meaningful P(r).
pub struct LeastSquaresSvd {
    /// Singular values below `svd_eps × σ_max` are treated as zero.
    pub svd_eps: f64,
}

impl Default for LeastSquaresSvd {
    fn default() -> Self {
        Self { svd_eps: 1e-10 }
    }
}

impl LeastSquaresSvd {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_eps(svd_eps: f64) -> Self {
        Self { svd_eps }
    }
}

impl Solver for LeastSquaresSvd {
    fn solve(
        &self,
        k_weighted: &DMatrix<f64>,
        i_weighted: &[f64],
        k_unweighted: &DMatrix<f64>,
        i_observed: &[f64],
        sigma: &[f64],
        r: &[f64],
    ) -> Result<Solution> {
        let (n_q, n_r) = (k_weighted.nrows(), k_weighted.ncols());

        if n_q != i_weighted.len() {
            return Err(anyhow!(SolverError::DimensionMismatch {
                kernel_rows: n_q,
                kernel_cols: n_r,
                data_len: i_weighted.len(),
            }));
        }

        let b = DVector::from_column_slice(i_weighted);
        let svd = k_weighted.clone().svd(true, true);
        let coeffs: DVector<f64> = svd
            .solve(&b, self.svd_eps)
            .map_err(|e| anyhow!("SVD solve failed: {}", e))?;

        let p_r: Vec<f64> = coeffs.iter().cloned().collect();
        let i_calc: Vec<f64> = (k_unweighted * &coeffs).iter().cloned().collect();
        let chi_squared = reduced_chi_squared(i_observed, &i_calc, sigma);

        Ok(Solution {
            r: r.to_vec(),
            p_r,
            p_r_err: None,
            i_calc,
            chi_squared,
            lambda_effective: None,
        })
    }
}

// ---------------------------------------------------------------------------
// TikhonovSolver — regularised solve (M2)
// ---------------------------------------------------------------------------

/// Tikhonov-regularised solver.
///
/// Minimises: ‖K_w · c − I_w‖² + λ_eff · ‖L · c‖²
///
/// where λ_eff = λ_user · tr(KᵀK) / tr(LᵀL).
///
/// # Why the internal scaling?
///
/// The error weights `w_i = 1/σ_i` (with σ_i = I(q_i)/k) can reach ~1e8 at
/// high q where I(q) decays steeply, making `tr(KᵀK)` many orders of magnitude
/// larger than `tr(LᵀL) ≈ 6`. Without scaling, the user's λ would need to be
/// ~1e12 to have any effect — clearly unusable. The trace-ratio scaling makes
/// λ = 1 correspond to equal weight for data and regularisation terms (in the
/// trace-norm sense), and useful values typically fall in [1e-4, 10].
///
/// The `Solution` field `lambda_effective` records the actual λ_eff used, which
/// is useful for comparing runs across datasets with different intensity scales.
pub struct TikhonovSolver {
    /// User-facing regularisation strength (internally scaled before use).
    pub lambda: f64,

    /// Penalty matrix strategy. Default: `SecondDerivative` (penalises curvature).
    pub regulariser: Box<dyn Regulariser>,

    /// Non-negativity enforcement. Default: `IterativeClipping`.
    pub nonneg: Box<dyn NonNegativityStrategy>,

    /// Maximum iterations for the non-negativity loop.
    pub max_nonneg_iter: usize,
}

impl TikhonovSolver {
    /// Construct with `SecondDerivative` regularisation and `ProjectedGradient`.
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            regulariser: Box::new(SecondDerivative),
            nonneg: Box::new(ProjectedGradient::default()),
            max_nonneg_iter: 500,
        }
    }

    /// Construct without non-negativity enforcement (useful for debugging).
    pub fn without_nonneg(lambda: f64) -> Self {
        use crate::nonneg::NoConstraint;
        Self {
            lambda,
            regulariser: Box::new(SecondDerivative),
            nonneg: Box::new(NoConstraint),
            max_nonneg_iter: 0,
        }
    }
}

impl Solver for TikhonovSolver {
    fn solve(
        &self,
        k_weighted: &DMatrix<f64>,
        i_weighted: &[f64],
        k_unweighted: &DMatrix<f64>,
        i_observed: &[f64],
        sigma: &[f64],
        r: &[f64],
    ) -> Result<Solution> {
        let (n_q, n_r) = (k_weighted.nrows(), k_weighted.ncols());

        if n_q != i_weighted.len() {
            return Err(anyhow!(SolverError::DimensionMismatch {
                kernel_rows: n_q,
                kernel_cols: n_r,
                data_len: i_weighted.len(),
            }));
        }

        let i_w_vec = DVector::from_column_slice(i_weighted);

        // Pre-compute KᵀK and KᵀI.
        let ktk = k_weighted.transpose() * k_weighted;
        let kti = k_weighted.transpose() * &i_w_vec;

        // Pre-compute LᵀL.
        let ltl = self.regulariser.gram_matrix(n_r);

        // Scale λ so that the regularisation term is comparable to the data term
        // regardless of intensity scale and error magnitudes. See struct docs.
        let trace_ratio = ktk.trace() / ltl.trace().max(1e-10);
        let lambda_eff = self.lambda * trace_ratio;

        // Full system matrix A = KᵀK + λ_eff LᵀL.
        let a_full = &ktk + &ltl * lambda_eff;

        // Unconstrained solve (warm start for projected gradient).
        let c_unc = solve_unconstrained(&a_full, &kti)?;

        // Apply non-negativity constraint (or not).
        let p_r: Vec<f64> = if !self.nonneg.is_constraining() {
            c_unc.iter().cloned().collect()
        } else {
            // Projected gradient NNLS: converges to the true constrained minimum
            // without the cascade-zeroing problem of iterative clipping.
            // Each call is independent of other λ evaluations → rayon::par_iter() ready.
            projected_gradient_nnls(&a_full, &kti, &c_unc, self.max_nonneg_iter, 1e-8)
        };

        let coeffs = DVector::from_column_slice(&p_r);
        let i_calc: Vec<f64> = (k_unweighted * &coeffs).iter().cloned().collect();
        let chi_squared = reduced_chi_squared(i_observed, &i_calc, sigma);

        Ok(Solution {
            r: r.to_vec(),
            p_r,
            p_r_err: None,
            i_calc,
            chi_squared,
            lambda_effective: Some(lambda_eff),
        })
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Solve Ac = b for the unconstrained Tikhonov system.
///
/// Tries Cholesky first (optimal for symmetric positive-definite A); falls
/// back to LU if Cholesky fails (e.g. near-singular systems at very small λ).
fn solve_unconstrained(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>> {
    if let Some(chol) = a.clone().cholesky() {
        Ok(chol.solve(b))
    } else {
        a.clone()
            .lu()
            .solve(b)
            .ok_or_else(|| anyhow!("Tikhonov solve failed: system is singular"))
    }
}

/// Reduced chi-squared: Σ[(I_obs − I_calc)² / σ²] / N.
fn reduced_chi_squared(i_observed: &[f64], i_calc: &[f64], sigma: &[f64]) -> f64 {
    let n = i_observed.len();
    if n == 0 {
        return 0.0;
    }
    i_observed
        .iter()
        .zip(i_calc.iter())
        .zip(sigma.iter())
        .map(|((&i_obs, &i_cal), &s)| {
            let s = if s > 0.0 { s } else { 1.0 };
            ((i_obs - i_cal) / s).powi(2)
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
    use nalgebra::DMatrix;

    #[test]
    fn tikhonov_nnls_3x3_clamps_negative_bins_to_zero() {
        // K = I (3×3 identity), λ = 0.
        // Weighted system is the same as unweighted when σ = [1, 1, 1].
        // With i_weighted = [0.1, 2.0, -0.5]:
        //   unconstrained c = [0.1, 2.0, -0.5]
        //   NNLS solution   = [0.1, 2.0, 0.0]  (last bin clamped)
        let k = DMatrix::<f64>::identity(3, 3);
        let i_w = vec![0.1_f64, 2.0, -0.5];
        let sigma = vec![1.0_f64; 3];
        let r = vec![10.0_f64, 20.0, 30.0];

        let solver = TikhonovSolver::new(0.0);
        let sol = solver
            .solve(&k, &i_w, &k, &i_w, &sigma, &r)
            .expect("solve must succeed");

        assert_eq!(sol.p_r.len(), 3);
        assert!(
            sol.p_r.iter().all(|&x| x >= 0.0),
            "all P(r) values must be non-negative; got {:?}",
            sol.p_r
        );
        assert!(
            (sol.p_r[0] - 0.1).abs() < 1e-5,
            "p_r[0] should be ~0.1, got {}",
            sol.p_r[0]
        );
        assert!(
            (sol.p_r[1] - 2.0).abs() < 1e-5,
            "p_r[1] should be ~2.0, got {}",
            sol.p_r[1]
        );
        assert!(
            sol.p_r[2].abs() < 1e-5,
            "p_r[2] should be ~0.0, got {}",
            sol.p_r[2]
        );
    }
}
