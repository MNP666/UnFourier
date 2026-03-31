use anyhow::{anyhow, Result};
use nalgebra::{DMatrix, DVector};
use thiserror::Error;

use crate::nonneg::{IterativeClipping, NonNegativityStrategy};
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
    /// Construct with `SecondDerivative` regularisation and `IterativeClipping`.
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            regulariser: Box::new(SecondDerivative),
            nonneg: Box::new(IterativeClipping),
            max_nonneg_iter: 20,
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

        // Pre-compute KᵀK and KᵀI once; both are reused across non-negativity
        // iterations without rebuilding from the active columns each time.
        let ktk = k_weighted.transpose() * k_weighted;
        let kti = k_weighted.transpose() * &i_w_vec;

        // Pre-compute LᵀL.
        let ltl = self.regulariser.gram_matrix(n_r);

        // Scale λ so that the regularisation term is comparable to the data term
        // regardless of intensity scale and error magnitudes. See struct docs.
        let trace_ratio = ktk.trace() / ltl.trace().max(1e-10);
        let lambda_eff = self.lambda * trace_ratio;

        // Non-negativity loop: accumulate fixed-to-zero indices and re-solve.
        let mut fixed: Vec<usize> = vec![];
        let mut p_r = vec![0.0_f64; n_r];

        for _ in 0..=self.max_nonneg_iter {
            p_r = solve_tikhonov_active(&ktk, &kti, &ltl, lambda_eff, &fixed, n_r)?;
            let violations = self.nonneg.find_violations(&p_r);
            if violations.is_empty() {
                break;
            }
            fixed.extend(violations);
            fixed.sort_unstable();
            fixed.dedup();
        }

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

/// Solve (KᵀK_a + λ · LᵀL_a) c_a = KᵀI_a restricted to the active index set.
///
/// Takes pre-computed `ktk` (n_basis × n_basis) and `kti` (n_basis) so that
/// the matrix products are not recomputed on every non-negativity iteration.
/// Returns a full-length vector with zeros at fixed positions.
fn solve_tikhonov_active(
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

    // Extract [active × active] submatrices and the active subvector of KᵀI.
    let ktk_a = DMatrix::from_fn(n_active, n_active, |i, j| ktk[(active[i], active[j])]);
    let ltl_a = DMatrix::from_fn(n_active, n_active, |i, j| ltl[(active[i], active[j])]);
    let kti_a = DVector::from_fn(n_active, |i, _| kti[active[i]]);

    let a = ktk_a + ltl_a * lambda;

    // Cholesky is the natural choice for symmetric positive-definite systems.
    // Fall back to LU if needed (e.g. nearly-singular active set).
    let c_active: DVector<f64> = if let Some(chol) = a.clone().cholesky() {
        chol.solve(&kti_a)
    } else {
        a.lu()
            .solve(&kti_a)
            .ok_or_else(|| anyhow!("Tikhonov solve failed: system is singular"))?
    };

    let mut coeffs = vec![0.0_f64; n_basis];
    for (k, &j) in active.iter().enumerate() {
        coeffs[j] = c_active[k];
    }
    Ok(coeffs)
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
