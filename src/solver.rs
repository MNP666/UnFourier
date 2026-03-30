use anyhow::{anyhow, Result};
use nalgebra::{DMatrix, DVector};
use thiserror::Error;

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
/// - `Regulariser` (M2)              → regularisation
///
/// Each call to `solve` is intentionally stateless: it takes all inputs by
/// reference and returns an owned `Solution`. This means multiple independent
/// calls (e.g., for a λ grid search in M3) can be trivially parallelised with
/// `rayon::par_iter()` when that milestone arrives.
pub trait Solver: Send + Sync {
    /// Solve for P(r) coefficients.
    ///
    /// # Arguments
    /// - `k_weighted` — weighted kernel matrix (n_q × n_basis), from `build_weighted_system`
    /// - `i_weighted` — weighted intensity vector (length n_q)
    /// - `k_unweighted` — unweighted kernel (used only for back-calculation and χ²)
    /// - `i_observed` — original (unweighted) intensities
    /// - `sigma` — measurement errors (for χ² calculation)
    /// - `r` — r-grid values from the BasisSet
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
///
/// # Limitations
///
/// With no regularisation, the solution will typically be oscillatory and
/// contain large positive and negative values. This is expected in M1 and
/// serves only to validate the end-to-end pipeline. M2 introduces Tikhonov
/// regularisation to produce physically meaningful results.
pub struct LeastSquaresSvd {
    /// Singular values below `svd_eps × σ_max` are treated as zero.
    /// The default (1e-10) is conservative; you may need to increase it for
    /// noisy data.
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

        // Compute pseudo-inverse solution via SVD: c = K_w⁺ · I_w
        let svd = k_weighted.clone().svd(true, true);
        let coeffs: DVector<f64> = svd
            .solve(&b, self.svd_eps)
            .map_err(|e| anyhow!("SVD solve failed: {}", e))?;

        let p_r: Vec<f64> = coeffs.iter().cloned().collect();

        // Back-calculate I(q) using the unweighted kernel
        let i_calc_vec = k_unweighted * &coeffs;
        let i_calc: Vec<f64> = i_calc_vec.iter().cloned().collect();

        // Reduced chi-squared
        let chi_squared = i_observed
            .iter()
            .zip(i_calc.iter())
            .zip(sigma.iter())
            .map(|((&i_obs, &i_cal), &s)| {
                let s = if s > 0.0 { s } else { 1.0 };
                ((i_obs - i_cal) / s).powi(2)
            })
            .sum::<f64>()
            / n_q as f64;

        Ok(Solution {
            r: r.to_vec(),
            p_r,
            p_r_err: None,
            i_calc,
            chi_squared,
        })
    }
}
