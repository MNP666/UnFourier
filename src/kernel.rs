use crate::basis::BasisSet;
use crate::data::SaxsData;
use nalgebra::DMatrix;

/// Build the weighted kernel matrix W·K, where:
///   K_ij  = contribution of basis function j to I(q_i)  [from BasisSet]
///   W     = diag(1/σ_i)  (row weights from measurement errors)
///
/// The weighted system W·K·c ≈ W·I minimises the chi-squared objective:
///   χ² = Σ_i [(I(q_i) - (Kc)_i) / σ_i]²
///
/// Returns (K_weighted, I_weighted) ready to pass to a `Solver`.
///
/// Keeping the weighting here (rather than inside the solver) means the
/// solver only sees a standard unweighted least-squares problem, which
/// makes it easier to swap solvers and regularisers independently.
pub fn build_weighted_system(
    basis: &dyn BasisSet,
    data: &SaxsData,
) -> (DMatrix<f64>, Vec<f64>) {
    let k = basis.build_kernel_matrix(&data.q);
    let n_q = data.q.len();

    // Compute row weights w_i = 1/σ_i.
    // If σ_i ≤ 0 (malformed data), fall back to unit weight with a warning.
    let weights: Vec<f64> = data
        .error
        .iter()
        .enumerate()
        .map(|(i, &sigma)| {
            if sigma > 0.0 {
                1.0 / sigma
            } else {
                eprintln!(
                    "warning: non-positive error σ={} at q[{}]={:.4e}; using unit weight",
                    sigma, i, data.q[i]
                );
                1.0
            }
        })
        .collect();

    // Apply weights row-by-row: K_w[i,j] = w_i * K[i,j]
    let k_weighted = DMatrix::from_fn(n_q, k.ncols(), |i, j| k[(i, j)] * weights[i]);

    // Apply same weights to the intensity vector
    let i_weighted: Vec<f64> = data
        .intensity
        .iter()
        .zip(weights.iter())
        .map(|(&i_val, &w)| i_val * w)
        .collect();

    (k_weighted, i_weighted)
}

/// Back-calculate I(q) from a coefficient vector and the (unweighted) kernel.
///
/// I_calc = K · c
pub fn back_calculate(basis: &dyn BasisSet, q: &[f64], coeffs: &[f64]) -> Vec<f64> {
    let k = basis.build_kernel_matrix(q);
    let c = nalgebra::DVector::from_column_slice(coeffs);
    let i_calc = k * c;
    i_calc.iter().cloned().collect()
}
