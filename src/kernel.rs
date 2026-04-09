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

/// Append two soft equality constraint rows to the weighted system:
///   [weight, 0, …, 0] · c ≈ 0   (forces first coefficient → 0, i.e. P(r=0) = 0)
///   [0, …, 0, weight] · c ≈ 0   (forces last  coefficient → 0, i.e. P(r=D_max) = 0)
///
/// Both `k_w` and `i_w` grow by 2 rows/elements in-place.
/// These rows flow through GCV/L-curve/Bayes unchanged — they look like
/// two additional data points with target value zero.
pub fn append_boundary_constraints(
    k_w: &mut DMatrix<f64>,
    i_w: &mut Vec<f64>,
    weight: f64,
) {
    let n_rows = k_w.nrows();
    let n_cols = k_w.ncols();

    let mut new_k = DMatrix::zeros(n_rows + 2, n_cols);
    new_k.rows_mut(0, n_rows).copy_from(k_w);
    // Row n_rows: P(r=0)    → first coefficient = 0
    new_k[(n_rows, 0)] = weight;
    // Row n_rows+1: P(r=Dmax) → last coefficient = 0
    new_k[(n_rows + 1, n_cols - 1)] = weight;

    *k_w = new_k;
    i_w.push(0.0);
    i_w.push(0.0);
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn boundary_constraint_shape_and_values() {
        // 3×2 matrix: 3 data points, 2 basis functions
        let mut k = DMatrix::<f64>::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]);
        let mut i_w = vec![7.0, 8.0, 9.0];

        append_boundary_constraints(&mut k, &mut i_w, 10.0);

        // Shape grows to 5×2
        assert_eq!(k.nrows(), 5);
        assert_eq!(k.ncols(), 2);

        // Original rows unchanged
        assert_eq!(k[(0, 0)], 1.0);
        assert_eq!(k[(2, 1)], 6.0);

        // Constraint row for P(r=0): [weight, 0]
        assert_eq!(k[(3, 0)], 10.0);
        assert_eq!(k[(3, 1)], 0.0);

        // Constraint row for P(r=D_max): [0, weight]
        assert_eq!(k[(4, 0)], 0.0);
        assert_eq!(k[(4, 1)], 10.0);

        // i_w grows by 2 zeros
        assert_eq!(i_w.len(), 5);
        assert_eq!(i_w[3], 0.0);
        assert_eq!(i_w[4], 0.0);
    }

    /// End-to-end: a 5-interior-bin rect basis (7 total with boundary ghosts)
    /// with a large boundary weight should drive the boundary ghost coefficients
    /// (indices 0 and 6) to essentially zero.
    #[test]
    fn large_boundary_weight_clamps_edge_bins() {
        use crate::basis::{BasisSet, UniformGrid};
        use crate::data::SaxsData;
        use crate::solver::{Solver, TikhonovSolver};

        // Build a 5-interior-bin rect basis over [0, 5] Å.
        // After the boundary-ghost change the basis has 7 grid points total.
        let r_max = 5.0_f64;
        let n_interior = 5_usize;
        let n_total = n_interior + 2; // 7: ghost at 0, 5 interior bins, ghost at r_max
        let basis = UniformGrid::new(r_max, n_interior);
        assert_eq!(basis.n_basis(), n_total);

        // Synthetic q grid: 30 evenly-spaced q values from 0.05 to 0.5 Å⁻¹.
        let n_q = 30_usize;
        let q: Vec<f64> = (0..n_q)
            .map(|i| 0.05 + i as f64 * (0.45 / (n_q - 1) as f64))
            .collect();

        // Forward-calculate a reference I(q) from interior bins = 1, boundary ghosts = 0.
        // Ghost bins have zero kernel contribution so setting them to 0 is correct.
        let k_raw = basis.build_kernel_matrix(&q);
        let mut c_flat = nalgebra::DVector::from_element(n_total, 1.0_f64);
        c_flat[0] = 0.0;           // boundary ghost — zero kernel, pinned to 0
        c_flat[n_total - 1] = 0.0; // boundary ghost — zero kernel, pinned to 0
        let i_ref = (k_raw.clone() * c_flat).iter().cloned().collect::<Vec<_>>();

        // Uniform small errors (σ = 0.01).
        let sigma = vec![0.01_f64; n_q];

        let data = SaxsData { q, intensity: i_ref, error: sigma };

        // Build weighted system and apply a large boundary weight.
        let (mut kw, mut iw) = build_weighted_system(&basis, &data);
        let rms_inv_sigma = (data.error.iter().map(|s| 1.0 / s / s).sum::<f64>()
            / data.error.len() as f64)
            .sqrt();
        let w_large = 1e3 * (data.len() as f64).sqrt() * rms_inv_sigma;
        append_boundary_constraints(&mut kw, &mut iw, w_large);

        // Use a tiny λ so Tikhonov regularisation is weak relative to the
        // boundary constraint.
        let solver = TikhonovSolver::new(1e-6);
        let sol = solver
            .solve(&kw, &iw, &k_raw, &data.intensity, &data.error, basis.r_values())
            .unwrap();

        assert_eq!(sol.p_r.len(), n_total);
        let peak = sol.p_r.iter().cloned().fold(0.0_f64, f64::max);
        assert!(peak > 0.0, "solution is all-zero");

        let threshold = 1e-3 * peak;
        assert!(
            sol.p_r[0] < threshold,
            "p_r[0] (ghost r=0) = {:.3e} is not ≈ 0 (peak={:.3e})",
            sol.p_r[0],
            peak
        );
        assert!(
            sol.p_r[n_total - 1] < threshold,
            "p_r[n_total-1] (ghost r=r_max) = {:.3e} is not ≈ 0 (peak={:.3e})",
            sol.p_r[n_total - 1],
            peak
        );
    }
}
