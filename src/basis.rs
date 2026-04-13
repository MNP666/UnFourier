use crate::bspline;
use nalgebra::DMatrix;

/// A set of basis functions for representing P(r).
///
/// A `BasisSet` defines how P(r) is parameterised — as a sum of basis
/// functions φ_j(r) with coefficients c_j:
///
///   P(r) ≈ Σ_j c_j · φ_j(r)
///
/// It also knows how to build the kernel matrix K where K_ij encodes the
/// contribution of basis function j to I(q_i):
///
///   I(q_i) = 4π ∫ P(r) sin(q_i r)/(q_i r) dr
///           ≈ Σ_j c_j · K_ij
///
/// # Implementations
///
/// - `CubicBSpline`: compact-support splines for a smooth representation
pub trait BasisSet: Send + Sync {
    /// Representative r-values for the free coefficients.
    ///
    /// For splines these are Greville abscissae of the free interior basis
    /// functions. They are useful diagnostics, but they are not sampled P(r)
    /// output positions.
    fn r_values(&self) -> &[f64];

    /// Upper bound of the support of P(r).
    fn r_max(&self) -> f64;

    /// Number of basis functions (= number of coefficients).
    fn n_basis(&self) -> usize {
        self.r_values().len()
    }

    /// Build the kernel matrix K (n_q × n_basis) for the given q values.
    ///
    /// K_ij = contribution of basis function j to I(q_i).
    fn build_kernel_matrix(&self, q: &[f64]) -> DMatrix<f64>;

    /// Dense r-grid used when writing or plotting the evaluated P(r) curve.
    fn output_grid(&self) -> Vec<f64>;

    /// Evaluate P(r) = Σ_j c_j φ_j(r) on the supplied r-grid.
    fn evaluate_pr(&self, coeffs: &[f64], r: &[f64]) -> Vec<f64>;

    /// Propagate marginal coefficient standard deviations to the output grid.
    ///
    /// This assumes independent coefficient marginals because the current M4
    /// posterior helper exposes only diagonal standard deviations. It is still
    /// preferable to writing raw coefficient σ values on a denser output grid.
    fn evaluate_pr_sigma(&self, coeff_sigma: &[f64], r: &[f64]) -> Vec<f64>;
}

// ---------------------------------------------------------------------------
// CubicBSpline — smooth spline basis (M5)
// ---------------------------------------------------------------------------

/// Cubic B-spline basis on `[0, r_max]` with zero boundary conditions.
///
/// P(r) is represented as a linear combination of `n_basis` interior cubic
/// B-spline functions.  The two endpoint B-splines (B_0 at r=0, B_{n-1} at
/// r=r_max) are **excluded** from the design matrix; their coefficients are
/// implicitly zero, which enforces P(0) = P(r_max) = 0 by construction.
///
/// ```text
/// P(r) = Σ_{j=1}^{n_basis}  c_j · B_j(r)   (B_0 and B_{n+1} dropped)
/// ```
///
/// Endpoint values P(0)=P(r_max)=0 are enforced by the parameterisation.  The
/// solved coefficients are spline control weights, not sampled P(r) values;
/// output must be produced with [`BasisSet::evaluate_pr`].
///
/// # Parameters
///
/// `n_basis` is the number of **free** (interior) parameters.
/// The underlying knot vector uses `n_basis - 2` interior knots.
/// The default recommended value is `n_basis = 20`.
pub struct CubicBSpline {
    knots: Vec<f64>,
    /// Greville abscissae of the n_basis INTERIOR basis functions only
    /// (B_1 … B_{n_basis}, excluding the two endpoint functions).
    r_interior: Vec<f64>,
    r_max: f64,
}

impl CubicBSpline {
    /// Create a cubic B-spline basis on `[0, r_max]` with `n_basis` free parameters.
    ///
    /// Interior knots are placed uniformly.  The returned basis has `n_basis`
    /// columns (interior B-splines B_1 … B_{n_basis} only; the two endpoint
    /// B-splines are excluded so that P(0)=P(r_max)=0 holds by construction).
    ///
    /// # Panics
    ///
    /// Panics if `r_max <= 0` or `n_basis < 2`.
    pub fn new(r_max: f64, n_basis: usize) -> Self {
        assert!(r_max > 0.0, "r_max must be positive");
        assert!(n_basis >= 2, "n_basis must be at least 2");

        let n_interior = n_basis - 2;
        let knots = bspline::clamped_knots(r_max, n_interior);

        // All Greville abscissae (n_basis + 2 total), then keep only the
        // interior ones (indices 1..n_basis+1), dropping the two endpoints.
        let all_greville = bspline::greville(&knots, 3);
        let r_interior = all_greville[1..all_greville.len() - 1].to_vec();

        Self {
            knots,
            r_interior,
            r_max,
        }
    }

    /// Basis matrix for the free interior spline coefficients only.
    fn free_basis_matrix(&self, r: &[f64]) -> DMatrix<f64> {
        let full = bspline::basis_matrix(&self.knots, 3, r);
        let n_cols = full.ncols();
        let n_free = n_cols - 2;
        DMatrix::from_fn(full.nrows(), n_free, |i, j| full[(i, j + 1)])
    }
}

impl BasisSet for CubicBSpline {
    fn r_values(&self) -> &[f64] {
        &self.r_interior
    }

    fn r_max(&self) -> f64 {
        self.r_max
    }

    /// K_ij = 4π ∫ B_j(r) sin(q_i r)/(q_i r) dr   for the n_basis INTERIOR
    /// basis functions (B_1 … B_{n_basis}).
    ///
    /// The two endpoint columns (B_0 and B_{n_basis+1}) are dropped because
    /// their coefficients are implicitly zero (P(0)=P(r_max)=0 by construction).
    /// The returned matrix has shape `(n_q × n_basis)`.
    fn build_kernel_matrix(&self, q: &[f64]) -> DMatrix<f64> {
        let full = bspline::sinc_kernel_matrix(&self.knots, 3, q);
        let n_cols = full.ncols(); // n_basis + 2
        let n_free = n_cols - 2; // n_basis
        DMatrix::from_fn(full.nrows(), n_free, |i, j| full[(i, j + 1)])
    }

    fn output_grid(&self) -> Vec<f64> {
        let n_intervals = (self.r_interior.len() * 10).max(200);
        (0..=n_intervals)
            .map(|i| self.r_max * i as f64 / n_intervals as f64)
            .collect()
    }

    fn evaluate_pr(&self, coeffs: &[f64], r: &[f64]) -> Vec<f64> {
        let b = self.free_basis_matrix(r);
        assert_eq!(
            coeffs.len(),
            b.ncols(),
            "coefficient count must match the free spline basis"
        );

        let c = nalgebra::DVector::from_column_slice(coeffs);
        (b * c).iter().cloned().collect()
    }

    fn evaluate_pr_sigma(&self, coeff_sigma: &[f64], r: &[f64]) -> Vec<f64> {
        let b = self.free_basis_matrix(r);
        assert_eq!(
            coeff_sigma.len(),
            b.ncols(),
            "coefficient sigma count must match the free spline basis"
        );

        (0..b.nrows())
            .map(|i| {
                (0..b.ncols())
                    .map(|j| {
                        let term = b[(i, j)] * coeff_sigma[j];
                        term * term
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// n_basis free parameters → correct kernel dimensions.
    #[test]
    fn cubic_bspline_dimensions() {
        let bs = CubicBSpline::new(150.0, 20);
        // Interior only: n_basis = 20 (endpoint B-splines dropped)
        assert_eq!(bs.n_basis(), 20);
        assert_eq!(bs.r_values().len(), 20);
        assert_eq!(bs.r_max(), 150.0);

        let q: Vec<f64> = (1..=50).map(|i| i as f64 * 0.01).collect();
        let k = bs.build_kernel_matrix(&q);
        assert_eq!(k.nrows(), 50);
        assert_eq!(k.ncols(), 20);
    }

    /// r_values() returns interior Greville abscissae only (no endpoints at 0
    /// or r_max), all strictly between 0 and r_max.
    #[test]
    fn greville_interior_only() {
        let r_max = 100.0_f64;
        let bs = CubicBSpline::new(r_max, 15);
        let r = bs.r_values();
        // Interior only: 15 points
        assert_eq!(r.len(), 15);
        // All strictly inside (0, r_max) — endpoints are no longer included
        for &xi in r {
            assert!(
                xi > 0.0 && xi < r_max,
                "Greville abscissa {xi} not strictly inside (0, {r_max})"
            );
        }
    }

    /// At q = 0, K_ij = 4π ∫ B_j(r) dr for interior B-splines.
    /// All interior B-splines have positive integrals.
    #[test]
    fn kernel_entries_positive_at_q0() {
        let bs = CubicBSpline::new(80.0, 12);
        let k = bs.build_kernel_matrix(&[0.0]);
        for j in 0..k.ncols() {
            assert!(
                k[(0, j)] > 0.0,
                "K[0,{j}] = {} at q=0 (expected > 0)",
                k[(0, j)]
            );
        }
    }

    #[test]
    fn output_grid_is_dense_and_includes_endpoints() {
        let bs = CubicBSpline::new(100.0, 12);
        let grid = bs.output_grid();

        assert_ne!(
            grid.len(),
            bs.n_basis(),
            "output grid must not be the raw coefficient grid"
        );
        assert!(grid.len() > bs.n_basis());
        assert_eq!(grid.first().copied(), Some(0.0));
        assert_eq!(grid.last().copied(), Some(100.0));
    }

    #[test]
    fn evaluated_output_endpoints_are_zero_without_mutating_coeffs() {
        let bs = CubicBSpline::new(100.0, 8);
        let coeffs: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let original = coeffs.clone();
        let r = bs.output_grid();
        let p = bs.evaluate_pr(&coeffs, &r);

        assert_eq!(coeffs, original, "evaluation must not mutate coefficients");
        assert_eq!(p.len(), r.len());
        assert!(p[0].abs() < 1e-12, "P(0) must be exactly zero");
        assert!(p[p.len() - 1].abs() < 1e-12, "P(Dmax) must be exactly zero");
        assert!(
            p.iter().any(|&v| v > 0.0),
            "non-zero coefficients should produce non-zero interior P(r)"
        );
    }

    #[test]
    fn output_evaluation_does_not_change_back_calculated_intensity() {
        use crate::kernel::back_calculate;

        let bs = CubicBSpline::new(120.0, 10);
        let coeffs: Vec<f64> = (0..10).map(|i| 0.2 + i as f64 * 0.1).collect();
        let q = vec![0.01, 0.04, 0.08, 0.12];

        let before = back_calculate(&bs, &q, &coeffs);
        let r = bs.output_grid();
        let _p = bs.evaluate_pr(&coeffs, &r);
        let after = back_calculate(&bs, &q, &coeffs);

        assert_eq!(before, after);
    }
}
