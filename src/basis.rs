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
    /// The r-values at which the basis is centred (one per coefficient).
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
/// Boundary conditions P(0)=P'(0)=P(r_max)=P'(r_max)=0 are ensured by the
/// boundary-anchored regulariser (which penalises slopes from/to the implicit
/// zero boundary) together with a post-solve hard projection of the
/// boundary-adjacent coefficients to zero.
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
}
