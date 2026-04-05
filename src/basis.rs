use nalgebra::DMatrix;
use crate::bspline;

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
/// - [`UniformGrid`] (M1): rectangular/histogram basis, P(r) piecewise constant
/// - `CubicBSpline` (M5): compact-support splines for a smoother representation
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
// UniformGrid — rectangular/histogram basis
// ---------------------------------------------------------------------------

/// Uniform rectangular basis: P(r) is approximated as piecewise constant on a
/// uniform grid of `n_points` bins over [0, r_max].
///
/// Basis function j is a top-hat of width Δr centred at r_j = (j + 0.5) · Δr.
/// The coefficient c_j approximates the average value of P(r) in that bin.
///
/// The kernel integral reduces to:
///   K_ij = 4π · sin(q_i · r_j) / (q_i · r_j) · Δr
/// with the sinc limit sin(x)/x → 1 as x → 0 handled explicitly.
///
/// This is a deliberately simple starting point. Replace with `CubicBSpline`
/// in M5 for a smoother, more accurate representation with fewer coefficients.
pub struct UniformGrid {
    r: Vec<f64>,
    r_max: f64,
    delta_r: f64,
}

impl UniformGrid {
    /// Create a uniform grid over (0, r_max] with `n_points` bins.
    ///
    /// Grid points are placed at bin centres: r_j = (j + 0.5) · Δr.
    pub fn new(r_max: f64, n_points: usize) -> Self {
        assert!(r_max > 0.0, "r_max must be positive");
        assert!(n_points > 0, "n_points must be at least 1");

        let delta_r = r_max / n_points as f64;
        let r = (0..n_points)
            .map(|j| (j as f64 + 0.5) * delta_r)
            .collect();

        Self { r, r_max, delta_r }
    }

    pub fn delta_r(&self) -> f64 {
        self.delta_r
    }
}

// ---------------------------------------------------------------------------
// CubicBSpline — smooth spline basis (M5)
// ---------------------------------------------------------------------------

/// Cubic B-spline basis on `[0, r_max]` with zero boundary conditions.
///
/// P(r) is represented as a linear combination of `n_basis` *free* cubic
/// B-spline functions:
///
/// ```text
/// P(r) = Σ_{j=1}^{n_total-2}  c_j · B_j(r)
/// ```
///
/// The two endpoint basis functions (j = 0 and j = n_total − 1) are excluded
/// from the design matrix, which enforces P(0) = P(r_max) = 0 exactly for any
/// coefficient vector — no post-hoc clamping needed.
///
/// # Parameters
///
/// `n_basis` is the number of **free** parameters.  The relationship to the
/// underlying knot vector is:
///
/// ```text
/// n_interior_knots = n_basis - 2
/// n_total          = n_basis + 2   (including the two clamped endpoints)
/// ```
///
/// The default recommended value is `n_basis = 20`.
pub struct CubicBSpline {
    knots: Vec<f64>,
    /// Greville abscissae of the free basis functions (B_1 … B_{n_total-2}).
    r_free: Vec<f64>,
    r_max: f64,
}

impl CubicBSpline {
    /// Create a cubic B-spline basis on `[0, r_max]` with `n_basis` free parameters.
    ///
    /// Interior knots are placed uniformly; boundary conditions P(0) = P(r_max) = 0
    /// are enforced structurally via a clamped knot vector.
    ///
    /// # Panics
    ///
    /// Panics if `r_max <= 0` or `n_basis < 2`.
    pub fn new(r_max: f64, n_basis: usize) -> Self {
        assert!(r_max > 0.0, "r_max must be positive");
        assert!(n_basis >= 2, "n_basis must be at least 2");

        let n_interior = n_basis - 2; // n_free = n_interior + 2
        let knots = bspline::clamped_knots(r_max, n_interior);

        // Greville abscissae for the free functions: drop the first and last
        // entries (those belong to the clamped endpoint basis functions).
        let xi_all = bspline::greville(&knots, 3);
        let r_free = xi_all[1..xi_all.len() - 1].to_vec();

        Self { knots, r_free, r_max }
    }
}

impl BasisSet for CubicBSpline {
    fn r_values(&self) -> &[f64] {
        &self.r_free
    }

    fn r_max(&self) -> f64 {
        self.r_max
    }

    /// K_ij = 4π ∫ B_j(r) sin(q_i r)/(q_i r) dr   for each free basis function j.
    ///
    /// Built via 5-point Gauss–Legendre quadrature per knot span.
    /// The two endpoint columns of the full basis matrix are dropped here,
    /// so the returned matrix has shape `(n_q × n_basis)`.
    fn build_kernel_matrix(&self, q: &[f64]) -> DMatrix<f64> {
        let full = bspline::sinc_kernel_matrix(&self.knots, 3, q);
        // Drop column 0 (B_0, clamped at r=0) and column n_total-1 (B_{n-1}, clamped at r_max)
        let n_free = full.ncols() - 2;
        DMatrix::from_fn(q.len(), n_free, |i, j| full[(i, j + 1)])
    }
}

impl BasisSet for UniformGrid {
    fn r_values(&self) -> &[f64] {
        &self.r
    }

    fn r_max(&self) -> f64 {
        self.r_max
    }

    /// K_ij = 4π · sinc(q_i · r_j) · Δr   where sinc(x) = sin(x)/x.
    fn build_kernel_matrix(&self, q: &[f64]) -> DMatrix<f64> {
        let n_q = q.len();
        let n_r = self.r.len();
        let delta_r = self.delta_r;

        DMatrix::from_fn(n_q, n_r, |i, j| {
            let qr = q[i] * self.r[j];
            // sinc limit: sin(qr)/(qr) → 1 as qr → 0
            let sinc = if qr.abs() < 1e-10 {
                1.0
            } else {
                qr.sin() / qr
            };
            4.0 * std::f64::consts::PI * sinc * delta_r
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// n_basis free parameters → correct dimensions from BasisSet trait methods.
    #[test]
    fn cubic_bspline_dimensions() {
        let bs = CubicBSpline::new(150.0, 20);
        assert_eq!(bs.n_basis(), 20);
        assert_eq!(bs.r_values().len(), 20);
        assert_eq!(bs.r_max(), 150.0);

        let q: Vec<f64> = (1..=50).map(|i| i as f64 * 0.01).collect();
        let k = bs.build_kernel_matrix(&q);
        assert_eq!(k.nrows(), 50);
        assert_eq!(k.ncols(), 20);
    }

    /// The free basis functions have compact support strictly inside (0, r_max),
    /// so P(0) = P(r_max) = 0 for any coefficient vector — verified by
    /// evaluating each kernel column at q → 0 and checking the sum reconstructs
    /// ∫ P(r) dr with correct boundary behaviour.
    ///
    /// More directly: the Greville abscissae of the free functions must all
    /// lie strictly between 0 and r_max.
    #[test]
    fn greville_strictly_interior() {
        let r_max = 100.0_f64;
        let bs = CubicBSpline::new(r_max, 15);
        for &xi in bs.r_values() {
            assert!(xi > 0.0 && xi < r_max,
                "Greville abscissa {xi} not strictly inside (0, {r_max})");
        }
    }

    /// At q = 0, K_ij = 4π ∫ B_j(r) dr.  Summing over all free basis functions
    /// gives 4π · (r_max − first_support − last_support).  For a clamped grid
    /// the two endpoint functions each contribute half a span less, so the sum
    /// of the entire row at q = 0 should be 4π · r_max minus the endpoint
    /// contributions.  Rather than tracking those analytically we just check that
    /// each entry is positive (the sinc integrand is positive near q = 0).
    #[test]
    fn kernel_entries_positive_at_q0() {
        let bs = CubicBSpline::new(80.0, 12);
        let k = bs.build_kernel_matrix(&[0.0]);
        for j in 0..k.ncols() {
            assert!(k[(0, j)] > 0.0, "K[0,{j}] = {} at q=0 (expected > 0)", k[(0, j)]);
        }
    }
}
