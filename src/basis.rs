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
/// uniform grid of `n_points` interior bins over (0, r_max), plus two explicit
/// boundary points at r = 0 and r = r_max.
///
/// The full r-grid has `n_points + 2` entries:
///   r[0]          = 0        (boundary ghost — zero kernel contribution)
///   r[1..n+1]     = bin centres (j + 0.5) · Δr  (interior bins, kernel Δr·sinc)
///   r[n+1]        = r_max   (boundary ghost — zero kernel contribution)
///
/// The two boundary ghosts are pinned to zero by `append_boundary_constraints`.
/// Because the regulariser operates on all n+2 coefficients, the first-derivative
/// penalty now includes the jump from c[0]=0 to c[1] and from c[n] to c[n+1]=0,
/// which eliminates the discontinuity artefact at both boundaries.
///
/// This is a deliberately simple starting point. Replace with `CubicBSpline`
/// in M5 for a smoother, more accurate representation with fewer coefficients.
pub struct UniformGrid {
    r: Vec<f64>,
    r_max: f64,
    delta_r: f64,
}

impl UniformGrid {
    /// Create a uniform grid with `n_points` interior bins plus boundary ghosts.
    ///
    /// Interior bin centres: r_j = (j + 0.5) · Δr  for j = 0 … n_points−1.
    /// The returned basis has `n_points + 2` r-values and kernel columns.
    pub fn new(r_max: f64, n_points: usize) -> Self {
        assert!(r_max > 0.0, "r_max must be positive");
        assert!(n_points > 0, "n_points must be at least 1");

        let delta_r = r_max / n_points as f64;
        let mut r = Vec::with_capacity(n_points + 2);
        r.push(0.0);
        for j in 0..n_points {
            r.push((j as f64 + 0.5) * delta_r);
        }
        r.push(r_max);

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
/// P(r) is represented as a linear combination of `n_free + 2` cubic B-spline
/// functions, where the two endpoint functions (B_0 at r=0, B_{n-1} at r=r_max)
/// are included in the design matrix but pinned to zero by boundary constraints:
///
/// ```text
/// P(r) = Σ_{j=0}^{n_free+1}  c_j · B_j(r),   with c_0 = c_{n_free+1} = 0
/// ```
///
/// Keeping the endpoint basis functions in the design matrix means the
/// regulariser operates on all n_free + 2 coefficients, so the first-derivative
/// penalty naturally includes the jumps c_1 − c_0 and c_{n_free+1} − c_{n_free}.
/// This eliminates the slope discontinuity at both boundaries.
///
/// # Parameters
///
/// `n_basis` is the number of **free** (interior) parameters.  The total number
/// of basis functions (including the two pinned endpoints) is `n_basis + 2`.
/// The relationship to the underlying knot vector is:
///
/// ```text
/// n_interior_knots = n_basis - 2
/// n_total          = n_basis + 2
/// ```
///
/// The default recommended value is `n_basis = 20` (gives 22 total columns).
pub struct CubicBSpline {
    knots: Vec<f64>,
    /// Greville abscissae of ALL basis functions (B_0 … B_{n_total-1}).
    /// Includes the endpoint abscissae at 0 and r_max.
    r_all: Vec<f64>,
    r_max: f64,
}

impl CubicBSpline {
    /// Create a cubic B-spline basis on `[0, r_max]` with `n_basis` free parameters.
    ///
    /// Interior knots are placed uniformly.  The returned basis has `n_basis + 2`
    /// columns (free interior functions plus the two boundary functions at 0 and r_max).
    /// Boundary conditions P(0) = P(r_max) = 0 must be enforced by calling
    /// `append_boundary_constraints` on the assembled weighted system.
    ///
    /// # Panics
    ///
    /// Panics if `r_max <= 0` or `n_basis < 2`.
    pub fn new(r_max: f64, n_basis: usize) -> Self {
        assert!(r_max > 0.0, "r_max must be positive");
        assert!(n_basis >= 2, "n_basis must be at least 2");

        let n_interior = n_basis - 2;
        let knots = bspline::clamped_knots(r_max, n_interior);

        // All Greville abscissae including the two endpoint ones (at 0 and r_max).
        let r_all = bspline::greville(&knots, 3);

        Self { knots, r_all, r_max }
    }
}

impl BasisSet for CubicBSpline {
    fn r_values(&self) -> &[f64] {
        &self.r_all
    }

    fn r_max(&self) -> f64 {
        self.r_max
    }

    /// K_ij = 4π ∫ B_j(r) sin(q_i r)/(q_i r) dr   for ALL basis functions j,
    /// including the two endpoint functions at r=0 and r=r_max.
    ///
    /// Built via 5-point Gauss–Legendre quadrature per knot span.
    /// The returned matrix has shape `(n_q × (n_free + 2))`.
    /// The first and last columns (endpoint B-splines) are kept so that the
    /// regulariser can penalise the slope at the boundaries; their coefficients
    /// are pinned to zero by `append_boundary_constraints`.
    fn build_kernel_matrix(&self, q: &[f64]) -> DMatrix<f64> {
        bspline::sinc_kernel_matrix(&self.knots, 3, q)
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
    ///
    /// Boundary ghost columns (j = 0 and j = n_r − 1) return 0 because the
    /// ghost bins have zero width and therefore contribute nothing to I(q).
    fn build_kernel_matrix(&self, q: &[f64]) -> DMatrix<f64> {
        let n_q = q.len();
        let n_r = self.r.len();
        let delta_r = self.delta_r;
        let last = n_r - 1;

        DMatrix::from_fn(n_q, n_r, |i, j| {
            if j == 0 || j == last {
                return 0.0; // ghost boundary bin — zero kernel contribution
            }
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

    /// n_basis free parameters → correct total dimensions (free + 2 boundary).
    #[test]
    fn cubic_bspline_dimensions() {
        let bs = CubicBSpline::new(150.0, 20);
        // Total = n_free + 2 boundary endpoints
        assert_eq!(bs.n_basis(), 22);
        assert_eq!(bs.r_values().len(), 22);
        assert_eq!(bs.r_max(), 150.0);

        let q: Vec<f64> = (1..=50).map(|i| i as f64 * 0.01).collect();
        let k = bs.build_kernel_matrix(&q);
        assert_eq!(k.nrows(), 50);
        assert_eq!(k.ncols(), 22);
    }

    /// r_values() now includes ALL Greville abscissae: the two endpoints at 0
    /// and r_max (for the boundary basis functions) and the free interior ones
    /// strictly between them.
    #[test]
    fn greville_endpoints_and_interior() {
        let r_max = 100.0_f64;
        let bs = CubicBSpline::new(r_max, 15);
        let r = bs.r_values();
        // Total = 15 free + 2 boundary = 17
        assert_eq!(r.len(), 17);
        // First and last are the boundary abscissae
        assert!((r[0] - 0.0).abs() < 1e-14, "first abscissa must be 0, got {}", r[0]);
        assert!((r[r.len()-1] - r_max).abs() < 1e-14,
            "last abscissa must be r_max, got {}", r[r.len()-1]);
        // Interior ones are strictly between 0 and r_max
        for &xi in &r[1..r.len()-1] {
            assert!(xi > 0.0 && xi < r_max,
                "interior Greville abscissa {xi} not strictly inside (0, {r_max})");
        }
    }

    /// At q = 0, K_ij = 4π ∫ B_j(r) dr.  All columns, including the endpoint
    /// basis functions, should be positive (the sinc integrand is positive near q = 0).
    #[test]
    fn kernel_entries_positive_at_q0() {
        let bs = CubicBSpline::new(80.0, 12);
        let k = bs.build_kernel_matrix(&[0.0]);
        for j in 0..k.ncols() {
            assert!(k[(0, j)] > 0.0, "K[0,{j}] = {} at q=0 (expected > 0)", k[(0, j)]);
        }
    }
}
