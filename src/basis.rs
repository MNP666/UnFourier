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
