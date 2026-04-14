use crate::bspline;
use nalgebra::DMatrix;
use serde::Deserialize;

/// Boundary constraints enforced by fixing full clamped-spline coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplineBoundaryMode {
    /// Fix only the endpoint coefficients: `[0, c..., 0]`.
    ValueZero,
    /// Fix endpoint and boundary-adjacent coefficients: `[0, 0, c..., 0, 0]`.
    ValueSlopeZero,
}

impl Default for SplineBoundaryMode {
    fn default() -> Self {
        Self::ValueZero
    }
}

impl std::fmt::Display for SplineBoundaryMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ValueZero => write!(f, "value_zero"),
            Self::ValueSlopeZero => write!(f, "value_slope_zero"),
        }
    }
}

impl SplineBoundaryMode {
    fn first_free_index(self) -> usize {
        match self {
            Self::ValueZero => 1,
            Self::ValueSlopeZero => 2,
        }
    }

    fn fixed_coefficients(self) -> usize {
        2 * self.first_free_index()
    }
}

/// Mapping between solved free coefficients and the full clamped spline vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplineCoefficientMap {
    mode: SplineBoundaryMode,
    n_free: usize,
    n_full: usize,
    first_free: usize,
}

impl SplineCoefficientMap {
    pub fn new(n_free: usize, mode: SplineBoundaryMode) -> Self {
        assert!(
            n_free >= 2,
            "spline coefficient map requires at least 2 free coefficients, got {}",
            n_free
        );
        let first_free = mode.first_free_index();
        let n_full = n_free + mode.fixed_coefficients();
        Self {
            mode,
            n_free,
            n_full,
            first_free,
        }
    }

    pub fn mode(&self) -> SplineBoundaryMode {
        self.mode
    }

    pub fn n_free(&self) -> usize {
        self.n_free
    }

    pub fn n_full(&self) -> usize {
        self.n_full
    }

    pub fn full_index(&self, free_index: usize) -> usize {
        assert!(
            free_index < self.n_free,
            "free coefficient index {} out of range 0..{}",
            free_index,
            self.n_free
        );
        self.first_free + free_index
    }

    pub fn expand(&self, free_coeffs: &[f64]) -> Vec<f64> {
        assert_eq!(
            free_coeffs.len(),
            self.n_free,
            "free coefficient count must match the spline coefficient map"
        );
        let mut full = vec![0.0_f64; self.n_full];
        for (j, &c) in free_coeffs.iter().enumerate() {
            full[self.full_index(j)] = c;
        }
        full
    }

    pub fn project_columns(&self, full: &DMatrix<f64>) -> DMatrix<f64> {
        assert_eq!(
            full.ncols(),
            self.n_full,
            "full matrix column count must match the spline coefficient map"
        );
        DMatrix::from_fn(full.nrows(), self.n_free, |i, j| {
            full[(i, self.full_index(j))]
        })
    }

    pub fn project_values(&self, full_values: &[f64]) -> Vec<f64> {
        assert_eq!(
            full_values.len(),
            self.n_full,
            "full value count must match the spline coefficient map"
        );
        (0..self.n_free)
            .map(|j| full_values[self.full_index(j)])
            .collect()
    }
}

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
/// P(r) is represented as a linear combination of `n_basis` free cubic
/// B-spline coefficients.  Boundary conditions are enforced by embedding those
/// free coefficients in the full clamped spline coefficient vector with fixed
/// zero coefficients at the ends.
///
/// ```text
/// value_zero:       [0, c..., 0]
/// value_slope_zero: [0, 0, c..., 0, 0]
/// ```
///
/// Endpoint values P(0)=P(r_max)=0 are enforced by both modes.  The
/// `value_slope_zero` mode also fixes the boundary-adjacent coefficients,
/// giving zero endpoint slope for the clamped cubic spline. The solved
/// coefficients are spline control weights, not sampled P(r) values; output
/// must be produced with [`BasisSet::evaluate_pr`].
///
/// # Parameters
///
/// `n_basis` is the number of **free** (interior) parameters.
/// The underlying knot vector uses enough uniformly spaced interior knots to
/// provide those free parameters plus the fixed boundary coefficients required
/// by the selected boundary mode.
/// The default recommended value is `n_basis = 20`.
pub struct CubicBSpline {
    knots: Vec<f64>,
    coeff_map: SplineCoefficientMap,
    /// Greville abscissae of the free basis functions only.
    r_free: Vec<f64>,
    r_max: f64,
}

impl CubicBSpline {
    /// Create a cubic B-spline basis on `[0, r_max]` with `n_basis` free parameters.
    pub fn new(r_max: f64, n_basis: usize) -> Self {
        Self::with_boundary_mode(r_max, n_basis, SplineBoundaryMode::default())
    }

    /// Create a cubic B-spline basis with an explicit boundary mode.
    ///
    /// Interior knots are placed uniformly.  The returned basis has `n_basis`
    /// free columns; the full clamped spline has additional fixed zero
    /// coefficients according to `boundary_mode`.
    ///
    /// # Panics
    ///
    /// Panics if `r_max <= 0` or `n_basis < 2`.
    pub fn with_boundary_mode(
        r_max: f64,
        n_basis: usize,
        boundary_mode: SplineBoundaryMode,
    ) -> Self {
        assert!(r_max > 0.0, "r_max must be positive");
        assert!(n_basis >= 2, "n_basis must be at least 2");

        let coeff_map = SplineCoefficientMap::new(n_basis, boundary_mode);
        let n_interior = coeff_map.n_full() - 4;
        let knots = bspline::clamped_knots(r_max, n_interior);

        let all_greville = bspline::greville(&knots, 3);
        let r_free = coeff_map.project_values(&all_greville);

        Self {
            knots,
            coeff_map,
            r_free,
            r_max,
        }
    }

    pub fn boundary_mode(&self) -> SplineBoundaryMode {
        self.coeff_map.mode()
    }

    pub fn coefficient_map(&self) -> &SplineCoefficientMap {
        &self.coeff_map
    }

    /// Basis matrix for the free spline coefficients only.
    fn free_basis_matrix(&self, r: &[f64]) -> DMatrix<f64> {
        let full = bspline::basis_matrix(&self.knots, 3, r);
        self.coeff_map.project_columns(&full)
    }
}

impl BasisSet for CubicBSpline {
    fn r_values(&self) -> &[f64] {
        &self.r_free
    }

    fn r_max(&self) -> f64 {
        self.r_max
    }

    /// K_ij = 4π ∫ B_j(r) sin(q_i r)/(q_i r) dr for the free spline
    /// coefficients. Fixed boundary coefficients are projected out through the
    /// shared coefficient map.
    fn build_kernel_matrix(&self, q: &[f64]) -> DMatrix<f64> {
        let full = bspline::sinc_kernel_matrix(&self.knots, 3, q);
        self.coeff_map.project_columns(&full)
    }

    fn output_grid(&self) -> Vec<f64> {
        let n_intervals = (self.r_free.len() * 10).max(200);
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

    #[test]
    fn value_zero_mapping_expands_free_coefficients() {
        let map = SplineCoefficientMap::new(3, SplineBoundaryMode::ValueZero);
        assert_eq!(map.n_free(), 3);
        assert_eq!(map.n_full(), 5);
        assert_eq!(map.expand(&[1.0, 2.0, 3.0]), vec![0.0, 1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn value_slope_zero_mapping_expands_free_coefficients() {
        let map = SplineCoefficientMap::new(3, SplineBoundaryMode::ValueSlopeZero);
        assert_eq!(map.n_free(), 3);
        assert_eq!(map.n_full(), 7);
        assert_eq!(
            map.expand(&[1.0, 2.0, 3.0]),
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
        );
    }

    /// n_basis free parameters → correct kernel dimensions.
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

    #[test]
    fn value_slope_zero_keeps_same_free_kernel_shape() {
        let bs = CubicBSpline::with_boundary_mode(150.0, 20, SplineBoundaryMode::ValueSlopeZero);
        assert_eq!(bs.n_basis(), 20);
        assert_eq!(bs.coefficient_map().n_full(), 24);

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
    fn value_slope_zero_output_has_flat_endpoints() {
        let bs = CubicBSpline::with_boundary_mode(100.0, 8, SplineBoundaryMode::ValueSlopeZero);
        let coeffs: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let eps = 1e-4_f64;
        let r = vec![0.0, eps, 100.0 - eps, 100.0];
        let p = bs.evaluate_pr(&coeffs, &r);

        assert!(p[0].abs() < 1e-12, "P(0) must be exactly zero");
        assert!(p[3].abs() < 1e-12, "P(Dmax) must be exactly zero");
        assert!(
            ((p[1] - p[0]) / eps).abs() < 1e-4,
            "left endpoint slope should be near zero"
        );
        assert!(
            ((p[3] - p[2]) / eps).abs() < 1e-4,
            "right endpoint slope should be near zero"
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
