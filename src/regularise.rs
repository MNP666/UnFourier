use nalgebra::DMatrix;

use crate::basis::{SplineBoundaryMode, SplineCoefficientMap};

/// Produces a regularisation matrix L such that ‖Lc‖² penalises undesirable
/// features of the coefficient vector (roughness, curvature, etc.).
///
/// The Tikhonov-regularised normal equations become:
///
///   (KᵀK + λ · LᵀL) c = KᵀI
///
/// The trait returns L itself (not λLᵀL) so that the caller can scale by λ
/// and cache LᵀL across a λ grid search without recomputing L.
pub trait Regulariser: Send + Sync {
    fn name(&self) -> &str;

    /// Build the regularisation matrix L for a basis of size `n_basis`.
    /// Dimensions depend on the implementation (e.g. (n-2) × n for second derivative).
    fn matrix(&self, n_basis: usize) -> DMatrix<f64>;

    /// Convenience: compute LᵀL directly.
    fn gram_matrix(&self, n_basis: usize) -> DMatrix<f64> {
        let l = self.matrix(n_basis);
        l.transpose() * l
    }
}

// ---------------------------------------------------------------------------
// SecondDerivative
// ---------------------------------------------------------------------------

/// Penalises curvature in P(r) via finite-difference second derivatives.
///
/// L is an (n−2) × n banded matrix:
///
///   L[i, i]   =  1
///   L[i, i+1] = -2
///   L[i, i+2] =  1
///
/// so that (Lc)[i] ≈ c[i] − 2c[i+1] + c[i+2], the discrete second derivative
/// at position i+1. Minimising ‖Lc‖² encourages P(r) to be smooth.
///
/// This is the standard regularisation used in GNOM-style IFT.
// ---------------------------------------------------------------------------
// FirstDerivative
// ---------------------------------------------------------------------------

/// Penalises slope (first differences) in P(r).
///
/// L is an (n−1) × n banded matrix:
///
///   L[i, i]   = -1
///   L[i, i+1] = +1
///
/// so that (Lc)[i] ≈ c[i+1] − c[i], the discrete first derivative.
/// Minimising ‖Lc‖² discourages large step changes between adjacent bins.
pub struct FirstDerivative;

impl Regulariser for FirstDerivative {
    fn name(&self) -> &str {
        "first-derivative"
    }

    fn matrix(&self, n_basis: usize) -> DMatrix<f64> {
        assert!(
            n_basis >= 2,
            "FirstDerivative regularisation requires at least 2 basis functions, got {}",
            n_basis
        );
        let nrows = n_basis - 1;
        DMatrix::from_fn(nrows, n_basis, |i, j| {
            if j == i {
                -1.0
            } else if j == i + 1 {
                1.0
            } else {
                0.0
            }
        })
    }
}

// ---------------------------------------------------------------------------
// CombinedDerivative
// ---------------------------------------------------------------------------

/// Combined slope + curvature penalty: ‖Lc‖² = d₁‖D₁c‖² + d₂‖D₂c‖².
///
/// L is the vertically stacked matrix `[ sqrt(d₁)·D₁ ; sqrt(d₂)·D₂ ]`
/// with shape `(2n−3) × n`.
///
/// When `d1_weight = 0.0` and `d2_weight = 1.0` this reduces exactly to
/// `SecondDerivative` — the Gram matrix `LᵀL` is identical entry-wise.
pub struct CombinedDerivative {
    pub d1_weight: f64,
    pub d2_weight: f64,
}

impl Regulariser for CombinedDerivative {
    fn name(&self) -> &str {
        "combined-derivative"
    }

    fn matrix(&self, n_basis: usize) -> DMatrix<f64> {
        let d1 = FirstDerivative.matrix(n_basis);
        let d2 = SecondDerivative.matrix(n_basis);
        let nrows = d1.nrows() + d2.nrows(); // (n-1) + (n-2) = 2n-3
        let sqrt_d1 = self.d1_weight.sqrt();
        let sqrt_d2 = self.d2_weight.sqrt();
        DMatrix::from_fn(nrows, n_basis, |i, j| {
            if i < d1.nrows() {
                d1[(i, j)] * sqrt_d1
            } else {
                d2[(i - d1.nrows(), j)] * sqrt_d2
            }
        })
    }
}

// ---------------------------------------------------------------------------
// ProjectedSplineRegulariser
// ---------------------------------------------------------------------------

/// Combined derivative penalty built in full clamped-spline coefficient space
/// and projected onto the free coefficients.
///
/// The spline basis fixes boundary coefficients by parameterisation. This
/// regulariser applies the same coefficient map to the derivative matrix:
///
/// ```text
/// L_free = L_full * B
/// ```
///
/// where `B` embeds free coefficients into the full clamped coefficient vector.
pub struct ProjectedSplineRegulariser {
    pub boundary_mode: SplineBoundaryMode,
    pub d1_weight: f64,
    pub d2_weight: f64,
}

impl Regulariser for ProjectedSplineRegulariser {
    fn name(&self) -> &str {
        "projected-spline-derivative"
    }

    fn matrix(&self, n_basis: usize) -> DMatrix<f64> {
        let coeff_map = SplineCoefficientMap::new(n_basis, self.boundary_mode);
        let full = CombinedDerivative {
            d1_weight: self.d1_weight,
            d2_weight: self.d2_weight,
        }
        .matrix(coeff_map.n_full());
        coeff_map.project_columns(&full)
    }
}

// ---------------------------------------------------------------------------
// SecondDerivative
// ---------------------------------------------------------------------------

pub struct SecondDerivative;

impl Regulariser for SecondDerivative {
    fn name(&self) -> &str {
        "second-derivative"
    }

    fn matrix(&self, n_basis: usize) -> DMatrix<f64> {
        assert!(
            n_basis >= 3,
            "SecondDerivative regularisation requires at least 3 basis functions, got {}",
            n_basis
        );
        let nrows = n_basis - 2;
        DMatrix::from_fn(nrows, n_basis, |i, j| {
            if j == i {
                1.0
            } else if j == i + 1 {
                -2.0
            } else if j == i + 2 {
                1.0
            } else {
                0.0
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_derivative_shape() {
        let l = FirstDerivative.matrix(4);
        assert_eq!(l.nrows(), 3); // n-1 = 3
        assert_eq!(l.ncols(), 4);
        // Check first row: [-1, 1, 0, 0]
        assert_eq!(l[(0, 0)], -1.0);
        assert_eq!(l[(0, 1)], 1.0);
        assert_eq!(l[(0, 2)], 0.0);
    }

    #[test]
    fn combined_matches_second_when_d1_zero() {
        let n = 5;
        let combined = CombinedDerivative {
            d1_weight: 0.0,
            d2_weight: 1.0,
        };
        let ltl_combined = combined.gram_matrix(n);
        let ltl_second = SecondDerivative.gram_matrix(n);

        for i in 0..n {
            for j in 0..n {
                let diff = (ltl_combined[(i, j)] - ltl_second[(i, j)]).abs();
                assert!(
                    diff < 1e-12,
                    "ltl_combined[{i},{j}]={} != ltl_second[{i},{j}]={}",
                    ltl_combined[(i, j)],
                    ltl_second[(i, j)]
                );
            }
        }
    }

    #[test]
    fn combined_d1_increases_trace() {
        let n = 10;
        let combined = CombinedDerivative {
            d1_weight: 1.0,
            d2_weight: 1.0,
        };
        let trace_combined = combined.gram_matrix(n).trace();
        let trace_second = SecondDerivative.gram_matrix(n).trace();
        assert!(
            trace_combined > trace_second,
            "trace_combined={} should exceed trace_second={}",
            trace_combined,
            trace_second
        );
    }

    #[test]
    fn projected_regulariser_value_zero_projects_full_derivative() {
        let n = 5;
        let mode = SplineBoundaryMode::ValueZero;
        let projected = ProjectedSplineRegulariser {
            boundary_mode: mode,
            d1_weight: 0.0,
            d2_weight: 1.0,
        };
        let l = projected.matrix(n);

        let map = SplineCoefficientMap::new(n, mode);
        let full = CombinedDerivative {
            d1_weight: 0.0,
            d2_weight: 1.0,
        }
        .matrix(map.n_full());
        let expected = map.project_columns(&full);

        assert_eq!(l.nrows(), expected.nrows());
        assert_eq!(l.ncols(), n);
        for i in 0..l.nrows() {
            for j in 0..l.ncols() {
                assert_eq!(l[(i, j)], expected[(i, j)]);
            }
        }
    }

    #[test]
    fn projected_regulariser_value_slope_zero_projects_full_derivative() {
        let n = 4;
        let mode = SplineBoundaryMode::ValueSlopeZero;
        let projected = ProjectedSplineRegulariser {
            boundary_mode: mode,
            d1_weight: 1.0,
            d2_weight: 1.0,
        };
        let l = projected.matrix(n);

        let map = SplineCoefficientMap::new(n, mode);
        let full = CombinedDerivative {
            d1_weight: 1.0,
            d2_weight: 1.0,
        }
        .matrix(map.n_full());
        let expected = map.project_columns(&full);

        assert_eq!(l.nrows(), expected.nrows());
        assert_eq!(l.ncols(), n);
        for i in 0..l.nrows() {
            for j in 0..l.ncols() {
                assert_eq!(l[(i, j)], expected[(i, j)]);
            }
        }
    }
}
