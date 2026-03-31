use nalgebra::DMatrix;

/// Produces a regularisation matrix L such that ‖Lc‖² penalises undesirable
/// features of the coefficient vector (roughness, curvature, etc.).
///
/// The Tikhonov-regularised normal equations become:
///
///   (KᵀK + λ · LᵀL) c = KᵀI
///
/// The trait returns L itself (not λLᵀL) so that the caller can scale by λ
/// and cache LᵀL across a λ grid search without recomputing L.
///
/// # Future implementations
///
/// - `FirstDerivative` — penalises slope; less smooth than second derivative
/// - `Identity` — ridge regression / Tikhonov with L = I
/// - `DataAdaptive` — weight the penalty by local curvature of the data
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
