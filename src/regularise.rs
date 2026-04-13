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

// ---------------------------------------------------------------------------
// Boundary-anchored derivatives
// ---------------------------------------------------------------------------

/// First-derivative matrix anchored to zero boundary conditions (n+1) × n.
///
/// The first and last rows penalise the slope from/to the implicit zero
/// boundaries (c[-1] = 0 at the left, c[n] = 0 at the right):
///
/// ```text
///   Row 0:   [ 1,  0, …,  0]  ← c[0]−0    = c[0]    (left boundary slope)
///   Row k:   [−1,  1, 0, …]   ← c[k]−c[k−1]
///   Row n:   [ 0, …, 0, −1]   ← 0−c[n−1] = −c[n−1]  (right boundary slope)
/// ```
fn ba_d1_matrix(n: usize) -> DMatrix<f64> {
    assert!(n >= 1, "ba_d1_matrix requires at least 1 interior coefficient");
    DMatrix::from_fn(n + 1, n, |i, j| {
        if i == 0 {
            if j == 0 { 1.0 } else { 0.0 }
        } else if i == n {
            if j == n - 1 { -1.0 } else { 0.0 }
        } else {
            // Interior row: difference c[i] − c[i−1]
            if j + 1 == i { -1.0 } else if j == i { 1.0 } else { 0.0 }
        }
    })
}

/// Second-derivative matrix anchored to zero boundary conditions, n × n.
///
/// Rows include the implicit zero boundary values (c[-1]=0, c[n]=0):
///
/// ```text
///   Row 0:   [−2, 1, 0, …]  ← 0−2c[0]+c[1]
///   Row k:   […, 1, −2, 1, …]  ← c[k−1]−2c[k]+c[k+1]
///   Row n−1: [0, …, 1, −2]  ← c[n−2]−2c[n−1]+0
/// ```
fn ba_d2_matrix(n: usize) -> DMatrix<f64> {
    assert!(n >= 1, "ba_d2_matrix requires at least 1 interior coefficient");
    DMatrix::from_fn(n, n, |i, j| {
        if j == i {
            -2.0
        } else if j == i + 1 || (i > 0 && j + 1 == i) {
            1.0
        } else {
            0.0
        }
    })
}

/// Combined slope + curvature penalty with zero boundary anchoring.
///
/// `‖L·c‖² = d₁·‖D̃₁·c‖² + d₂·‖D̃₂·c‖²`
///
/// where D̃₁ (n+1 × n) and D̃₂ (n × n) are the boundary-anchored first- and
/// second-derivative matrices.  They include the slope from/to the implicit
/// zero boundary values, which eliminates the competition between soft
/// boundary constraints and the regulariser.
///
/// When `d1_weight = 0.0` (the default), the combined matrix reduces to
/// `D̃₂` alone (boundary-anchored curvature only).
pub struct BoundaryAnchoredCombined {
    pub d1_weight: f64,
    pub d2_weight: f64,
}

impl Regulariser for BoundaryAnchoredCombined {
    fn name(&self) -> &str {
        "boundary-anchored-combined"
    }

    fn matrix(&self, n_basis: usize) -> DMatrix<f64> {
        let d2 = ba_d2_matrix(n_basis);
        let sqrt_d2 = self.d2_weight.sqrt();
        if self.d1_weight == 0.0 {
            return d2 * sqrt_d2;
        }
        let d1 = ba_d1_matrix(n_basis);
        let nrows = d1.nrows() + d2.nrows(); // (n+1) + n = 2n+1
        let sqrt_d1 = self.d1_weight.sqrt();
        DMatrix::from_fn(nrows, n_basis, |i, j| {
            if i < d1.nrows() {
                d1[(i, j)] * sqrt_d1
            } else {
                d2[(i - d1.nrows(), j)] * sqrt_d2
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
        let combined = CombinedDerivative { d1_weight: 0.0, d2_weight: 1.0 };
        let ltl_combined = combined.gram_matrix(n);
        let ltl_second   = SecondDerivative.gram_matrix(n);

        for i in 0..n {
            for j in 0..n {
                let diff = (ltl_combined[(i, j)] - ltl_second[(i, j)]).abs();
                assert!(
                    diff < 1e-12,
                    "ltl_combined[{i},{j}]={} != ltl_second[{i},{j}]={}",
                    ltl_combined[(i, j)], ltl_second[(i, j)]
                );
            }
        }
    }

    #[test]
    fn combined_d1_increases_trace() {
        let n = 10;
        let combined = CombinedDerivative { d1_weight: 1.0, d2_weight: 1.0 };
        let trace_combined = combined.gram_matrix(n).trace();
        let trace_second   = SecondDerivative.gram_matrix(n).trace();
        assert!(
            trace_combined > trace_second,
            "trace_combined={} should exceed trace_second={}",
            trace_combined, trace_second
        );
    }

    // ------------------------------------------------------------------
    // Boundary-anchored helpers
    // ------------------------------------------------------------------

    #[test]
    fn ba_d1_shape_and_boundary_rows() {
        let n = 3;
        let d = ba_d1_matrix(n);
        assert_eq!(d.nrows(), n + 1); // 4
        assert_eq!(d.ncols(), n);     // 3

        // Row 0: [1, 0, 0]
        assert_eq!(d[(0, 0)],  1.0);
        assert_eq!(d[(0, 1)],  0.0);

        // Row 1: [-1, 1, 0]
        assert_eq!(d[(1, 0)], -1.0);
        assert_eq!(d[(1, 1)],  1.0);
        assert_eq!(d[(1, 2)],  0.0);

        // Row 3 (last): [0, 0, -1]
        assert_eq!(d[(3, 1)],  0.0);
        assert_eq!(d[(3, 2)], -1.0);
    }

    #[test]
    fn ba_d2_shape_and_boundary_rows() {
        let n = 3;
        let d = ba_d2_matrix(n);
        assert_eq!(d.nrows(), n); // 3
        assert_eq!(d.ncols(), n); // 3

        // Row 0: [-2, 1, 0]
        assert_eq!(d[(0, 0)], -2.0);
        assert_eq!(d[(0, 1)],  1.0);
        assert_eq!(d[(0, 2)],  0.0);

        // Row 1: [1, -2, 1]
        assert_eq!(d[(1, 0)],  1.0);
        assert_eq!(d[(1, 1)], -2.0);
        assert_eq!(d[(1, 2)],  1.0);

        // Row 2 (last): [0, 1, -2]
        assert_eq!(d[(2, 0)],  0.0);
        assert_eq!(d[(2, 1)],  1.0);
        assert_eq!(d[(2, 2)], -2.0);
    }

    #[test]
    fn ba_combined_d1_zero_reduces_to_ba_d2_gram() {
        let n = 5;
        let combined = BoundaryAnchoredCombined { d1_weight: 0.0, d2_weight: 1.0 };
        let ltl_combined = combined.gram_matrix(n);
        // Should equal ba_d2_matrix(n)^T * ba_d2_matrix(n)
        let d2 = ba_d2_matrix(n);
        let ltl_d2 = d2.transpose() * &d2;
        for i in 0..n {
            for j in 0..n {
                let diff = (ltl_combined[(i, j)] - ltl_d2[(i, j)]).abs();
                assert!(
                    diff < 1e-12,
                    "ltl_combined[{i},{j}]={} != ltl_d2[{i},{j}]={}",
                    ltl_combined[(i, j)], ltl_d2[(i, j)]
                );
            }
        }
    }

    #[test]
    fn ba_combined_d1_adds_to_trace() {
        let n = 8;
        let d2_only = BoundaryAnchoredCombined { d1_weight: 0.0, d2_weight: 1.0 };
        let combined = BoundaryAnchoredCombined { d1_weight: 1.0, d2_weight: 1.0 };
        assert!(
            combined.gram_matrix(n).trace() > d2_only.gram_matrix(n).trace(),
            "adding d1 weight must increase trace"
        );
    }
}
