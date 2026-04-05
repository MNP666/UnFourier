//! Cox–de Boor B-spline evaluation utilities.
//!
//! Internal module used by `CubicBSpline` (M5).  Not part of the public trait
//! hierarchy — nothing here needs to be touched when adding other basis types.
//!
//! # Conventions
//!
//! * Only cubic B-splines (degree 3) are the design target, but the recursion
//!   is degree-generic so all functions accept a `degree` parameter.
//! * The clamped (open) knot vector produced by [`clamped_knots`] has
//!   `degree + 1` repeated zeros at the left and `degree + 1` repeated `r_max`
//!   values at the right.  This guarantees `B_0(0) = 1` and `B_{n-1}(r_max) = 1`.
//! * [`basis_matrix`] returns the **full** `(n_r × n_basis)` matrix including
//!   both endpoint basis functions.  The caller (M5 `CubicBSpline`) is
//!   responsible for dropping the first and last columns to enforce
//!   `P(0) = P(r_max) = 0`.

use nalgebra::DMatrix;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a clamped (open) knot vector for cubic B-splines.
///
/// The returned vector has the form
/// ```text
/// [0, 0, 0, 0,  t_1, …, t_n_interior,  r_max, r_max, r_max, r_max]
/// ```
/// with `n_interior` uniformly spaced interior knots.
///
/// The total number of B-spline basis functions for this vector is
/// `n_interior + degree + 1 = n_interior + 4` (for degree 3).
pub fn clamped_knots(r_max: f64, n_interior: usize) -> Vec<f64> {
    let degree = 3_usize;
    let mut knots = Vec::with_capacity(n_interior + 2 * (degree + 1));

    // Left clamped: degree+1 zeros
    knots.extend(std::iter::repeat(0.0_f64).take(degree + 1));

    // Uniformly spaced interior knots
    for i in 1..=n_interior {
        knots.push(r_max * i as f64 / (n_interior + 1) as f64);
    }

    // Right clamped: degree+1 copies of r_max
    knots.extend(std::iter::repeat(r_max).take(degree + 1));

    knots
}

/// Greville abscissae (collocation points) for the B-spline basis.
///
/// For the j-th basis function:
/// ```text
/// ξ_j = (t_{j+1} + t_{j+2} + … + t_{j+degree}) / degree
/// ```
///
/// Returns a `Vec<f64>` of length `n_basis = len(knots) - degree - 1`.
pub fn greville(knots: &[f64], degree: usize) -> Vec<f64> {
    let n_basis = knots.len() - degree - 1;
    (0..n_basis)
        .map(|j| {
            let sum: f64 = knots[j + 1..j + degree + 1].iter().sum();
            sum / degree as f64
        })
        .collect()
}

/// Evaluate all B-spline basis functions at each point in `r`.
///
/// Uses the Cox–de Boor triangular recursion:
/// ```text
/// B_{i,0}(r) = 1  if t_i ≤ r < t_{i+1}  else 0
/// B_{i,d}(r) = (r − t_i)/(t_{i+d} − t_i)  · B_{i,d−1}(r)
///            + (t_{i+d+1} − r)/(t_{i+d+1} − t_{i+1}) · B_{i+1,d−1}(r)
/// ```
/// with the 0/0 convention = 0.
///
/// Returns a **full** `DMatrix<f64>` of shape `(n_r, n_basis)` where
/// `n_basis = len(knots) − degree − 1`.
///
/// At `r = r_max` the standard de Boor convention is applied: the point is
/// treated as belonging to the last non-degenerate knot span, which ensures
/// `B_{n_basis−1}(r_max) = 1.0`.
pub fn basis_matrix(knots: &[f64], degree: usize, r: &[f64]) -> DMatrix<f64> {
    let n_basis = knots.len() - degree - 1;
    let n_r = r.len();
    let mut mat = DMatrix::zeros(n_r, n_basis);
    for (row, &rv) in r.iter().enumerate() {
        let vals = basis_at(knots, degree, rv);
        for (col, v) in vals.into_iter().enumerate() {
            mat[(row, col)] = v;
        }
    }
    mat
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Evaluate all `n_basis` B-spline basis functions at a single point `r`.
///
/// Returns a `Vec<f64>` of length `n_basis`.
fn basis_at(knots: &[f64], degree: usize, r: f64) -> Vec<f64> {
    let m = knots.len() - 1; // index of last knot

    // ------------------------------------------------------------------
    // Degree-0 initialisation: exactly one knot span is active.
    //
    // For r strictly in [t_i, t_{i+1}) we set b[i] = 1.
    // For r == r_max (the right endpoint) we use the de Boor convention:
    // activate the last non-degenerate span (the one with t_i < t_{i+1}).
    // ------------------------------------------------------------------
    let mut b = vec![0.0_f64; m]; // b[i] = B_{i,0}(r)

    let r_max = knots[m];
    if r >= r_max {
        // Right endpoint: find last span with positive length
        for i in (0..m).rev() {
            if knots[i + 1] > knots[i] {
                b[i] = 1.0;
                break;
            }
        }
    } else {
        // Normal interior point: find the unique active span
        for i in 0..m {
            if knots[i] <= r && r < knots[i + 1] {
                b[i] = 1.0;
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // Triangular recurrence: build degrees 1 up to `degree`.
    // After step d the vector has length m - d.
    // After `degree` steps we have m - degree = n_basis entries.
    // ------------------------------------------------------------------
    for d in 1..=degree {
        let new_len = m - d;
        let mut new_b = vec![0.0_f64; new_len];
        for i in 0..new_len {
            // Left term
            let denom1 = knots[i + d] - knots[i];
            if denom1.abs() > f64::EPSILON {
                new_b[i] += (r - knots[i]) / denom1 * b[i];
            }
            // Right term
            let denom2 = knots[i + d + 1] - knots[i + 1];
            if denom2.abs() > f64::EPSILON {
                new_b[i] += (knots[i + d + 1] - r) / denom2 * b[i + 1];
            }
        }
        b = new_b;
    }

    b
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A clamped cubic knot vector with 5 interior knots gives 5+4 = 9 basis
    /// functions spanning 5+1 = 6 non-degenerate knot spans.  The degree-0
    /// basis functions form a partition of unity trivially; the recurrence
    /// preserves this property, so checking at a dense r grid verifies the
    /// implementation end-to-end.
    #[test]
    fn partition_of_unity() {
        let r_max = 100.0_f64;
        let knots = clamped_knots(r_max, 5);
        let r: Vec<f64> = (1..=99).map(|i| i as f64).collect();
        let b = basis_matrix(&knots, 3, &r);
        for row in 0..b.nrows() {
            let sum: f64 = (0..b.ncols()).map(|c| b[(row, c)]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "row {row}: partition of unity violated, sum = {sum}"
            );
        }
    }

    /// At r = 0 the clamped knot vector forces B_0(0) = 1 and all others = 0.
    /// At r = r_max the last basis function equals 1 and all others = 0.
    #[test]
    fn endpoint_values() {
        let r_max = 80.0_f64;
        let knots = clamped_knots(r_max, 4); // 4 interior knots → 8 basis fns
        let n = knots.len() - 3 - 1;        // n_basis

        let b = basis_matrix(&knots, 3, &[0.0, r_max]);

        // r = 0 row
        assert!(
            (b[(0, 0)] - 1.0).abs() < 1e-14,
            "B_0(0) = {} (expected 1.0)",
            b[(0, 0)]
        );
        for j in 1..n {
            assert!(
                b[(0, j)].abs() < 1e-14,
                "B_{j}(0) = {} (expected 0.0)",
                b[(0, j)]
            );
        }

        // r = r_max row
        assert!(
            (b[(1, n - 1)] - 1.0).abs() < 1e-14,
            "B_{{n-1}}(r_max) = {} (expected 1.0)",
            b[(1, n - 1)]
        );
        for j in 0..n - 1 {
            assert!(
                b[(1, j)].abs() < 1e-14,
                "B_{j}(r_max) = {} (expected 0.0)",
                b[(1, j)]
            );
        }
    }

    /// greville() should return n_basis values, all within [0, r_max].
    #[test]
    fn greville_range() {
        let r_max = 50.0_f64;
        let knots = clamped_knots(r_max, 6);
        let xi = greville(&knots, 3);
        let n_basis = knots.len() - 4;
        assert_eq!(xi.len(), n_basis);
        for &v in &xi {
            assert!(v >= 0.0 && v <= r_max, "Greville abscissa {v} out of range");
        }
        // First and last Greville abscissae for a clamped cubic knot vector
        // are always 0 and r_max respectively.
        assert!((xi[0] - 0.0).abs() < 1e-14);
        assert!((xi[n_basis - 1] - r_max).abs() < 1e-14);
    }
}
