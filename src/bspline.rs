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
// Gauss–Legendre quadrature constants (5-point, nodes and weights on [-1, 1])
// ---------------------------------------------------------------------------

const GL_NODES: [f64; 5] = [
    -0.906_179_845_938_664_0,
    -0.538_469_310_105_683_1,
     0.0,
     0.538_469_310_105_683_1,
     0.906_179_845_938_664_0,
];

const GL_WEIGHTS: [f64; 5] = [
    0.236_926_885_056_189_1,
    0.478_628_670_499_366_5,
    0.568_888_888_888_888_9,
    0.478_628_670_499_366_5,
    0.236_926_885_056_189_1,
];

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

/// Compute the kernel integral for one basis function at one q value:
///
/// ```text
/// K_j(q) = 4π ∫ B_j(r) · sin(qr)/(qr) dr
/// ```
///
/// Integration is by 5-point Gauss–Legendre on each of the (up to `degree + 1`)
/// non-degenerate knot spans that make up the compact support `[t_j, t_{j+degree+1}]`.
///
/// The q → 0 limit sin(qr)/(qr) → 1 is handled explicitly for |qr| < 1e-8.
pub fn integrate_basis_sinc(knots: &[f64], degree: usize, j: usize, q: f64) -> f64 {
    let mut integral = 0.0;

    // The support of B_j (degree d) spans knot intervals j, j+1, …, j+d.
    for span in j..=j + degree {
        let a = knots[span];
        let b = knots[span + 1];
        let h = b - a;
        if h < f64::EPSILON {
            continue; // degenerate span (repeated knot) — contributes nothing
        }

        for k in 0..5 {
            // Map GL node from [-1, 1] to [a, b]
            let r = 0.5 * (b + a) + 0.5 * h * GL_NODES[k];
            let w = 0.5 * h * GL_WEIGHTS[k];

            let b_j = basis_at(knots, degree, r)[j];

            let qr = q * r;
            let sinc = if qr.abs() < 1e-8 { 1.0 } else { qr.sin() / qr };

            integral += w * b_j * sinc;
        }
    }

    4.0 * std::f64::consts::PI * integral
}

/// Build the full kernel matrix K of shape `(n_q × n_basis)` using
/// [`integrate_basis_sinc`] for every (q, j) pair.
///
/// **Includes all basis functions**, the two endpoint columns (j = 0 and
/// j = n_basis − 1) among them.  The `CubicBSpline` struct (M5, task 4) drops
/// those columns when it implements `BasisSet::build_kernel_matrix`.
pub fn sinc_kernel_matrix(knots: &[f64], degree: usize, q: &[f64]) -> DMatrix<f64> {
    let n_basis = knots.len() - degree - 1;
    let n_q = q.len();
    DMatrix::from_fn(n_q, n_basis, |i, j| integrate_basis_sinc(knots, degree, j, q[i]))
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

    /// At q = 0 the sinc kernel reduces to 4π ∫ B_j(r) dr.
    /// For a clamped uniform cubic B-spline on [0, 50] with 3 interior knots
    /// (span width 12.5) the 5-point GL quadrature should be exact to machine
    /// precision because the integrand (a piecewise cubic polynomial) is
    /// integrated exactly by 2-point GL, let alone 5-point.
    ///
    /// Exact integrals (derivable from B-spline volume formula):
    ///   ∫ B_j dr = span_width × { 1/4, 1/2, 3/4, 1, 3/4, 1/2, 1/4 } for j = 0..6
    ///   K_j(0)  = 4π × ∫ B_j dr
    #[test]
    fn kernel_q0_exact() {
        let knots = clamped_knots(50.0, 3); // spans of width 12.5
        let span = 50.0 / 4.0; // 12.5
        let fracs = [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25_f64];
        let expected: Vec<f64> = fracs.iter().map(|&f| 4.0 * std::f64::consts::PI * span * f).collect();

        for (j, &exp) in expected.iter().enumerate() {
            let got = integrate_basis_sinc(&knots, 3, j, 0.0);
            assert!(
                (got - exp).abs() < 1e-10,
                "j={j}: got {got:.12}, expected {exp:.12}, diff={:.2e}",
                (got - exp).abs()
            );
        }
    }

    /// Compare against Python scipy reference values computed with the same
    /// 5-point Gauss–Legendre formula.  Both implementations evaluate the
    /// identical integral; differences should be at floating-point roundoff
    /// (< 1e-7 absolute for the values tested here).
    ///
    /// Reference generated by Dev/spline_pr_tests.py with n_gl=5,
    /// r_max=50, n_interior=3, knots=[0,0,0,0, 12.5, 25, 37.5, 50,50,50,50].
    #[test]
    fn kernel_matches_python_reference() {
        // knots: [0,0,0,0, 12.5, 25.0, 37.5, 50,50,50,50]
        let knots = clamped_knots(50.0, 3);

        // Reference values (5-pt GL, from Python)
        struct Case { q: f64, j: usize, expected: f64 }
        let cases = [
            Case { q: 0.00, j: 0, expected: 39.269_908_169_9 },
            Case { q: 0.00, j: 3, expected: 157.079_632_679_5 },
            Case { q: 0.05, j: 0, expected: 39.100_177_000_4 },
            Case { q: 0.05, j: 3, expected: 117.328_635_484_0 },
            Case { q: 0.05, j: 6, expected: 11.481_906_632_6 },
            Case { q: 0.20, j: 0, expected: 36.716_716_170_4 },
            Case { q: 0.20, j: 1, expected: 49.687_860_411_5 },
            Case { q: 0.20, j: 3, expected: -9.820_720_300_5 },
        ];

        for c in &cases {
            let got = integrate_basis_sinc(&knots, 3, c.j, c.q);
            assert!(
                (got - c.expected).abs() < 1e-6,
                "q={}, j={}: got {:.10}, expected {:.10}, diff={:.2e}",
                c.q, c.j, got, c.expected,
                (got - c.expected).abs()
            );
        }
    }

    /// Comprehensive kernel match against Python 5-pt GL reference on a
    /// realistic SAXS parameter set: r_max=150 Å, n_interior=18 (20 free
    /// parameters), q spanning 0.01–0.50 Å⁻¹.
    ///
    /// Both Rust and Python use identical 5-point GL nodes/weights, so the
    /// results should agree to floating-point round-off (< 1e-8 absolute).
    /// The todo specifies < 1e-5; we are comfortably within that.
    #[test]
    fn kernel_saxs_realistic() {
        // r_max=150, n_interior=18 → 22 total basis fns, 20 free
        let knots = clamped_knots(150.0, 18);

        struct Case { q: f64, j: usize, expected: f64 }
        // Reference values generated by Dev/spline_pr_tests.py (5-pt GL, Python)
        let cases = [
            Case { q: 0.01, j:  1, expected:  49.580_055_334_037 },
            Case { q: 0.01, j:  5, expected:  97.534_180_650_444 },
            Case { q: 0.01, j: 10, expected:  91.039_585_896_515 },
            Case { q: 0.01, j: 15, expected:  80.185_120_765_809 },
            Case { q: 0.01, j: 20, expected:  33.907_173_341_474 },
            Case { q: 0.05, j:  1, expected:  49.007_359_019_168 },
            Case { q: 0.05, j:  5, expected:  62.525_370_712_647 },
            Case { q: 0.05, j: 10, expected: -10.545_006_180_078 },
            Case { q: 0.05, j: 15, expected: -12.150_265_783_268 },
            Case { q: 0.05, j: 20, expected:   5.618_214_405_505 },
            Case { q: 0.10, j:  1, expected:  47.269_157_112_391 },
            Case { q: 0.10, j:  5, expected:   1.521_585_125_864 },
            Case { q: 0.10, j: 10, expected:   8.989_943_956_808 },
            Case { q: 0.10, j: 15, expected:  -8.092_895_280_745 },
            Case { q: 0.10, j: 20, expected:   3.059_657_454_506 },
            Case { q: 0.20, j:  1, expected:  41.036_368_068_609 },
            Case { q: 0.20, j:  5, expected:  -1.142_643_055_737 },
            Case { q: 0.20, j: 10, expected:   4.579_071_684_723 },
            Case { q: 0.20, j: 15, expected:  -0.219_487_613_798 },
            Case { q: 0.20, j: 20, expected:  -1.053_004_682_761 },
            Case { q: 0.30, j:  1, expected:  32.798_671_738_670 },
            Case { q: 0.30, j:  5, expected:   0.693_172_051_204 },
            Case { q: 0.30, j: 10, expected:   1.217_376_527_762 },
            Case { q: 0.30, j: 15, expected:   1.113_416_898_118 },
            Case { q: 0.30, j: 20, expected:  -0.303_154_238_063 },
            Case { q: 0.50, j:  1, expected:  17.834_970_078_671 },
            Case { q: 0.50, j:  5, expected:   0.107_511_372_034 },
            Case { q: 0.50, j: 10, expected:  -0.089_906_179_010 },
            Case { q: 0.50, j: 15, expected:  -0.083_391_987_927 },
            Case { q: 0.50, j: 20, expected:  -0.168_592_160_282 },
        ];

        for c in &cases {
            let got = integrate_basis_sinc(&knots, 3, c.j, c.q);
            assert!(
                (got - c.expected).abs() < 1e-5,
                "SAXS case q={}, j={}: got {:.12}, expected {:.12}, diff={:.2e}",
                c.q, c.j, got, c.expected, (got - c.expected).abs()
            );
        }
    }

    /// sinc_kernel_matrix produces the same values as integrate_basis_sinc
    /// called individually — checks the matrix assembly, not the integration.
    #[test]
    fn kernel_matrix_consistency() {
        let knots = clamped_knots(80.0, 6);
        let degree = 3;
        let q: Vec<f64> = [0.0, 0.05, 0.1, 0.3].to_vec();
        let k = sinc_kernel_matrix(&knots, degree, &q);
        let n_basis = knots.len() - degree - 1;

        for (i, &qi) in q.iter().enumerate() {
            for j in 0..n_basis {
                let expected = integrate_basis_sinc(&knots, degree, j, qi);
                let got = k[(i, j)];
                assert!(
                    (got - expected).abs() < 1e-13,
                    "K[{i},{j}] q={qi}: matrix={got}, scalar={expected}"
                );
            }
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
