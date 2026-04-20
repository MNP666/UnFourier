use nalgebra::{DMatrix, DVector};

/// Strategy for enforcing non-negativity of P(r) coefficients.
///
/// The trait's single responsibility is identifying *which* coefficients
/// are currently violating the constraint. The solver owns the re-solve loop
/// and accumulates the fixed-zero set across iterations. This separation means
/// swapping strategies (e.g. replacing iterative clipping with NNLS) only
/// requires a new implementation of this trait, not changes to the solver.
///
/// # How the solver uses this
///
/// ```text
/// fixed = {}
/// loop (up to max_iter):
///     c = solve_tikhonov(fixed_to_zero = fixed)
///     violations = strategy.find_violations(&c)
///     if violations.is_empty(): break
///     fixed ∪= violations
/// ```
///
/// For NNLS the solver uses [`projected_gradient_nnls`] directly rather than
/// the find-violations loop; `is_constraining()` signals which path to take.
pub trait NonNegativityStrategy: Send + Sync {
    fn name(&self) -> &str;

    /// Return the indices of coefficients that violate the non-negativity
    /// constraint. An empty `Vec` means the current solution is feasible.
    fn find_violations(&self, coeffs: &[f64]) -> Vec<usize>;

    /// Whether this strategy enforces non-negativity.
    /// `NoConstraint` returns `false`; all others return `true`.
    fn is_constraining(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// NoConstraint
// ---------------------------------------------------------------------------

/// No non-negativity enforcement. The solution may contain negative P(r) values.
///
/// Useful for debugging (isolates regularisation from constraint effects) and
/// as the default for `LeastSquaresSvd` in M1.
pub struct NoConstraint;

impl NonNegativityStrategy for NoConstraint {
    fn name(&self) -> &str {
        "none"
    }

    fn find_violations(&self, _coeffs: &[f64]) -> Vec<usize> {
        vec![]
    }

    fn is_constraining(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// IterativeClipping
// ---------------------------------------------------------------------------

/// Enforce non-negativity by iteratively fixing negative coefficients to zero
/// and re-solving.
///
/// On each iteration, all currently negative coefficients are added to the
/// fixed-zero set and the system is re-solved on the remaining active indices.
/// This is repeated until no negative coefficients remain or `max_iter` is
/// reached.
///
/// This is a simple projected-iteration approach: not guaranteed to give the
/// true NNLS solution, but converges quickly in practice for smooth, well-
/// regularised P(r) (typically 1–3 iterations).
///
/// # Superseded by
///
/// [`ProjectedGradient`] is now the default strategy. IterativeClipping is
/// retained for backward compatibility and as a reference baseline.
pub struct IterativeClipping;

impl NonNegativityStrategy for IterativeClipping {
    fn name(&self) -> &str {
        "iterative-clipping"
    }

    /// Returns indices of all strictly negative coefficients.
    fn find_violations(&self, coeffs: &[f64]) -> Vec<usize> {
        coeffs
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c < 0.0)
            .map(|(i, _)| i)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ProjectedGradient
// ---------------------------------------------------------------------------

/// Enforce non-negativity by projected gradient descent (NNLS).
///
/// Solves `min_{c ≥ 0} ½ cᵀAc − bᵀc` directly, where
/// `A = KᵀK + λ_eff LᵀL` and `b = KᵀI`. Unlike iterative clipping, no
/// coefficient is permanently zeroed — the algorithm converges to the true
/// NNLS solution even when the unconstrained solution has large oscillations.
///
/// The actual optimisation loop lives in [`projected_gradient_nnls`], which
/// is called directly by the solver. This struct acts as the strategy selector
/// and exposes `find_violations` for diagnostic use.
pub struct ProjectedGradient {
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for ProjectedGradient {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-8,
        }
    }
}

impl NonNegativityStrategy for ProjectedGradient {
    fn name(&self) -> &str {
        "projected-gradient"
    }

    /// Returns indices of all strictly negative coefficients (sign check).
    /// Used for diagnostics; the solver calls [`projected_gradient_nnls`] directly.
    fn find_violations(&self, coeffs: &[f64]) -> Vec<usize> {
        coeffs
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c < 0.0)
            .map(|(i, _)| i)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// projected_gradient_nnls
// ---------------------------------------------------------------------------

/// Solve `min_{c ≥ 0} ½ cᵀAc − bᵀc` via projected gradient descent.
///
/// # Algorithm
///
/// 1. **Warm start:** clamp `warm_start` element-wise to `[0, ∞)`.
/// 2. **Step size:** `α = 1 / trace(A)`. Since `trace(A) ≥ λ_max(A)`, this
///    guarantees a descent step with no eigendecomposition required.
/// 3. **Loop:** `c ← max(c − α ∇f(c), 0)` where `∇f = Ac − b`.
///    Stop when `‖c_new − c‖ < tol` or `max_iter` is reached.
///
/// # Why not iterative clipping?
///
/// Iterative clipping adds all negative bins to a fixed-zero set
/// simultaneously. When the unconstrained solution oscillates (under-
/// regularised), this cascade zeros out most bins, leaving an unphysical
/// "island" pattern. Projected gradient never permanently zeros any bin —
/// it converges to the true constrained minimum.
///
/// # Parallelism note
///
/// Each call is stateless and independent of other λ evaluations.
/// The surrounding λ grid loop is a natural target for `rayon::par_iter()`.
pub fn projected_gradient_nnls(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    warm_start: &DVector<f64>,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    // Step size: 1/trace(A) ≤ 1/λ_max(A) → always a valid descent step.
    let step = 1.0 / a.trace().max(1e-10);

    let mut c = warm_start.map(|x: f64| x.max(0.0));

    for _ in 0..max_iter {
        let grad = a * &c - b;
        let c_new = (&c - step * &grad).map(|x: f64| x.max(0.0));
        let delta = (&c_new - &c).norm();
        c = c_new;
        if delta < tol {
            break;
        }
    }

    c.iter().cloned().collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_gradient_find_violations_returns_negative_indices() {
        let pg = ProjectedGradient::default();
        assert_eq!(pg.find_violations(&[-1.0, 2.0, -0.5]), vec![0, 2]);
    }

    #[test]
    fn projected_gradient_nnls_recovers_nonneg_solution() {
        // 3-variable system: K = I (3×3), λ = 0 → A = I, b = [0.1, 2.0, -0.5].
        // Unconstrained solution: c = [0.1, 2.0, -0.5].
        // NNLS solution: c = [0.1, 2.0, 0.0] (last bin clamped to zero).
        //
        // With A = I and warm start = [0.1, 2.0, 0.0]:
        //   grad = A c - b = [0.0, 0.0, 0.5]
        //   step = 1/trace(I) = 1/3
        //   c_new = max([0.1, 2.0, -0.167], 0) = [0.1, 2.0, 0.0]  → converged in 1 step.
        let a = DMatrix::<f64>::identity(3, 3);
        let b = DVector::from_column_slice(&[0.1, 2.0, -0.5]);
        let warm = DVector::from_column_slice(&[0.1, 2.0, -0.5]);

        let c = projected_gradient_nnls(&a, &b, &warm, 500, 1e-8);

        assert!(
            c.iter().all(|&x| x >= 0.0),
            "all coefficients must be non-negative"
        );
        assert!(
            (c[0] - 0.1).abs() < 1e-6,
            "c[0] should be ~0.1, got {}",
            c[0]
        );
        assert!(
            (c[1] - 2.0).abs() < 1e-6,
            "c[1] should be ~2.0, got {}",
            c[1]
        );
        assert!(c[2].abs() < 1e-6, "c[2] should be ~0.0, got {}", c[2]);
    }

    #[test]
    fn no_constraint_is_not_constraining() {
        assert!(!NoConstraint.is_constraining());
    }

    #[test]
    fn projected_gradient_is_constraining() {
        assert!(ProjectedGradient::default().is_constraining());
    }
}
