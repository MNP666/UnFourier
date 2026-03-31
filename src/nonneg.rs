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
/// For NNLS in a future milestone, `find_violations` would check the
/// KKT conditions (dual feasibility) rather than simple sign tests.
pub trait NonNegativityStrategy: Send + Sync {
    fn name(&self) -> &str;

    /// Return the indices of coefficients that violate the non-negativity
    /// constraint. An empty `Vec` means the current solution is feasible.
    fn find_violations(&self, coeffs: &[f64]) -> Vec<usize>;
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
/// # Future replacement
///
/// Replace with an `Nnls` implementation that uses the Lawson-Hanson active-set
/// algorithm or a similar NNLS solver. The interface is identical — only
/// `find_violations` changes to check KKT conditions.
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
