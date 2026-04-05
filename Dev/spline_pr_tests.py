"""
B-spline basis prototype for IFT of SAXS data.

Replaces the earlier CubicSpline interpolating approach with a proper B-spline
design matrix.  The boundary conditions P(0) = P(r_max) = 0 are enforced
structurally by using a clamped (open) knot vector and excluding the two
endpoint basis functions from the design matrix.

Sections
--------
1. make_bspline_basis   — build knot vector, design matrix, Greville abscissae
2. Boundary-condition verification
3. Kernel matrix construction via Gauss–Legendre quadrature
4. P(r) recovery on clean Debye data with scipy.optimize.nnls
5. Convergence check: n_interior_knots 10 → 50
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import nnls

# ---------------------------------------------------------------------------
# 1.  make_bspline_basis
# ---------------------------------------------------------------------------

def make_bspline_basis(r_max: float, n_interior_knots: int, degree: int = 3):
    """
    Build a clamped B-spline basis on [0, r_max].

    Parameters
    ----------
    r_max : float
        Right boundary.
    n_interior_knots : int
        Number of uniformly spaced interior knots (t_1 … t_m).
    degree : int
        Spline degree (default 3 = cubic).

    Returns
    -------
    knots : ndarray of shape (n_interior_knots + 2*(degree+1),)
        Full clamped knot vector.
    B_free : callable  (r_eval: ndarray) -> ndarray shape (len(r_eval), n_free)
        Design matrix for the *free* basis functions (all columns except the
        two endpoints), evaluated at arbitrary r points.
    xi_free : ndarray of shape (n_free,)
        Greville abscissae of the free basis functions.

    Notes
    -----
    n_total  = n_interior_knots + degree + 1
    n_free   = n_total - 2   (drop first and last)
    """
    d = degree
    interior = np.linspace(0.0, r_max, n_interior_knots + 2)[1:-1]  # strictly interior

    knots = np.concatenate([
        np.zeros(d + 1),
        interior,
        np.full(d + 1, r_max),
    ])

    n_total = len(knots) - d - 1  # number of B-spline basis functions

    # Greville abscissae:  ξ_j = mean(t_{j+1}, …, t_{j+d})
    xi_all = np.array([knots[j+1:j+d+1].mean() for j in range(n_total)])
    xi_free = xi_all[1:-1]  # drop endpoint abscissae

    def B_free(r_eval: np.ndarray) -> np.ndarray:
        """Return design matrix (n_r, n_free) excluding the two endpoint columns."""
        r_eval = np.asarray(r_eval)
        n_r = len(r_eval)
        n_free = n_total - 2
        B = np.zeros((n_r, n_free))
        for col_free, col_full in enumerate(range(1, n_total - 1)):
            # Build the col_full-th B-spline basis function
            c = np.zeros(n_total)
            c[col_full] = 1.0
            spl = BSpline(knots, c, d, extrapolate=False)
            vals = spl(r_eval)
            vals = np.where(np.isnan(vals), 0.0, vals)
            B[:, col_free] = vals
        return B

    return knots, B_free, xi_free


# ---------------------------------------------------------------------------
# 2.  Boundary-condition verification
# ---------------------------------------------------------------------------

r_max = 150.0
n_int = 20

knots, B_free_fn, xi_free = make_bspline_basis(r_max, n_int)

rng = np.random.default_rng(0)
c_test = rng.uniform(0.0, 1.0, len(xi_free))

r_check = np.array([0.0, r_max])
B_check = B_free_fn(r_check)
vals_bc = B_check @ c_test

print("=== Boundary condition check ===")
print(f"  P(0)     = {vals_bc[0]:.2e}  (should be ~0)")
print(f"  P(r_max) = {vals_bc[1]:.2e}  (should be ~0)")
assert abs(vals_bc[0]) < 1e-12, "P(0) != 0"
assert abs(vals_bc[1]) < 1e-12, "P(r_max) != 0"
print("  PASS\n")


# ---------------------------------------------------------------------------
# 3.  Kernel matrix via 5-point Gauss–Legendre quadrature
# ---------------------------------------------------------------------------

# 5-point GL nodes and weights on [-1, 1]
_GL_NODES = np.array([
    -0.9061798459386640,
    -0.5384693101056831,
     0.0,
     0.5384693101056831,
     0.9061798459386640,
])
_GL_WEIGHTS = np.array([
    0.2369268850561891,
    0.4786286704993665,
    0.5688888888888889,
    0.4786286704993665,
    0.2369268850561891,
])


def _sinc_kernel(q: float, r: np.ndarray) -> np.ndarray:
    """4π · sin(q r) / (q r), with the q→0 limit handled."""
    if q == 0.0:
        return 4.0 * np.pi * np.ones_like(r)
    qr = q * r
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(np.abs(qr) < 1e-8, 4.0 * np.pi, 4.0 * np.pi * np.sin(qr) / qr)
    return val


def build_kernel_matrix(q: np.ndarray, knots: np.ndarray, B_free_fn, degree: int = 3) -> np.ndarray:
    """
    Compute K[i, j] = 4π ∫ B_j(r) sin(q_i r) / (q_i r) dr

    Integration is by 5-point Gauss–Legendre over each knot span [t_k, t_{k+1}].
    Only spans with positive length are included.

    Returns
    -------
    K : ndarray of shape (n_q, n_free)
    """
    # unique internal knot spans
    unique_knots = np.unique(knots)
    spans = list(zip(unique_knots[:-1], unique_knots[1:]))

    n_q = len(q)
    # probe to get n_free
    n_free = B_free_fn(np.array([0.5])).shape[1]

    K = np.zeros((n_q, n_free))

    for (a, b) in spans:
        h = b - a
        if h <= 0.0:
            continue
        # Map GL nodes from [-1,1] to [a,b]
        r_pts = 0.5 * (b + a) + 0.5 * h * _GL_NODES   # shape (5,)
        w_pts = 0.5 * h * _GL_WEIGHTS                   # shape (5,)

        B_pts = B_free_fn(r_pts)   # (5, n_free)

        for i, qi in enumerate(q):
            sinc_vals = _sinc_kernel(qi, r_pts)          # (5,)
            integrand = sinc_vals[:, None] * B_pts       # (5, n_free)
            K[i] += integrand.T @ w_pts                  # (n_free,)

    return K


# ---------------------------------------------------------------------------
# 4.  P(r) recovery on clean Debye data
# ---------------------------------------------------------------------------

from pathlib import Path

_DATA_FILE = Path(__file__).parent / "debye_noiseless.dat"
_REF_FILE  = Path(__file__).parent / "debye_pr_ref.dat"

def load_dat(path):
    return np.loadtxt(path, comments="#")

print("=== P(r) recovery on clean Debye data ===")

if not _DATA_FILE.exists():
    print(f"  Data file {_DATA_FILE} not found — skipping recovery section.")
    _run_recovery = False
else:
    _run_recovery = True

if _run_recovery:
    data = load_dat(_DATA_FILE)
    q_data, I_data, sigma_data = data[:, 0], data[:, 1], data[:, 2]

    r_max_fit = 150.0   # Debye Rg=30, so P(r)~0 beyond ~6 Rg=180; use 150
    n_int_fit = 20

    knots_fit, B_free_fit, xi_fit = make_bspline_basis(r_max_fit, n_int_fit)
    K_fit = build_kernel_matrix(q_data, knots_fit, B_free_fit)

    # Weight by 1/sigma
    W = 1.0 / sigma_data
    K_w = W[:, None] * K_fit
    I_w = W * I_data

    c_nnls, residual = nnls(K_w, I_w)
    print(f"  n_free = {len(xi_fit)},  NNLS residual = {residual:.4f}")

    r_eval = np.linspace(0.0, r_max_fit, 500)
    B_eval = B_free_fit(r_eval)
    pr_nnls = B_eval @ c_nnls

    # Load reference if available
    _has_ref = _REF_FILE.exists()
    if _has_ref:
        ref = load_dat(_REF_FILE)
        r_ref, pr_ref = ref[:, 0], ref[:, 1]
        # normalise both to peak=1 for visual comparison
        pr_ref_n = pr_ref / pr_ref.max()
    pr_nnls_n = pr_nnls / pr_nnls.max() if pr_nnls.max() > 0 else pr_nnls

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r_eval, pr_nnls_n, label=f"B-spline NNLS (n_free={len(xi_fit)})")
    if _has_ref:
        ax.plot(r_ref, pr_ref_n, "--", label="Sine-transform reference", alpha=0.7)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("P(r)  [normalised]")
    ax.set_title("P(r) recovery — B-spline basis vs reference")
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "bspline_pr_recovery.png", dpi=120)
    print("  Plot saved to Dev/bspline_pr_recovery.png")
    plt.show()


# ---------------------------------------------------------------------------
# 5.  Convergence check: n_interior_knots 10 → 50
# ---------------------------------------------------------------------------

print("\n=== Convergence check ===")

if _run_recovery:
    knot_counts = list(range(5, 51, 5))
    residuals = []
    peak_r = []

    for n_int_cv in knot_counts:
        kn, Bf, xf = make_bspline_basis(r_max_fit, n_int_cv)
        K_cv = build_kernel_matrix(q_data, kn, Bf)
        K_w_cv = (1.0 / sigma_data)[:, None] * K_cv
        I_w_cv = (1.0 / sigma_data) * I_data
        c_cv, res_cv = nnls(K_w_cv, I_w_cv)
        residuals.append(res_cv)

        # Peak position of recovered P(r)
        r_cv = np.linspace(0.0, r_max_fit, 500)
        pr_cv = Bf(r_cv) @ c_cv
        peak_r.append(r_cv[np.argmax(pr_cv)])

        print(f"  n_interior={n_int_cv:3d}  n_free={len(xf):3d}  "
              f"residual={res_cv:8.4f}  peak_r={peak_r[-1]:.1f} Å")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(knot_counts, residuals, "o-")
    ax1.set_xlabel("n_interior_knots")
    ax1.set_ylabel("NNLS weighted residual")
    ax1.set_title("Fit residual vs knot count")

    ax2.plot(knot_counts, peak_r, "s-")
    ax2.set_xlabel("n_interior_knots")
    ax2.set_ylabel("Peak r (Å)")
    ax2.set_title("P(r) peak position vs knot count")

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "bspline_convergence.png", dpi=120)
    print("\n  Convergence plot saved to Dev/bspline_convergence.png")
    plt.show()
else:
    print("  Skipped (no data file).")
