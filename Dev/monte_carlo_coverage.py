#!/usr/bin/env python3
"""
monte_carlo_coverage.py — M4 validation: statistical calibration of Bayesian error bars.

Runs unfourier --method bayes on N independent noise realisations of Debye data
and checks that the true P(r) falls within the posterior ±1σ error bars ~68% of
the time at each r point.

A well-calibrated posterior gives ~68% coverage everywhere. Under-coverage
(< 68%) means error bars are too narrow (over-confident); over-coverage (> 68%)
means too wide (conservative). Regions of systematic under-coverage at fixed r
indicate regularisation bias that the posterior does not account for.

The "true P(r)" is the analytic Gaussian-chain formula, normalised so that
4π ∫ P(r) dr = I(0) = 1 — the same normalisation implied by the unfourier
kernel K_ij = 4π sinc(q_i r_j) Δr, so absolute units are directly comparable.

Usage
-----
    python Dev/monte_carlo_coverage.py [--n 200] [--k 5] [--rg 30] [--rmax 180]
    python Dev/monte_carlo_coverage.py --n 50 --no-plot   # quick headless check

Requirements
------------
    numpy, matplotlib
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Debye / Gaussian chain helpers (inline — avoids subprocess for truth)
# ---------------------------------------------------------------------------

def debye_intensity(q: np.ndarray, Rg: float) -> np.ndarray:
    """Debye form factor for a Gaussian chain, normalised so I(0) = 1."""
    x = (q * Rg) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            x < 1e-8,
            1.0 - x / 3.0,
            2.0 * (np.exp(-x) - 1.0 + x) / x ** 2,
        )


def pr_analytic(r: np.ndarray, Rg: float) -> np.ndarray:
    """
    Analytic P(r) for a Gaussian chain, normalised so 4π ∫ P(r) dr = I(0) = 1.

        P(r) = A · r² · exp(-3r² / 4Rg²)
        A    = 3√3 / (8π^{3/2} Rg³)

    This matches the absolute units of unfourier's coefficient vector because
    the kernel K_ij = 4π sinc(q_i r_j) Δr absorbs the integration measure,
    leaving coefficients c_j ≈ P(r_j).
    """
    A = 3.0 * np.sqrt(3.0) / (8.0 * np.pi ** 1.5 * Rg ** 3)
    return A * r ** 2 * np.exp(-3.0 * r ** 2 / (4.0 * Rg ** 2))


def write_dat(path: Path, q: np.ndarray, I: np.ndarray, sigma: np.ndarray) -> None:
    """Write a 3-column .dat file in the format unfourier expects."""
    with open(path, "w") as f:
        f.write("# q(1/A)  I(q)  sigma(q)\n")
        for qi, Ii, si in zip(q, I, sigma):
            f.write(f"  {qi:.8e}  {Ii:.8e}  {si:.8e}\n")


# ---------------------------------------------------------------------------
# unfourier interface
# ---------------------------------------------------------------------------

def run_unfourier(
    binary: str,
    dat_file: Path,
    pr_out: Path,
    rmax: float,
    npoints: int,
    lambda_count: int = 60,
) -> bool:
    """Run unfourier --method bayes. Returns True on success."""
    cmd = [
        binary,
        str(dat_file),
        "--method", "bayes",
        "--rmax", str(rmax),
        "--npoints", str(npoints),
        "--lambda-count", str(lambda_count),
        "--output", str(pr_out),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def load_pr_output(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load r, P(r), sigma_P(r) from a 3-column unfourier output file.
    Returns None if the file is missing or does not have 3 columns.
    """
    try:
        data = np.loadtxt(path, comments="#")
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] < 3:
        return None
    return data[:, 0], data[:, 1], data[:, 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n", type=int, default=200, metavar="N",
        help="Number of noise realisations (default: 200)",
    )
    parser.add_argument(
        "--k", type=float, default=5.0, metavar="K",
        help="Noise level: σ(q) = I(q)/k (default: 5  →  ~20%% relative error)",
    )
    parser.add_argument(
        "--rg", type=float, default=30.0, metavar="RG",
        help="Radius of gyration in Å (default: 30)",
    )
    parser.add_argument(
        "--qmin", type=float, default=0.01, help="q_min in Å⁻¹ (default: 0.01)",
    )
    parser.add_argument(
        "--qmax", type=float, default=0.50, help="q_max in Å⁻¹ (default: 0.5)",
    )
    parser.add_argument(
        "--nq", type=int, default=200, help="Number of q points (default: 200)",
    )
    parser.add_argument(
        "--rmax", type=float, default=180.0, help="r_max for unfourier in Å (default: 180)",
    )
    parser.add_argument(
        "--npoints", type=int, default=100,
        help="r grid points for unfourier (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed; run i uses seed+i (default: 0)",
    )
    parser.add_argument(
        "--unfourier",
        default=str(REPO_ROOT / "target" / "release" / "unfourier"),
        help="Path to the unfourier binary",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip matplotlib output",
    )
    parser.add_argument(
        "--save", default=None, metavar="FILE",
        help="Save plot to this path instead of (or as well as) showing it",
    )
    args = parser.parse_args()

    if not Path(args.unfourier).exists():
        sys.exit(
            f"ERROR: unfourier binary not found at '{args.unfourier}'.\n"
            "Build it first with:  cargo build --release"
        )

    # ---- q grid and analytic truth ------------------------------------------
    q = np.linspace(args.qmin, args.qmax, args.nq)
    I_true = debye_intensity(q, args.rg)

    # r grid: must match what unfourier produces — bin centres at (j+0.5)*Δr
    delta_r = args.rmax / args.npoints
    r_grid  = (np.arange(args.npoints) + 0.5) * delta_r
    P_true  = pr_analytic(r_grid, args.rg)

    # ---- Monte Carlo loop ---------------------------------------------------
    n_r = args.npoints
    within_1sigma = np.zeros(n_r, dtype=np.int64)
    pr_all        = []
    sigma_all     = []
    n_success     = 0

    print(
        f"Monte Carlo coverage test\n"
        f"  N = {args.n} realisations,  k = {args.k},  Rg = {args.rg} Å,  "
        f"r_max = {args.rmax} Å,  n_points = {args.npoints}"
    )
    print(f"  binary: {args.unfourier}")
    print()

    with tempfile.TemporaryDirectory(prefix="unfourier_mc_") as tmpdir:
        tmp     = Path(tmpdir)
        dat_file = tmp / "data.dat"
        pr_out   = tmp / "pr.dat"

        for i in range(args.n):
            # New noise realisation with a deterministic seed per run
            rng     = np.random.default_rng(args.seed + i)
            sigma_q = I_true / args.k
            I_noisy = I_true + rng.normal(scale=sigma_q)
            write_dat(dat_file, q, I_noisy, sigma_q)

            if not run_unfourier(args.unfourier, dat_file, pr_out, args.rmax, args.npoints):
                print(f"  [{i+1:>4}/{args.n}] unfourier failed — skipping", file=sys.stderr)
                continue

            result = load_pr_output(pr_out)
            if result is None:
                print(
                    f"  [{i+1:>4}/{args.n}] output missing or 2-column (no error bars) — skipping",
                    file=sys.stderr,
                )
                continue

            _, pr_calc, pr_sigma = result
            if len(pr_calc) != n_r:
                print(
                    f"  [{i+1:>4}/{args.n}] r-grid size mismatch "
                    f"({len(pr_calc)} vs {n_r}) — skipping",
                    file=sys.stderr,
                )
                continue

            within_1sigma += (np.abs(pr_calc - P_true) <= pr_sigma).astype(np.int64)
            pr_all.append(pr_calc)
            sigma_all.append(pr_sigma)
            n_success += 1

            if (i + 1) % 20 == 0 or (i + 1) == args.n:
                pct = 100 * (i + 1) / args.n
                print(f"  [{i+1:>4}/{args.n}]  {pct:.0f}%  ({n_success} succeeded)",
                      flush=True)

    if n_success == 0:
        sys.exit("ERROR: all runs failed — check the unfourier binary and input data.")

    # ---- Coverage statistics ------------------------------------------------
    coverage = within_1sigma / n_success

    # Binomial 95% CI on the coverage estimate at each r point:
    # σ_coverage ≈ sqrt(p*(1-p)/N),  95% CI uses z = 1.96
    p_target = 0.6827  # P(|Z| < 1) for a standard normal
    ci_half  = 1.96 * np.sqrt(p_target * (1.0 - p_target) / n_success)
    lo, hi   = p_target - ci_half, p_target + ci_half

    under_mask = coverage < lo
    over_mask  = coverage > hi
    n_under    = under_mask.sum()
    n_over     = over_mask.sum()

    print()
    print(f"Results ({n_success} realisations):")
    print(f"  Mean coverage:      {100 * coverage.mean():.1f}%  (target: {100*p_target:.1f}%)")
    print(f"  Coverage range:     [{100*coverage.min():.1f}%, {100*coverage.max():.1f}%]")
    print(f"  95% CI on target:   [{100*lo:.1f}%, {100*hi:.1f}%]")
    print(f"  Under-confident r-points (coverage < {100*lo:.1f}%): {n_under} / {n_r}")
    print(f"  Over-confident  r-points (coverage > {100*hi:.1f}%): {n_over}  / {n_r}")

    if args.no_plot and args.save is None:
        return

    # ---- Plot ---------------------------------------------------------------
    pr_mean    = np.mean(pr_all,    axis=0)
    sigma_mean = np.mean(sigma_all, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    fig.suptitle(
        f"Bayesian IFT — Monte Carlo coverage  "
        f"(N = {n_success}, k = {args.k}, Rg = {args.rg} Å)",
        fontsize=13,
    )

    # -- Panel 1: coverage fraction vs r --------------------------------------
    ax = axes[0]
    ax.axhspan(lo, hi, color="green", alpha=0.15,
               label=f"95% binomial CI  [{100*lo:.0f}%–{100*hi:.0f}%]")
    ax.axhline(p_target, color="green", lw=1.5, ls="--",
               label=f"Target {100*p_target:.1f}%  (1σ)")
    ax.plot(r_grid, coverage, color="steelblue", lw=1.5, label="Empirical coverage")

    if n_under > 0:
        ax.plot(r_grid[under_mask], coverage[under_mask],
                "rv", ms=5, label=f"Under-confident  ({n_under} pts)")
    if n_over > 0:
        ax.plot(r_grid[over_mask], coverage[over_mask],
                "g^", ms=5, label=f"Over-confident  ({n_over} pts)")

    ax.set_xlabel("r (Å)")
    ax.set_ylabel("Coverage fraction")
    ax.set_title("Fraction of realisations where  |P_calc(r) − P_true(r)| ≤ σ_P(r)")
    ax.set_xlim(0, args.rmax)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="lower right")

    # -- Panel 2: mean P(r) ± mean σ vs analytic truth -----------------------
    ax = axes[1]
    ax.plot(r_grid, P_true, "k--", lw=2, label="P_true  (analytic Gaussian chain)", zorder=5)
    ax.plot(r_grid, pr_mean, color="steelblue", lw=1.5, label="Mean P_calc")
    ax.fill_between(
        r_grid,
        (pr_mean - sigma_mean).clip(0),
        pr_mean + sigma_mean,
        color="steelblue", alpha=0.30, label="Mean P_calc ± mean σ",
    )
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("P(r)")
    ax.set_title("Mean recovered P(r) ± mean posterior σ vs analytic truth")
    ax.set_xlim(0, args.rmax)
    ax.set_ylim(bottom=-0.05 * float(P_true.max()))
    ax.legend(fontsize=9)

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"\nPlot saved to '{args.save}'")
    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
