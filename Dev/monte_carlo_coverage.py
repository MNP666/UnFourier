#!/usr/bin/env python3
"""
monte_carlo_coverage.py - Bayesian spline error-bar coverage check.

Runs unfourier --method bayes on independent noisy Debye-chain data sets and
checks whether the analytic Gaussian-chain P(r) falls within the reported
posterior +/-1 sigma band at each emitted spline output point.

This script is deliberately spline-output based. It does not construct a
legacy sampled r grid, and it passes the spline basis count rather than the
removed grid-size option. The first successful unfourier run defines the
validation r grid; subsequent runs must emit the same grid.

The reported coverage is a diagnostic of the current posterior approximation.
Under-coverage at fixed r usually indicates regularisation bias that is not
represented in the posterior covariance.

Usage
-----
    python Dev/monte_carlo_coverage.py --n 200 --k 5 --n-basis 20
    python Dev/monte_carlo_coverage.py --n 50 --no-plot

Requirements
------------
    numpy, matplotlib
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unfourier_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# Debye / Gaussian-chain helpers
# ---------------------------------------------------------------------------


def debye_intensity(q: np.ndarray, rg: float) -> np.ndarray:
    """Debye form factor for a Gaussian chain, normalised so I(0) = 1."""
    x = (q * rg) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            x < 1e-8,
            1.0 - x / 3.0,
            2.0 * (np.exp(-x) - 1.0 + x) / x**2,
        )


def pr_analytic(r: np.ndarray, rg: float) -> np.ndarray:
    """
    Analytic P(r) for a Gaussian chain, normalised so 4*pi integral P(r) dr = 1.

        P(r) = A * r^2 * exp(-3r^2 / 4Rg^2)
        A    = 3sqrt(3) / (8*pi^(3/2)*Rg^3)

    The analytic curve is evaluated on the r grid written by unfourier, so this
    remains valid for spline output grids and boundary modes.
    """
    a = 3.0 * np.sqrt(3.0) / (8.0 * np.pi**1.5 * rg**3)
    return a * r**2 * np.exp(-3.0 * r**2 / (4.0 * rg**2))


def write_dat(path: Path, q: np.ndarray, intensity: np.ndarray, sigma: np.ndarray) -> None:
    """Write a 3-column .dat file in the format unfourier expects."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# q(1/A)  I(q)  sigma(q)\n")
        for qi, ii, si in zip(q, intensity, sigma):
            handle.write(f"  {qi:.8e}  {ii:.8e}  {si:.8e}\n")


# ---------------------------------------------------------------------------
# unfourier interface
# ---------------------------------------------------------------------------


def run_unfourier(
    binary: str,
    dat_file: Path,
    pr_out: Path,
    rmax: float,
    n_basis: int,
    lambda_count: int,
) -> bool:
    """Run unfourier --method bayes. Returns True on success."""
    cmd = [
        binary,
        str(dat_file),
        "--method",
        "bayes",
        "--rmax",
        str(rmax),
        "--n-basis",
        str(n_basis),
        "--lambda-count",
        str(lambda_count),
        "--negative-handling",
        "clip",
        "--output",
        str(pr_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=dat_file.parent)
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
        "--n",
        type=int,
        default=200,
        metavar="N",
        help="Number of noise realisations (default: 200)",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=5.0,
        metavar="K",
        help="Noise level: sigma(q) = I(q)/k (default: 5, about 20%% relative error)",
    )
    parser.add_argument(
        "--rg",
        type=float,
        default=30.0,
        metavar="RG",
        help="Radius of gyration in Angstrom (default: 30)",
    )
    parser.add_argument(
        "--qmin", type=float, default=0.01, help="q_min in inverse Angstrom (default: 0.01)"
    )
    parser.add_argument(
        "--qmax", type=float, default=0.50, help="q_max in inverse Angstrom (default: 0.5)"
    )
    parser.add_argument(
        "--nq", type=int, default=200, help="Number of q points (default: 200)"
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=180.0,
        help="r_max for unfourier in Angstrom (default: 180)",
    )
    parser.add_argument(
        "--n-basis",
        type=int,
        default=20,
        help="Number of spline basis functions for unfourier (default: 20)",
    )
    parser.add_argument(
        "--lambda-count",
        type=int,
        default=60,
        help="Number of lambda candidates for Bayesian evidence (default: 60)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed; run i uses seed+i (default: 0)",
    )
    parser.add_argument(
        "--unfourier",
        default=str(REPO_ROOT / "target" / "release" / "unfourier"),
        help="Path to the unfourier binary",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib output")
    parser.add_argument(
        "--save",
        default=None,
        metavar="FILE",
        help="Save plot to this path instead of, or as well as, showing it",
    )
    args = parser.parse_args()

    if not Path(args.unfourier).exists():
        sys.exit(
            f"ERROR: unfourier binary not found at '{args.unfourier}'.\n"
            "Build it first with: cargo build --release"
        )

    q = np.linspace(args.qmin, args.qmax, args.nq)
    intensity_true = debye_intensity(q, args.rg)

    r_grid: np.ndarray | None = None
    p_true: np.ndarray | None = None
    within_1sigma: np.ndarray | None = None
    pr_all: list[np.ndarray] = []
    sigma_all: list[np.ndarray] = []
    n_success = 0

    print(
        f"Monte Carlo coverage test\n"
        f"  N = {args.n} realisations,  k = {args.k},  Rg = {args.rg} Angstrom,  "
        f"r_max = {args.rmax} Angstrom,  n_basis = {args.n_basis}"
    )
    print(f"  binary: {args.unfourier}")
    print()

    with tempfile.TemporaryDirectory(prefix="unfourier_mc_") as tmpdir:
        tmp = Path(tmpdir)
        dat_file = tmp / "data.dat"
        pr_out = tmp / "pr.dat"

        for i in range(args.n):
            rng = np.random.default_rng(args.seed + i)
            sigma_q = intensity_true / args.k
            intensity_noisy = intensity_true + rng.normal(scale=sigma_q)
            write_dat(dat_file, q, intensity_noisy, sigma_q)

            ok = run_unfourier(
                args.unfourier,
                dat_file,
                pr_out,
                args.rmax,
                args.n_basis,
                args.lambda_count,
            )
            if not ok:
                print(f"  [{i + 1:>4}/{args.n}] unfourier failed, skipping", file=sys.stderr)
                continue

            result = load_pr_output(pr_out)
            if result is None:
                print(
                    f"  [{i + 1:>4}/{args.n}] output missing or 2-column, skipping",
                    file=sys.stderr,
                )
                continue

            r, pr_calc, pr_sigma = result
            if r_grid is None:
                r_grid = r
                p_true = pr_analytic(r_grid, args.rg)
                within_1sigma = np.zeros(len(r_grid), dtype=np.int64)
            elif len(r) != len(r_grid) or not np.allclose(r, r_grid, rtol=1e-10, atol=1e-10):
                print(
                    f"  [{i + 1:>4}/{args.n}] r-grid mismatch, skipping",
                    file=sys.stderr,
                )
                continue

            assert p_true is not None
            assert within_1sigma is not None
            within_1sigma += (np.abs(pr_calc - p_true) <= pr_sigma).astype(np.int64)
            pr_all.append(pr_calc)
            sigma_all.append(pr_sigma)
            n_success += 1

            if (i + 1) % 20 == 0 or (i + 1) == args.n:
                pct = 100 * (i + 1) / args.n
                print(f"  [{i + 1:>4}/{args.n}]  {pct:.0f}%  ({n_success} succeeded)", flush=True)

    if n_success == 0 or r_grid is None or p_true is None or within_1sigma is None:
        sys.exit("ERROR: all runs failed. Check the unfourier binary and input data.")

    n_r = len(r_grid)
    coverage = within_1sigma / n_success

    p_target = 0.6827
    ci_half = 1.96 * np.sqrt(p_target * (1.0 - p_target) / n_success)
    lo, hi = p_target - ci_half, p_target + ci_half

    under_mask = coverage < lo
    over_mask = coverage > hi
    n_under = int(under_mask.sum())
    n_over = int(over_mask.sum())

    print()
    print(f"Results ({n_success} realisations, {n_r} emitted r points):")
    print(f"  Mean coverage:      {100 * coverage.mean():.1f}%  (target: {100 * p_target:.1f}%)")
    print(f"  Coverage range:     [{100 * coverage.min():.1f}%, {100 * coverage.max():.1f}%]")
    print(f"  95% CI on target:   [{100 * lo:.1f}%, {100 * hi:.1f}%]")
    print(f"  Under-confident r-points (coverage < {100 * lo:.1f}%): {n_under} / {n_r}")
    print(f"  Over-confident  r-points (coverage > {100 * hi:.1f}%): {n_over} / {n_r}")

    if args.no_plot and args.save is None:
        return

    pr_mean = np.mean(pr_all, axis=0)
    sigma_mean = np.mean(sigma_all, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    fig.suptitle(
        f"Bayesian IFT - Monte Carlo coverage "
        f"(N = {n_success}, k = {args.k}, Rg = {args.rg} Angstrom)",
        fontsize=13,
    )

    ax = axes[0]
    ax.axhspan(
        lo,
        hi,
        color="green",
        alpha=0.15,
        label=f"95% binomial CI [{100 * lo:.0f}%-{100 * hi:.0f}%]",
    )
    ax.axhline(
        p_target,
        color="green",
        lw=1.5,
        ls="--",
        label=f"Target {100 * p_target:.1f}% (1 sigma)",
    )
    ax.plot(r_grid, coverage, color="steelblue", lw=1.5, label="Empirical coverage")

    if n_under > 0:
        ax.plot(r_grid[under_mask], coverage[under_mask], "rv", ms=5, label=f"Under {n_under}")
    if n_over > 0:
        ax.plot(r_grid[over_mask], coverage[over_mask], "g^", ms=5, label=f"Over {n_over}")

    ax.set_xlabel("r (Angstrom)")
    ax.set_ylabel("Coverage fraction")
    ax.set_title("Fraction of realisations where abs(P_calc(r) - P_true(r)) <= sigma_P(r)")
    ax.set_xlim(0, args.rmax)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="lower right")

    ax = axes[1]
    ax.plot(r_grid, p_true, "k--", lw=2, label="P_true (analytic Gaussian chain)", zorder=5)
    ax.plot(r_grid, pr_mean, color="steelblue", lw=1.5, label="Mean P_calc")
    ax.fill_between(
        r_grid,
        (pr_mean - sigma_mean).clip(0),
        pr_mean + sigma_mean,
        color="steelblue",
        alpha=0.30,
        label="Mean P_calc +/- mean sigma",
    )
    ax.set_xlabel("r (Angstrom)")
    ax.set_ylabel("P(r)")
    ax.set_title("Mean recovered P(r) +/- mean posterior sigma vs analytic truth")
    ax.set_xlim(0, args.rmax)
    ax.set_ylim(bottom=-0.05 * float(p_true.max()))
    ax.legend(fontsize=9)

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"\nPlot saved to '{args.save}'")
    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
