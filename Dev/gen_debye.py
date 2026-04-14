#!/usr/bin/env python3
"""
Generate synthetic SAXS data for a Gaussian chain (Debye form factor).

The Debye formula for a Gaussian chain with radius of gyration Rg:

    I(q) = 2 * (exp(-x) - 1 + x) / x²    where x = (q * Rg)²

Normalised to I(0) = 1. I(q) decays monotonically with no zeros, making
σ(q) = I(q) / k well-behaved everywhere — this is the preferred noisy test
fixture compared to the sphere, which has near-zero minima that blow up
error weights.

A numerical reference P(r) is optionally written via the sine transform:

    P(r) = r / (2π²) ∫ q · I(q) · sin(qr) dq

Usage examples
--------------
# Noiseless:
    python gen_debye.py --rg 30 --output debye_noiseless.dat

# With noise (k=5 ≈ 20% relative error):
    python gen_debye.py --rg 30 --k 5 --output debye_k5.dat --pr-reference pr_ref.dat

# Log-spaced q (better for fitting over many decades):
    python gen_debye.py --rg 30 --k 5 --log-spacing --output debye_log.dat
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def debye_intensity(q: np.ndarray, Rg: float) -> np.ndarray:
    """
    Debye form factor for a Gaussian chain, normalised to I(0) = 1.

    I(q) = 2(e^{-x} - 1 + x) / x²   where x = (q·Rg)²

    The limit I(0) = 1 is applied via a series expansion for small x.
    """
    x = (q * Rg) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        I = np.where(
            x < 1e-8,
            1.0 - x / 3.0,           # Taylor expansion for x → 0
            2.0 * (np.exp(-x) - 1.0 + x) / x**2,
        )
    return I


def compute_pr_reference(
    q: np.ndarray,
    I: np.ndarray,
    r_max: float,
    n_r: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numerical P(r) via sine transform of I(q).

    Uses the relation:
        I(q) = 4π ∫ P(r) sin(qr)/(qr) dr
    ⟹  P(r) = r / (2π²) ∫ q · I(q) · sin(qr) dq

    Integration is over the provided q range using the trapezoidal rule.
    Truncation artifacts increase at large r where the integrand oscillates
    rapidly, so treat the tail of the reference P(r) as approximate.
    """
    r = np.linspace(0.5, r_max, n_r)
    pr = np.array([
        ri / (2.0 * np.pi**2) * np.trapz(q * I * np.sin(q * ri), q)
        for ri in r
    ])
    # Zero out small negatives from truncation noise
    pr = np.maximum(pr, 0.0)
    return r, pr


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rg", type=float, default=30.0, metavar="RG",
        help="Radius of gyration in Å (default: 30)",
    )
    parser.add_argument(
        "--qmin", type=float, default=0.01, metavar="Q",
        help="Minimum q in Å⁻¹ (default: 0.01)",
    )
    parser.add_argument(
        "--qmax", type=float, default=0.5, metavar="Q",
        help="Maximum q in Å⁻¹ (default: 0.5)",
    )
    parser.add_argument(
        "--n-q", dest="n_q", type=int, default=200, metavar="N",
        help="Number of q points (default: 200)",
    )
    parser.add_argument(
        "--k", type=float, default=None, metavar="K",
        help=(
            "Noise scale: σ(q) = I(q) / k, I_noisy = I + N(0, σ). "
            "Omit for noiseless data with a small dummy σ."
        ),
    )
    parser.add_argument(
        "--log-spacing", action="store_true",
        help="Use log-spaced q values (default: linear)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed for reproducible noise (default: 42)",
    )
    parser.add_argument(
        "--output", default="-", metavar="FILE",
        help="Output .dat file; '-' for stdout (default: -)",
    )
    parser.add_argument(
        "--pr-reference", default=None, metavar="FILE",
        help="Write numerical reference P(r) to this file for comparison plots",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.log_spacing:
        q = np.logspace(np.log10(args.qmin), np.log10(args.qmax), args.n_q)
    else:
        q = np.linspace(args.qmin, args.qmax, args.n_q)

    I = debye_intensity(q, args.rg)

    if args.k is not None:
        sigma = I / args.k
        I = I + rng.normal(loc=0.0, scale=sigma)
    else:
        sigma = np.full_like(I, 0.001 * I.max())

    # Write I(q) data
    noise_desc = f"k = {args.k}" if args.k is not None else "noiseless"
    out = sys.stdout if args.output == "-" else open(args.output, "w")
    try:
        print(
            f"# Synthetic SAXS: Debye/Gaussian chain Rg = {args.rg} Å, "
            f"{noise_desc}, seed = {args.seed}",
            file=out,
        )
        print(f"# {'q(1/A)':>14}  {'I(q)':>14}  {'sigma(q)':>14}", file=out)
        for qi, Ii, si in zip(q, I, sigma):
            print(f"  {qi:>14.8e}  {Ii:>14.8e}  {si:>14.8e}", file=out)
    finally:
        if out is not sys.stdout:
            out.close()

    # Optionally write numerical reference P(r)
    if args.pr_reference is not None:
        # Use noiseless I for the reference transform
        I_clean = debye_intensity(q, args.rg)
        r_max_ref = 6.0 * args.rg     # P(r) is negligible beyond ~6 Rg
        r_ref, pr_ref = compute_pr_reference(q, I_clean, r_max_ref)

        # Normalise to peak = 1 for easy visual comparison
        peak = pr_ref.max()
        if peak > 0:
            pr_ref /= peak

        with open(args.pr_reference, "w") as f:
            print(
                f"# Numerical reference P(r): Debye chain Rg = {args.rg} Å "
                f"(normalised to peak = 1)",
                file=f,
            )
            print(f"# {'r(A)':>14}  {'P(r)':>14}", file=f)
            for ri, pri in zip(r_ref, pr_ref):
                print(f"  {ri:>14.6e}  {pri:>14.6e}", file=f)

        print(f"Reference P(r) written to '{args.pr_reference}'", file=sys.stderr)


if __name__ == "__main__":
    main()
