#!/usr/bin/env python3
"""
Generate synthetic SAXS data for a solid sphere.

The analytic scattering intensity for a homogeneous sphere of radius R is:

    I(q) = [3 (sin(qR) - qR cos(qR)) / (qR)^3]^2

normalised to I(0) = 1.

The analytic pair-distance distribution function (for comparison in plot_pr.py) is:

    P(r) ∝ r^2 * (1 - (3r)/(4R) + r^3/(16R^3))  for 0 ≤ r ≤ 2R

This is the standard Glatter/Kratky result for a homogeneous sphere.

Usage examples
--------------
# Noiseless, log-spaced q (good for visual comparison):
    python gen_sphere.py --radius 50 --output sphere_noiseless.dat

# 5% relative noise:
    python gen_sphere.py --radius 50 --noise 0.05 --output sphere_noisy.dat

# Smaller sphere, more q points, saved to stdout:
    python gen_sphere.py --radius 30 --npoints 300 --qmin 0.005 --qmax 0.8
"""

import argparse
import sys

import numpy as np


def sphere_intensity(q: np.ndarray, R: float) -> np.ndarray:
    """Analytic I(q) for a solid sphere of radius R, normalised to I(0) = 1."""
    qR = q * R
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(
            np.abs(qR) < 1e-10,
            1.0,
            3.0 * (np.sin(qR) - qR * np.cos(qR)) / qR**3,
        )
    return f**2


def sphere_pr(r: np.ndarray, R: float) -> np.ndarray:
    """
    Analytic P(r) for a solid sphere of radius R.

    Formula: P(r) ∝ r^2 * (1 - (3r)/(4R) + r^3/(16R^3))  for 0 ≤ r ≤ 2R

    Normalised so that the maximum value is 1 (for plotting comparison).
    """
    pr = np.zeros_like(r, dtype=float)
    mask = (r >= 0) & (r <= 2 * R)
    rr = r[mask]
    pr[mask] = rr**2 * (1.0 - (3.0 * rr) / (4.0 * R) + rr**3 / (16.0 * R**3))
    if pr.max() > 0:
        pr /= pr.max()
    return pr


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--radius", type=float, default=50.0, metavar="R",
        help="Sphere radius in Å (default: 50)",
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
        "--npoints", type=int, default=200, metavar="N",
        help="Number of q points (default: 200)",
    )
    parser.add_argument(
        "--noise", type=float, default=0.0, metavar="FRAC",
        help="Relative noise level, e.g. 0.05 = 5%% (default: 0 = noiseless)",
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
        help="Output file path; use '-' for stdout (default: -)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # q grid
    if args.log_spacing:
        q = np.logspace(np.log10(args.qmin), np.log10(args.qmax), args.npoints)
    else:
        q = np.linspace(args.qmin, args.qmax, args.npoints)

    # Analytic intensity, normalised to I(0) = 1
    intensity = sphere_intensity(q, args.radius)

    # Error model: constant relative noise floor
    # σ_i = noise_level × I_i  (with a small absolute floor to avoid zeros)
    if args.noise > 0.0:
        sigma = args.noise * intensity
        sigma = np.maximum(sigma, 1e-6 * intensity.max())
        intensity = intensity + rng.normal(0.0, sigma)
    else:
        # Noiseless: assign a small dummy σ so the parser always gets 3 columns
        sigma = np.full_like(intensity, 0.001 * intensity.max())

    # Write output
    out = sys.stdout if args.output == "-" else open(args.output, "w")
    try:
        print(
            f"# Synthetic SAXS: solid sphere R = {args.radius} Å, "
            f"noise = {args.noise:.3f}, seed = {args.seed}",
            file=out,
        )
        print(f"# {'q(1/A)':>14}  {'I(q)':>14}  {'sigma(q)':>14}", file=out)
        for qi, Ii, si in zip(q, intensity, sigma):
            print(f"  {qi:>14.8e}  {Ii:>14.8e}  {si:>14.8e}", file=out)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
