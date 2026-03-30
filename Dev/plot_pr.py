#!/usr/bin/env python3
"""
Plot P(r) output from unFourier, optionally with the fit to I(q) data.

When --fit is supplied, a two-panel figure is shown:
  Top panel:    I(q) — measured data with error bars, overlaid with I_calc
  Bottom panel: P(r) — computed pair-distance distribution

Without --fit, only the P(r) panel is shown.

Usage examples
--------------
# P(r) only:
    python plot_pr.py pr.dat

# Two-panel: P(r) + fit to data (requires --fit-output from unfourier):
    python plot_pr.py pr.dat --fit fit.dat

# Overlay analytic P(r) for a sphere of R = 50 Å:
    python plot_pr.py pr.dat --fit fit.dat --sphere 50

# Save to file:
    python plot_pr.py pr.dat --fit fit.dat --sphere 50 --save output.png

# Compare multiple P(r) runs (no fit panel in multi-file mode):
    python plot_pr.py pr_m1.dat pr_m2.dat --sphere 50 --labels "M1 (no reg)" "M2 (Tikhonov)"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    print("matplotlib is required:  pip install matplotlib", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pr(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a P(r) file produced by unfourier.

    Returns (r, p_r, p_r_err) where p_r_err is None if no error column.
    """
    r, pr, err = [], [], []
    has_err = False

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            try:
                if len(cols) >= 3:
                    r.append(float(cols[0]))
                    pr.append(float(cols[1]))
                    err.append(float(cols[2]))
                    has_err = True
                elif len(cols) == 2:
                    r.append(float(cols[0]))
                    pr.append(float(cols[1]))
            except ValueError:
                pass

    return np.array(r), np.array(pr), (np.array(err) if has_err else None)


def load_fit(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a fit file produced by unfourier --fit-output.

    Returns (q, I_obs, I_calc, sigma).
    """
    q, i_obs, i_calc, sigma = [], [], [], []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            try:
                if len(cols) >= 4:
                    q.append(float(cols[0]))
                    i_obs.append(float(cols[1]))
                    i_calc.append(float(cols[2]))
                    sigma.append(float(cols[3]))
            except ValueError:
                pass

    return np.array(q), np.array(i_obs), np.array(i_calc), np.array(sigma)


# ---------------------------------------------------------------------------
# Analytic reference
# ---------------------------------------------------------------------------

def sphere_pr(r: np.ndarray, R: float) -> np.ndarray:
    """
    Analytic P(r) for a solid sphere of radius R.

    P(r) ∝ r^2 * (1 - 3r/(4R) + r^3/(16R^3))  for 0 ≤ r ≤ 2R

    Normalised so that max = 1.
    """
    pr = np.zeros_like(r, dtype=float)
    mask = (r >= 0) & (r <= 2 * R)
    rr = r[mask]
    pr[mask] = rr**2 * (1.0 - (3.0 * rr) / (4.0 * R) + rr**3 / (16.0 * R**3))
    peak = pr.max()
    if peak > 0:
        pr /= peak
    return pr


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

PALETTE = list(mcolors.TABLEAU_COLORS.values())


def plot_fit_panel(ax, q, i_obs, i_calc, sigma, label=""):
    """Plot measured I(q) with error bars and back-calculated I_calc."""
    ax.errorbar(
        q, i_obs, yerr=sigma,
        fmt="o", ms=3, color="steelblue", ecolor="lightsteelblue",
        elinewidth=1, capsize=0, label="I(q) measured",
    )
    ax.plot(q, i_calc, color="tomato", lw=1.5, label="I(q) back-calc")
    ax.set_yscale("log")
    ax.set_xlabel("q  (Å⁻¹)", fontsize=11)
    ax.set_ylabel("I(q)", fontsize=11)
    ax.legend(fontsize=9)

    # Annotate with chi-squared if we can compute it
    mask = sigma > 0
    if mask.sum() > 0:
        chi2 = np.mean(((i_obs[mask] - i_calc[mask]) / sigma[mask]) ** 2)
        ax.set_title(f"Fit to data  (χ²_red = {chi2:.3f})", fontsize=11)
    else:
        ax.set_title("Fit to data", fontsize=11)


def plot_pr_panel(ax, runs, sphere_R=None, normalise=True):
    """
    Plot one or more P(r) curves.

    runs: list of (r, pr, err, label) tuples
    """
    for i, (r, pr, err, label) in enumerate(runs):
        color = PALETTE[i % len(PALETTE)]

        if normalise:
            scale = np.max(np.abs(pr)) or 1.0
            pr = pr / scale
            if err is not None:
                err = err / scale

        if err is not None:
            ax.fill_between(r, pr - err, pr + err, alpha=0.25, color=color)

        ax.plot(r, pr, color=color, lw=2, label=label)

    if sphere_R is not None:
        r_dense = np.linspace(0, 2 * sphere_R * 1.05, 600)
        pr_ana = sphere_pr(r_dense, sphere_R)
        ax.plot(
            r_dense, pr_ana,
            color="black", lw=1.5, ls="--",
            label=f"Analytic sphere R = {sphere_R} Å",
        )

    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ylabel = "P(r)  [normalised]" if normalise else "P(r)"
    ax.set_xlabel("r  (Å)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Pair-distance distribution P(r)", fontsize=11)
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="FILE",
        help="One or more P(r) files produced by unfourier",
    )
    parser.add_argument(
        "--fit", metavar="FILE",
        help="Fit file from unfourier --fit-output (adds I(q) panel above P(r))",
    )
    parser.add_argument(
        "--sphere", type=float, metavar="R",
        help="Overlay analytic P(r) for a solid sphere of radius R Å",
    )
    parser.add_argument(
        "--labels", nargs="+", metavar="LABEL",
        help="Labels for each input file (must match number of files)",
    )
    parser.add_argument(
        "--title", default="", metavar="TEXT",
        help="Overall figure title",
    )
    parser.add_argument(
        "--save", metavar="FILE",
        help="Save plot to FILE (PNG, PDF, SVG, …) instead of showing interactively",
    )
    parser.add_argument(
        "--no-normalise", action="store_true",
        help="Plot raw P(r) values instead of normalising peak to 1",
    )
    args = parser.parse_args()

    labels = args.labels or [Path(p).stem for p in args.inputs]
    if len(labels) != len(args.inputs):
        parser.error("--labels must have the same number of entries as input files")

    # Load P(r) files
    runs = []
    for path, label in zip(args.inputs, labels):
        try:
            r, pr, err = load_pr(path)
        except FileNotFoundError:
            print(f"error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        if len(r) == 0:
            print(f"warning: no data found in {path}", file=sys.stderr)
            continue
        runs.append((r, pr, err, label))

    # Layout: two rows if --fit supplied, one row otherwise
    has_fit = args.fit is not None
    if has_fit:
        fig, (ax_fit, ax_pr) = plt.subplots(
            2, 1, figsize=(8, 9),
            gridspec_kw={"height_ratios": [1, 1.2]},
        )
        try:
            q, i_obs, i_calc, sigma = load_fit(args.fit)
        except FileNotFoundError:
            print(f"error: fit file not found: {args.fit}", file=sys.stderr)
            sys.exit(1)
        plot_fit_panel(ax_fit, q, i_obs, i_calc, sigma)
    else:
        fig, ax_pr = plt.subplots(figsize=(8, 5))

    plot_pr_panel(ax_pr, runs, sphere_R=args.sphere, normalise=not args.no_normalise)

    if args.title:
        fig.suptitle(args.title, fontsize=13, y=1.01)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
