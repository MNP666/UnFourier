#!/usr/bin/env python3
"""
Plot P(r) output from unFourier, optionally overlaid with the analytic result
for a solid sphere fixture.

Usage examples
--------------
# Basic plot:
    python plot_pr.py pr.dat

# Overlay analytic sphere P(r) for R = 50 Å:
    python plot_pr.py pr.dat --sphere 50

# Save to file instead of showing interactively:
    python plot_pr.py pr.dat --sphere 50 --save pr_comparison.png

# Compare two runs side-by-side (e.g. before and after regularisation):
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
# Helpers
# ---------------------------------------------------------------------------

def load_pr(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a P(r) file produced by unFourier.

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
                pass  # skip header-like rows

    r_arr = np.array(r)
    pr_arr = np.array(pr)
    err_arr = np.array(err) if has_err else None
    return r_arr, pr_arr, err_arr


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


def normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise an array so that max |value| = 1."""
    scale = np.max(np.abs(arr))
    return arr / scale if scale > 0 else arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PALETTE = list(mcolors.TABLEAU_COLORS.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="FILE",
        help="One or more P(r) files produced by unFourier",
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
        help="Plot title",
    )
    parser.add_argument(
        "--save", metavar="FILE",
        help="Save plot to FILE (PNG, PDF, SVG, …) instead of showing interactively",
    )
    parser.add_argument(
        "--no-normalise", action="store_true",
        help="Plot raw P(r) values instead of normalising to peak = 1",
    )
    args = parser.parse_args()

    labels = args.labels or [Path(p).stem for p in args.inputs]
    if len(labels) != len(args.inputs):
        parser.error("--labels must have the same number of entries as input files")

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (path, label) in enumerate(zip(args.inputs, labels)):
        color = PALETTE[i % len(PALETTE)]
        try:
            r, pr, err = load_pr(path)
        except FileNotFoundError:
            print(f"error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

        if len(r) == 0:
            print(f"warning: no data found in {path}", file=sys.stderr)
            continue

        if not args.no_normalise:
            scale = np.max(np.abs(pr)) or 1.0
            pr = pr / scale
            if err is not None:
                err = err / scale

        if err is not None:
            ax.fill_between(r, pr - err, pr + err, alpha=0.25, color=color)

        ax.plot(r, pr, color=color, lw=2, label=label)

    # Analytic overlay
    if args.sphere is not None:
        r_dense = np.linspace(0, 2 * args.sphere * 1.05, 600)
        pr_ana = sphere_pr(r_dense, args.sphere)
        ax.plot(
            r_dense, pr_ana,
            color="black", lw=1.5, ls="--",
            label=f"Analytic sphere R = {args.sphere} Å",
        )

    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("r  (Å)", fontsize=12)
    ax.set_ylabel("P(r)  [normalised]" if not args.no_normalise else "P(r)", fontsize=12)
    ax.set_title(args.title or "Pair-distance distribution function P(r)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
