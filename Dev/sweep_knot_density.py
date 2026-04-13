#!/usr/bin/env python3
"""
sweep_knot_density.py — exploratory Epic 5 validation for spline resolution.

Runs unFourier on one input dataset with:

  n_basis:       12, 16, 20, 28, 36
  knot_spacing:  5.0, 7.5, 10.0

The script writes output into Dev/_knot_density_tmp by default, prints a compact
diagnostic table, and saves an overlay plot for quick visual inspection.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unfourier_matplotlib")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "unfourier"
DEFAULT_INPUT = SCRIPT_DIR / "debye_k5.dat"
DEFAULT_OUTDIR = SCRIPT_DIR / "_knot_density_tmp"


@dataclass
class RunResult:
    label: str
    mode: str
    value: float
    n_basis: int
    elapsed_s: float
    chi2_red: float
    p0: float
    p_end: float
    roughness: float
    pr_path: Path


def read_pr(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def read_fit(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def reduced_chi_squared(fit: np.ndarray) -> float:
    i_obs = fit[:, 1]
    i_calc = fit[:, 2]
    sigma = np.where(fit[:, 3] > 0.0, fit[:, 3], 1.0)
    return float(np.mean(((i_obs - i_calc) / sigma) ** 2))


def curve_roughness(pr: np.ndarray) -> float:
    r = pr[:, 0]
    p = pr[:, 1]
    if len(r) < 3:
        return float("nan")
    dr = np.diff(r)
    d1 = np.diff(p) / dr
    mid_dr = 0.5 * (dr[1:] + dr[:-1])
    d2 = np.diff(d1) / mid_dr
    return float(np.sqrt(np.mean(d2 ** 2)))


def derived_n_basis(rmax: float, spacing: float, min_basis: int, max_basis: int) -> int:
    return max(min_basis, min(max_basis, math.ceil(rmax / spacing)))


def run_unfourier(
    binary: Path,
    data: Path,
    outdir: Path,
    rmax: float,
    method: str,
    label: str,
    args: list[str],
    n_basis: int,
) -> RunResult:
    pr_path = outdir / f"pr_{label}.dat"
    fit_path = outdir / f"fit_{label}.dat"
    cmd = [
        str(binary),
        str(data),
        "--method",
        method,
        "--rmax",
        str(rmax),
        "--output",
        str(pr_path),
        "--fit-output",
        str(fit_path),
        *args,
    ]

    start = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=outdir)
    elapsed = time.perf_counter() - start

    pr = read_pr(pr_path)
    fit = read_fit(fit_path)
    mode, value = label.split("_", 1)
    return RunResult(
        label=label,
        mode=mode,
        value=float(value.replace("p", ".")),
        n_basis=n_basis,
        elapsed_s=elapsed,
        chi2_red=reduced_chi_squared(fit),
        p0=float(pr[0, 1]),
        p_end=float(pr[-1, 1]),
        roughness=curve_roughness(pr),
        pr_path=pr_path,
    )


def plot_results(results: list[RunResult], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, mode, title in [
        (axes[0], "n", "Explicit n_basis"),
        (axes[1], "k", "Derived from knot_spacing"),
    ]:
        for result in results:
            if result.mode != mode:
                continue
            pr = read_pr(result.pr_path)
            ax.plot(pr[:, 0], pr[:, 1], label=f"{result.value:g} -> n={result.n_basis}")
        ax.set_title(title)
        ax.set_xlabel("r (A)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("P(r)")
    fig.suptitle("unFourier knot-density sweep")
    fig.tight_layout()
    fig.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unfourier", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--rmax", type=float, default=150.0)
    parser.add_argument("--method", choices=["gcv", "lcurve", "bayes"], default="gcv")
    parser.add_argument("--min-basis", type=int, default=12)
    parser.add_argument("--max-basis", type=int, default=48)
    parser.add_argument("--n-basis-values", type=int, nargs="+", default=[12, 16, 20, 28, 36])
    parser.add_argument("--knot-spacing-values", type=float, nargs="+", default=[5.0, 7.5, 10.0])
    args = parser.parse_args()

    if not args.unfourier.exists():
        raise SystemExit(f"unFourier binary not found: {args.unfourier}")
    if not args.input.exists():
        raise SystemExit(f"input data not found: {args.input}")
    if args.min_basis > args.max_basis:
        raise SystemExit("--min-basis must be <= --max-basis")

    args.outdir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for n_basis in args.n_basis_values:
        results.append(
            run_unfourier(
                args.unfourier,
                args.input,
                args.outdir,
                args.rmax,
                args.method,
                f"n_{n_basis}",
                ["--n-basis", str(n_basis)],
                n_basis,
            )
        )

    for spacing in args.knot_spacing_values:
        n_basis = derived_n_basis(args.rmax, spacing, args.min_basis, args.max_basis)
        label = f"k_{str(spacing).replace('.', 'p')}"
        results.append(
            run_unfourier(
                args.unfourier,
                args.input,
                args.outdir,
                args.rmax,
                args.method,
                label,
                [
                    "--knot-spacing",
                    str(spacing),
                    "--min-basis",
                    str(args.min_basis),
                    "--max-basis",
                    str(args.max_basis),
                ],
                n_basis,
            )
        )

    print("mode  value   n_basis  chi2_red    roughness      P(0)       P(Dmax)   time_s")
    print("----  ------  -------  --------  -----------  ----------  ----------  ------")
    for result in results:
        print(
            f"{result.mode:>4}  {result.value:>6g}  {result.n_basis:>7d}  "
            f"{result.chi2_red:>8.4f}  {result.roughness:>11.4e}  "
            f"{result.p0:>10.3e}  {result.p_end:>10.3e}  {result.elapsed_s:>6.2f}"
        )

    plot_path = args.outdir / "knot_density_sweep.png"
    plot_results(results, plot_path)
    print(f"\nplot written to {plot_path}")


if __name__ == "__main__":
    main()
