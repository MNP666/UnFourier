#!/usr/bin/env python3
"""
sweep_smoothness.py - exploratory Epic 4 validation for D1/D2 regularisation.

Runs unFourier on one input dataset while sweeping d1_smoothness. Each run gets
its own temporary unfourier.toml so the production CLI can stay small while the
regulariser remains configurable.

By default the script uses Dev/debye_k5.dat. If that file does not exist, it is
generated with Dev/gen_debye.py. Pass --input to use a real dataset when the
reference data tree is available.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
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
DEFAULT_OUTDIR = SCRIPT_DIR / "_smoothness_tmp"


@dataclass
class RunResult:
    label: str
    d1: float
    d2: float
    elapsed_s: float
    chi2_red: float
    p0: float
    p_end: float
    left_slope: float
    right_slope: float
    roughness: float
    pr_path: Path


def ensure_debye_input(path: Path) -> None:
    if path.exists():
        return

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "gen_debye.py"),
        "--rg",
        "30",
        "--k",
        "5",
        "--output",
        str(path),
        "--pr-reference",
        str(path.with_name("debye_pr_ref.dat")),
    ]
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)


def read_table(path: Path) -> np.ndarray:
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
    return float(np.sqrt(np.mean(d2**2)))


def endpoint_slope(pr: np.ndarray) -> tuple[float, float]:
    r = pr[:, 0]
    p = pr[:, 1]
    left = (p[1] - p[0]) / (r[1] - r[0])
    right = (p[-1] - p[-2]) / (r[-1] - r[-2])
    return float(left), float(right)


def d1_label(d1: float) -> str:
    if d1 == -1.0:
        return "off"
    if d1 == 0.0:
        return "default"
    return str(d1).replace(".", "p")


def write_config(path: Path, boundary: str, d1: float, d2: float) -> None:
    path.write_text(
        "\n".join(
            [
                "[constraints]",
                f'spline_boundary = "{boundary}"',
                f"d1_smoothness = {d1}",
                f"d2_smoothness = {d2}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_unfourier(
    binary: Path,
    data: Path,
    outdir: Path,
    rmax: float,
    method: str,
    n_basis: int,
    boundary: str,
    d1: float,
    d2: float,
) -> RunResult:
    label = f"d1_{d1_label(d1)}"
    run_dir = outdir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(run_dir / "unfourier.toml", boundary, d1, d2)

    pr_path = run_dir / "pr.dat"
    fit_path = run_dir / "fit.dat"
    cmd = [
        str(binary),
        str(data),
        "--method",
        method,
        "--rmax",
        str(rmax),
        "--n-basis",
        str(n_basis),
        "--output",
        str(pr_path),
        "--fit-output",
        str(fit_path),
    ]

    start = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=run_dir)
    elapsed = time.perf_counter() - start

    pr = read_table(pr_path)
    fit = read_table(fit_path)
    left_slope, right_slope = endpoint_slope(pr)

    return RunResult(
        label=label,
        d1=d1,
        d2=d2,
        elapsed_s=elapsed,
        chi2_red=reduced_chi_squared(fit),
        p0=float(pr[0, 1]),
        p_end=float(pr[-1, 1]),
        left_slope=left_slope,
        right_slope=right_slope,
        roughness=curve_roughness(pr),
        pr_path=pr_path,
    )


def plot_results(results: list[RunResult], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for result in results:
        pr = read_table(result.pr_path)
        p = pr[:, 1]
        peak = p.max()
        y = p / peak if peak > 0 else p
        axes[0].plot(pr[:, 0], y, label=result.label)

    axes[0].set_xlabel("r (A)")
    axes[0].set_ylabel("P(r), peak normalised")
    axes[0].set_title("P(r) overlay")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    x = np.arange(len(results))
    axes[1].plot(x, [r.roughness for r in results], marker="o", label="roughness")
    axes[1].plot(x, [abs(r.left_slope) for r in results], marker="s", label="abs left slope")
    axes[1].plot(x, [abs(r.right_slope) for r in results], marker="^", label="abs right slope")
    axes[1].set_xticks(x, [r.label for r in results], rotation=30, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_title("Endpoint and roughness diagnostics")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unfourier", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--rmax", type=float, default=150.0)
    parser.add_argument("--method", choices=["gcv", "lcurve", "bayes"], default="gcv")
    parser.add_argument("--n-basis", type=int, default=20)
    parser.add_argument(
        "--boundary",
        choices=["value_zero", "value_slope_zero"],
        default="value_zero",
    )
    parser.add_argument(
        "--d1-values",
        type=float,
        nargs="+",
        default=[-1.0, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--d2", type=float, default=1.0)
    args = parser.parse_args()

    if not args.unfourier.exists():
        raise SystemExit(f"unFourier binary not found: {args.unfourier}")
    if args.input == DEFAULT_INPUT:
        ensure_debye_input(args.input)
    if not args.input.exists():
        raise SystemExit(f"input data not found: {args.input}")
    if args.d2 < 0.0 or not math.isfinite(args.d2):
        raise SystemExit("--d2 must be non-negative and finite")

    args.outdir.mkdir(parents=True, exist_ok=True)

    results = [
        run_unfourier(
            args.unfourier,
            args.input.resolve(),
            args.outdir.resolve(),
            args.rmax,
            args.method,
            args.n_basis,
            args.boundary,
            d1,
            args.d2,
        )
        for d1 in args.d1_values
    ]

    print("d1          d2     chi2_red    roughness    left_slope  right_slope      P(0)    P(Dmax)   time_s")
    print("----------  -----  --------  -----------  ------------  -----------  --------  --------  ------")
    for result in results:
        print(
            f"{result.label:>10}  {result.d2:>5.2f}  {result.chi2_red:>8.4f}  "
            f"{result.roughness:>11.4e}  {result.left_slope:>12.4e}  "
            f"{result.right_slope:>11.4e}  {result.p0:>8.1e}  "
            f"{result.p_end:>8.1e}  {result.elapsed_s:>6.2f}"
        )

    plot_path = args.outdir / "smoothness_sweep.png"
    plot_results(results, plot_path)
    print(f"\nplot written to {plot_path}")


if __name__ == "__main__":
    main()
