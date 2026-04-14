#!/usr/bin/env python3
"""
sweep_noise.py - spline-only noise and parameter sweep for unFourier 0.9.

The script generates Debye-chain data at several noise levels, runs the active
cubic B-spline CLI with GCV and L-curve selection, and compares the recovered
P(r) to the numerical Debye reference.

It also performs two compact parameter sweeps at one representative noise level:

  * n_basis = 12, 16, 20, 28, 36
  * d1_smoothness = -1, 0, 0.05, 0.1, 0.25

No legacy basis-selection flags or histogram-grid plot labels are used.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unfourier_matplotlib")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "unfourier"
DEFAULT_OUTDIR = SCRIPT_DIR / "_sweep_tmp"


@dataclass
class RunResult:
    label: str
    k: float
    method: str
    n_basis: int
    d1: float
    lambda_selected: float
    chi2_red: float
    ise: float
    pr_path: Path
    fit_path: Path


def generate_debye(rg: float, k: float | None, outfile: Path, pr_ref: Path | None = None) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "gen_debye.py"),
        "--rg",
        str(rg),
        "--output",
        str(outfile),
        "--seed",
        "42",
    ]
    if k is not None:
        cmd += ["--k", str(k)]
    if pr_ref is not None:
        cmd += ["--pr-reference", str(pr_ref)]
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)


def write_config(path: Path, d1: float, d2: float, boundary: str) -> None:
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


def d1_label(d1: float) -> str:
    if d1 == -1.0:
        return "off"
    if d1 == 0.0:
        return "default"
    return str(d1).replace(".", "p")


def load_two_col(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1]


def integrated_squared_error(
    r_calc: np.ndarray,
    pr_calc: np.ndarray,
    r_ref: np.ndarray,
    pr_ref: np.ndarray,
) -> float:
    r_lo = max(float(r_calc[0]), float(r_ref[0]))
    r_hi = min(float(r_calc[-1]), float(r_ref[-1]))
    mask = (r_calc >= r_lo) & (r_calc <= r_hi)
    r_common = r_calc[mask]
    if r_common.size < 2:
        return float("nan")

    pr_ref_interp = np.interp(r_common, r_ref, pr_ref, left=0.0, right=0.0)
    pr_calc_interp = pr_calc[mask]

    peak_ref = pr_ref_interp.max()
    peak_calc = pr_calc_interp.max()
    if peak_ref < 1e-12 or peak_calc < 1e-12:
        return float("nan")

    pr_ref_n = pr_ref_interp / peak_ref
    pr_calc_n = pr_calc_interp / peak_calc
    ise_num = np.trapezoid((pr_calc_n - pr_ref_n) ** 2, r_common)
    ise_denom = np.trapezoid(pr_ref_n**2, r_common)
    return float(ise_num / ise_denom) if ise_denom > 0 else float("nan")


def parse_verbose(stderr: str) -> tuple[float, float]:
    lam = float("nan")
    chi = float("nan")
    for line in stderr.splitlines():
        if "selected lambda" in line.replace("λ", "lambda"):
            try:
                lam = float(line.split("=")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        if "χ²_red" in line:
            try:
                chi = float(line.split("=")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
    return lam, chi


def run_unfourier(
    binary: Path,
    dat_file: Path,
    run_dir: Path,
    label: str,
    rmax: float,
    method: str,
    n_basis: int,
    d1: float,
    d2: float,
    boundary: str,
    r_ref: np.ndarray,
    pr_ref: np.ndarray,
    k: float,
) -> RunResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(run_dir / "unfourier.toml", d1, d2, boundary)

    pr_out = run_dir / "pr.dat"
    fit_out = run_dir / "fit.dat"
    cmd = [
        str(binary),
        str(dat_file),
        "--rmax",
        str(rmax),
        "--method",
        method,
        "--n-basis",
        str(n_basis),
        "--output",
        str(pr_out),
        "--fit-output",
        str(fit_out),
        "--verbose",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=run_dir)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stderr}")

    r_calc, pr_calc = load_two_col(pr_out)
    ise = integrated_squared_error(r_calc, pr_calc, r_ref, pr_ref)
    lam, chi = parse_verbose(result.stderr)
    return RunResult(
        label=label,
        k=k,
        method=method,
        n_basis=n_basis,
        d1=d1,
        lambda_selected=lam,
        chi2_red=chi,
        ise=ise,
        pr_path=pr_out,
        fit_path=fit_out,
    )


def print_results(title: str, results: list[RunResult]) -> None:
    print(f"\n{title}")
    print("label                 k  method   n_basis      d1      lambda  chi2_red        ISE")
    print("----------------  -----  -------  -------  ------  ----------  --------  ---------")
    for r in results:
        print(
            f"{r.label:<16}  {r.k:>5.1f}  {r.method:<7}  {r.n_basis:>7d}  "
            f"{d1_label(r.d1):>6}  {r.lambda_selected:>10.3e}  "
            f"{r.chi2_red:>8.4f}  {r.ise:>9.4e}"
        )


def plot_results(
    noise_results: list[RunResult],
    n_basis_results: list[RunResult],
    d1_results: list[RunResult],
    r_ref: np.ndarray,
    pr_ref: np.ndarray,
    out_fig: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("unFourier spline noise and smoothness sweep", fontsize=13)

    ax = axes[0, 0]
    ax.plot(r_ref, pr_ref / pr_ref.max(), "k--", lw=2, label="Debye ref")
    for result in noise_results:
        if result.method != "gcv":
            continue
        r, p = load_two_col(result.pr_path)
        ax.plot(r, p / max(p.max(), 1e-15), lw=1.4, label=f"k={result.k:g}")
    ax.set_title("GCV P(r) across noise levels")
    ax.set_xlabel("r (A)")
    ax.set_ylabel("P(r), peak normalised")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for method, marker in [("gcv", "o"), ("lcurve", "s")]:
        subset = [r for r in noise_results if r.method == method]
        ax.semilogy([r.k for r in subset], [r.ise for r in subset], marker + "-", label=method)
    ax.set_title("ISE vs noise")
    ax.set_xlabel("k (higher = less noise)")
    ax.set_ylabel("ISE")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot([r.n_basis for r in n_basis_results], [r.ise for r in n_basis_results], "o-")
    ax.set_title("n_basis sweep")
    ax.set_xlabel("n_basis")
    ax.set_ylabel("ISE")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(range(len(d1_results)), [r.ise for r in d1_results], "o-")
    ax.set_xticks(range(len(d1_results)), [d1_label(r.d1) for r in d1_results], rotation=25)
    ax.set_title("d1_smoothness sweep")
    ax.set_xlabel("d1")
    ax.set_ylabel("ISE")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rg", type=float, default=30.0)
    parser.add_argument("--rmax", type=float, default=180.0)
    parser.add_argument("--unfourier", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[3, 5, 10, 20])
    parser.add_argument("--methods", nargs="+", choices=["gcv", "lcurve"], default=["gcv", "lcurve"])
    parser.add_argument("--n-basis", type=int, default=20)
    parser.add_argument("--n-basis-values", nargs="+", type=int, default=[12, 16, 20, 28, 36])
    parser.add_argument("--d1-values", nargs="+", type=float, default=[-1.0, 0.0, 0.05, 0.1, 0.25])
    parser.add_argument("--d2", type=float, default=1.0)
    parser.add_argument("--parameter-k", type=float, default=5.0)
    parser.add_argument("--boundary", choices=["value_zero", "value_slope_zero"], default="value_zero")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if not args.unfourier.exists():
        raise SystemExit(f"unfourier binary not found at {args.unfourier}. Run: cargo build --release")

    args.outdir.mkdir(parents=True, exist_ok=True)

    ref_file = args.outdir / "debye_pr_ref.dat"
    noiseless = args.outdir / "debye_noiseless.dat"
    if not ref_file.exists():
        generate_debye(args.rg, None, noiseless, pr_ref=ref_file)
    r_ref, pr_ref = load_two_col(ref_file)

    data_by_k: dict[float, Path] = {}
    for k in sorted(set([*args.noise_levels, args.parameter_k])):
        dat = args.outdir / f"debye_k{str(k).replace('.', 'p')}.dat"
        if not dat.exists():
            generate_debye(args.rg, k, dat)
        data_by_k[k] = dat

    noise_results: list[RunResult] = []
    for k in args.noise_levels:
        for method in args.methods:
            label = f"noise_k{str(k).replace('.', 'p')}_{method}"
            noise_results.append(
                run_unfourier(
                    args.unfourier,
                    data_by_k[k],
                    args.outdir / label,
                    label,
                    args.rmax,
                    method,
                    args.n_basis,
                    0.0,
                    args.d2,
                    args.boundary,
                    r_ref,
                    pr_ref,
                    k,
                )
            )

    n_basis_results = [
        run_unfourier(
            args.unfourier,
            data_by_k[args.parameter_k],
            args.outdir / f"n_basis_{n}",
            f"n_basis_{n}",
            args.rmax,
            "gcv",
            n,
            0.0,
            args.d2,
            args.boundary,
            r_ref,
            pr_ref,
            args.parameter_k,
        )
        for n in args.n_basis_values
    ]

    d1_results = [
        run_unfourier(
            args.unfourier,
            data_by_k[args.parameter_k],
            args.outdir / f"d1_{d1_label(d1)}",
            f"d1_{d1_label(d1)}",
            args.rmax,
            "gcv",
            args.n_basis,
            d1,
            args.d2,
            args.boundary,
            r_ref,
            pr_ref,
            args.parameter_k,
        )
        for d1 in args.d1_values
    ]

    print_results("Noise sweep", noise_results)
    print_results("n_basis sweep", n_basis_results)
    print_results("d1_smoothness sweep", d1_results)

    if not args.no_plot:
        out_fig = args.outdir / "sweep_noise.png"
        plot_results(noise_results, n_basis_results, d1_results, r_ref, pr_ref, out_fig)
        print(f"\nPlot saved to {out_fig}")


if __name__ == "__main__":
    main()
