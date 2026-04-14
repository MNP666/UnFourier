#!/usr/bin/env python3
"""
validate_real_data.py - spline-only validation against GNOM reference outputs.

For each real SAXS dataset with a matching GNOM .out reference, this script:

  1. Parses Dmax and Rg from the GNOM output.
  2. Runs unFourier with the active cubic B-spline CLI.
  3. Optionally runs rebin variants for large datasets.
  4. Reports ISE, Rg agreement, endpoint values, endpoint slopes, and runtime.
  5. Saves a comparison plot with endpoint diagnostics.

The script intentionally relies only on the current spline CLI surface.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unfourier_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make parse_gnom importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).parent))
from parse_gnom import parse_gnom_out


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "unfourier"
DEFAULT_DAT_DIR = REPO_ROOT / "data" / "dat_ref"
DEFAULT_REF_DIR = REPO_ROOT / "data" / "prs_ref"
DEFAULT_PLOT = SCRIPT_DIR / "validation_plot.png"
DATASETS = ["SASDME2", "SASDF42", "SASDYU3"]


@dataclass
class RunResult:
    dataset: str
    variant: str
    rebin: int
    elapsed_s: float
    ise: float
    rg_unfourier: float
    rg_ref: float
    rg_rel_err: float
    p0: float
    p_end: float
    left_slope: float
    right_slope: float
    pr: np.ndarray


def run_unfourier(
    binary: Path,
    dat_file: Path,
    rmax: float,
    outdir: Path,
    dataset: str,
    variant: str,
    method: str,
    n_basis: int,
    rebin: int,
) -> tuple[np.ndarray, float]:
    pr_path = outdir / f"{dataset}_{variant}_pr.dat"
    cmd = [
        str(binary),
        str(dat_file),
        "--rmax",
        str(rmax),
        "--negative-handling",
        "clip",
        "--method",
        method,
        "--n-basis",
        str(n_basis),
        "--output",
        str(pr_path),
    ]
    if rebin > 0:
        cmd += ["--rebin", str(rebin)]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=outdir)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"unfourier failed for {dataset}/{variant} "
            f"(exit {result.returncode}):\n{result.stderr}"
        )

    pr = np.loadtxt(pr_path, comments="#")
    if pr.ndim != 2 or pr.shape[1] < 2:
        raise RuntimeError(f"unfourier produced malformed P(r) output: {pr_path}")
    return pr[:, :2], elapsed


def rg_from_pr(r: np.ndarray, pr: np.ndarray) -> float:
    denom = np.trapezoid(pr, r)
    if denom <= 0:
        return float("nan")
    numer = np.trapezoid(r**2 * pr, r)
    return float(np.sqrt(max(numer / (2.0 * denom), 0.0)))


def ise_normalised(
    r_ref: np.ndarray,
    pr_ref: np.ndarray,
    r_new: np.ndarray,
    pr_new: np.ndarray,
) -> float:
    peak_ref = pr_ref.max()
    peak_new = pr_new.max()
    if peak_ref <= 0 or peak_new <= 0:
        return float("nan")

    pr_ref_n = pr_ref / peak_ref
    pr_new_interp = np.interp(r_ref, r_new, pr_new / peak_new, left=0.0, right=0.0)
    return float(np.trapezoid((pr_ref_n - pr_new_interp) ** 2, r_ref))


def endpoint_slope(pr: np.ndarray) -> tuple[float, float]:
    r = pr[:, 0]
    p = pr[:, 1]
    if len(r) < 2:
        return float("nan"), float("nan")
    left = (p[1] - p[0]) / (r[1] - r[0])
    right = (p[-1] - p[-2]) / (r[-1] - r[-2])
    return float(left), float(right)


def make_result(
    dataset: str,
    variant: str,
    rebin: int,
    pr: np.ndarray,
    elapsed_s: float,
    r_ref: np.ndarray,
    pr_ref: np.ndarray,
    rg_ref: float,
) -> RunResult:
    r = pr[:, 0]
    p = pr[:, 1]
    rg_unfourier = rg_from_pr(r, p)
    ise = ise_normalised(r_ref, pr_ref, r, p)
    rg_rel_err = abs(rg_unfourier - rg_ref) / rg_ref if rg_ref > 0 else float("nan")
    left_slope, right_slope = endpoint_slope(pr)
    return RunResult(
        dataset=dataset,
        variant=variant,
        rebin=rebin,
        elapsed_s=elapsed_s,
        ise=ise,
        rg_unfourier=rg_unfourier,
        rg_ref=rg_ref,
        rg_rel_err=rg_rel_err,
        p0=float(p[0]),
        p_end=float(p[-1]),
        left_slope=left_slope,
        right_slope=right_slope,
        pr=pr,
    )


def peak_norm(p: np.ndarray) -> np.ndarray:
    peak = p.max()
    return p / peak if peak > 0 else p


def plot_results(results: list[RunResult], refs: dict[str, dict], out_path: Path) -> None:
    datasets = list(refs.keys())
    fig, axes = plt.subplots(2, len(datasets), figsize=(5.4 * len(datasets), 7.5))
    if len(datasets) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    fig.suptitle("unFourier spline validation vs GNOM reference", fontsize=13)

    for col, name in enumerate(datasets):
        ref = refs[name]
        matching = [r for r in results if r.dataset == name]

        ax = axes[0, col]
        ax.plot(ref["r"], peak_norm(ref["pr"]), "k-", lw=2, label="GNOM ref")
        for result in matching:
            r = result.pr[:, 0]
            p = result.pr[:, 1]
            label = result.variant if result.rebin == 0 else f"{result.variant}, rebin={result.rebin}"
            ax.plot(r, peak_norm(p), lw=1.5, label=label)
        ax.set_title(name)
        ax.set_xlabel("r (A)")
        ax.set_ylabel("P(r), peak normalised")
        ax.set_xlim(0, ref["d_max"] * 1.05)
        ax.set_ylim(-0.08, 1.25)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[1, col]
        labels = [r.variant if r.rebin == 0 else f"rebin={r.rebin}" for r in matching]
        x = np.arange(len(matching))
        ax.bar(x - 0.2, [abs(r.left_slope) for r in matching], width=0.2, label="abs left slope")
        ax.bar(x, [abs(r.right_slope) for r in matching], width=0.2, label="abs right slope")
        ax.bar(x + 0.2, [abs(r.p_end) for r in matching], width=0.2, label="abs P(Dmax)")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Endpoint diagnostic")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unfourier", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--dat-dir", type=Path, default=DEFAULT_DAT_DIR)
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "_real_validation_tmp")
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--method", choices=["gcv", "lcurve", "bayes"], default="gcv")
    parser.add_argument("--n-basis", type=int, default=20)
    parser.add_argument("--rebin-large", type=int, default=200)
    parser.add_argument("--ise-threshold", type=float, default=0.15)
    parser.add_argument("--rg-rel-threshold", type=float, default=0.15)
    args = parser.parse_args()

    if not args.unfourier.exists():
        raise SystemExit(f"binary not found: {args.unfourier}\nRun: cargo build --release")
    if not args.dat_dir.exists() or not args.ref_dir.exists():
        raise SystemExit(
            f"real-data directories not found:\n  dat: {args.dat_dir}\n  ref: {args.ref_dir}"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)

    refs: dict[str, dict] = {}
    results: list[RunResult] = []

    header = (
        f"{'Dataset':<10}  {'Variant':<12}  {'Rebin':>6}  {'ISE':>10}  "
        f"{'dRg/Rg':>9}  {'Rg':>8}  {'P(0)':>10}  {'P(Dmax)':>10}  "
        f"{'left dP':>10}  {'right dP':>10}  {'Time':>7}"
    )
    print(header)
    print("-" * len(header))

    had_missing = False
    for name in DATASETS:
        dat_file = args.dat_dir / f"{name}.dat"
        ref_file = args.ref_dir / f"{name}.out"
        if not dat_file.exists() or not ref_file.exists():
            print(f"{name:<10}  missing input/reference; skipping")
            had_missing = True
            continue

        ref = parse_gnom_out(ref_file)
        d_max = float(ref["d_max"])
        rg_ref = float(ref["rg"])
        pr_ref_arr = ref["pr"]
        r_ref = pr_ref_arr[:, 0]
        pr_ref = pr_ref_arr[:, 1]
        refs[name] = {"r": r_ref, "pr": pr_ref, "d_max": d_max}

        variants = [("spline", 0)]
        if name == "SASDYU3" and args.rebin_large > 0:
            variants.append(("spline-rebin", args.rebin_large))

        for variant, rebin in variants:
            pr, elapsed = run_unfourier(
                args.unfourier,
                dat_file,
                d_max,
                args.outdir,
                name,
                variant,
                args.method,
                args.n_basis,
                rebin,
            )
            result = make_result(name, variant, rebin, pr, elapsed, r_ref, pr_ref, rg_ref)
            results.append(result)

            ise_ok = result.ise < args.ise_threshold
            rg_ok = result.rg_rel_err < args.rg_rel_threshold
            rebin_str = str(rebin) if rebin > 0 else "-"
            print(
                f"{name:<10}  {variant:<12}  {rebin_str:>6}  "
                f"{result.ise:>7.4f} {'Y' if ise_ok else 'N':>2}  "
                f"{result.rg_rel_err:>6.4f} {'Y' if rg_ok else 'N':>2}  "
                f"{result.rg_unfourier:>8.3f}  {result.p0:>10.2e}  "
                f"{result.p_end:>10.2e}  {result.left_slope:>10.2e}  "
                f"{result.right_slope:>10.2e}  {result.elapsed_s:>7.2f}"
            )

    if refs and results:
        plot_results(results, refs, args.plot)
        print(f"\nValidation plot saved to: {args.plot}")

    primary = [r for r in results if r.variant == "spline" and r.rebin == 0]
    ise_pass = sum(r.ise < args.ise_threshold for r in primary)
    rg_all_pass = all(r.rg_rel_err < args.rg_rel_threshold for r in primary)
    required_ise = min(2, len(primary))
    overall_ok = (
        not had_missing
        and len(primary) == len(DATASETS)
        and ise_pass >= required_ise
        and rg_all_pass
    )

    print()
    print(
        f"Pass criteria: primary spline ISE < {args.ise_threshold} for "
        f">={required_ise}/{len(primary)} datasets; "
        f"primary dRg/Rg < {args.rg_rel_threshold} for all"
    )
    print(f"  ISE: {ise_pass}/{len(primary)} primary datasets pass")
    print(f"  Rg:  {'PASS' if rg_all_pass else 'FAIL'}")
    print(f"  Overall: {'PASS' if overall_ok else 'FAIL'}")

    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
