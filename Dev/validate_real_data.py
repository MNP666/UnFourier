#!/usr/bin/env python3
"""
validate_real_data.py - spline-only validation against GNOM reference outputs.

For each real SAXS dataset with a matching GNOM .out reference, this script:

  1. Parses Dmax and Rg from the GNOM output.
  2. Runs unFourier with the active cubic B-spline CLI.
  3. Optionally runs rebin variants for large datasets.
  4. Reports ISE, Rg agreement, chi2, endpoint values, endpoint slopes, and runtime.
  5. Saves a comparison plot with P(r), I(q) fit, and endpoint diagnostics.

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
DEFAULT_REBIN_THRESHOLD = 1000


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
    chi_squared: float
    pr: np.ndarray
    fit: np.ndarray


def discover_datasets(dat_dir: Path, ref_dir: Path) -> tuple[list[str], list[str], list[str]]:
    dat_names = {path.stem for path in dat_dir.glob("*.dat")}
    ref_names = {path.stem for path in ref_dir.glob("*.out")}
    datasets = sorted(dat_names & ref_names)
    missing_refs = sorted(dat_names - ref_names)
    missing_data = sorted(ref_names - dat_names)
    return datasets, missing_refs, missing_data


def count_data_points(dat_file: Path) -> int:
    n = 0
    with dat_file.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 3:
                continue
            try:
                float(cols[0])
                float(cols[1])
                float(cols[2])
            except ValueError:
                continue
            n += 1
    return n


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
    config_dir: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    pr_path = outdir / f"{dataset}_{variant}_pr.dat"
    fit_path = outdir / f"{dataset}_{variant}_fit.dat"
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
        "--fit-output",
        str(fit_path),
    ]
    if rebin > 0:
        cmd += ["--rebin", str(rebin)]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=config_dir)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        rendered_cmd = " ".join(cmd)
        raise RuntimeError(
            f"unfourier failed for {dataset}/{variant} "
            f"(exit {result.returncode}):\n{rendered_cmd}\n\n{result.stderr}"
        )

    pr = np.loadtxt(pr_path, comments="#")
    if pr.ndim != 2 or pr.shape[1] < 2:
        raise RuntimeError(f"unfourier produced malformed P(r) output: {pr_path}")

    fit = np.loadtxt(fit_path, comments="#")
    if fit.ndim != 2 or fit.shape[1] < 4:
        raise RuntimeError(f"unfourier produced malformed fit output: {fit_path}")
    return pr[:, :2], fit[:, :4], elapsed


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


def chi_squared_from_fit(fit: np.ndarray) -> float:
    sigma = np.where(fit[:, 3] > 0.0, fit[:, 3], 1.0)
    residual = (fit[:, 1] - fit[:, 2]) / sigma
    return float(np.mean(residual**2))


def make_result(
    dataset: str,
    variant: str,
    rebin: int,
    pr: np.ndarray,
    fit: np.ndarray,
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
    chi_squared = chi_squared_from_fit(fit)
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
        chi_squared=chi_squared,
        pr=pr,
        fit=fit,
    )


def peak_norm(p: np.ndarray) -> np.ndarray:
    peak = p.max()
    return p / peak if peak > 0 else p


def variant_label(result: RunResult) -> str:
    return result.variant if result.rebin == 0 else f"{result.variant}, rebin={result.rebin}"


def plot_results(results: list[RunResult], refs: dict[str, dict], out_path: Path) -> None:
    datasets = list(refs.keys())
    fig, axes = plt.subplots(3, len(datasets), figsize=(5.2 * len(datasets), 10.8), squeeze=False)

    fig.suptitle("unFourier spline validation vs GNOM reference and data fit", fontsize=13)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for col, name in enumerate(datasets):
        ref = refs[name]
        matching = [r for r in results if r.dataset == name]

        ax = axes[0, col]
        ax.plot(ref["r"], peak_norm(ref["pr"]), "k-", lw=2, label="GNOM ref")
        for idx, result in enumerate(matching):
            r = result.pr[:, 0]
            p = result.pr[:, 1]
            ax.plot(r, peak_norm(p), lw=1.5, color=colors[idx % len(colors)], label=variant_label(result))
        ax.set_title(name)
        ax.set_xlabel("r (A)")
        ax.set_ylabel("P(r), peak normalised")
        ax.set_xlim(0, ref["d_max"] * 1.05)
        ax.set_ylim(-0.08, 1.25)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[1, col]
        for idx, result in enumerate(matching):
            q = result.fit[:, 0]
            i_obs = result.fit[:, 1]
            i_calc = result.fit[:, 2]
            obs_mask = (q > 0.0) & (i_obs > 0.0)
            calc_mask = (q > 0.0) & (i_calc > 0.0)
            color = colors[idx % len(colors)]
            if idx == 0:
                ax.scatter(q[obs_mask], i_obs[obs_mask], s=7, color="black", alpha=0.4, label="data")
            elif len(q) != len(matching[0].fit[:, 0]):
                ax.scatter(q[obs_mask], i_obs[obs_mask], s=11, color=color, alpha=0.35, label=f"{variant_label(result)} data")
            ax.plot(q[calc_mask], i_calc[calc_mask], lw=1.4, color=color, label=f"{variant_label(result)} fit")
        ax.set_xlabel("q (1/A)")
        ax.set_ylabel("I(q)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[2, col]
        labels = [r.variant if r.rebin == 0 else f"rebin={r.rebin}" for r in matching]
        x = np.arange(len(matching))
        floor = 1e-12
        ax.bar(x - 0.2, [max(abs(r.left_slope), floor) for r in matching], width=0.2, label="abs left slope")
        ax.bar(x, [max(abs(r.right_slope), floor) for r in matching], width=0.2, label="abs right slope")
        ax.bar(x + 0.2, [max(abs(r.p_end), floor) for r in matching], width=0.2, label="abs P(Dmax)")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
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
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_ROOT,
        help="Working directory for unFourier runs; controls which unfourier.toml is loaded",
    )
    parser.add_argument("--method", choices=["gcv", "lcurve", "bayes"], default="gcv")
    parser.add_argument("--n-basis", type=int, default=20)
    parser.add_argument("--rebin-large", type=int, default=200)
    parser.add_argument(
        "--rebin-threshold",
        type=int,
        default=DEFAULT_REBIN_THRESHOLD,
        help="Also run a rebin variant for datasets with more than this many points",
    )
    parser.add_argument("--ise-threshold", type=float, default=0.15)
    parser.add_argument("--rg-rel-threshold", type=float, default=0.15)
    parser.add_argument(
        "--min-ise-passes",
        type=int,
        default=None,
        help="Minimum number of primary datasets that must pass ISE; default is all",
    )
    args = parser.parse_args()

    args.unfourier = args.unfourier.expanduser().resolve()
    args.dat_dir = args.dat_dir.expanduser().resolve()
    args.ref_dir = args.ref_dir.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    args.plot = args.plot.expanduser().resolve()
    args.config_dir = args.config_dir.expanduser().resolve()

    if not args.unfourier.exists():
        raise SystemExit(f"binary not found: {args.unfourier}\nRun: cargo build --release")
    if not args.dat_dir.exists() or not args.ref_dir.exists():
        raise SystemExit(
            f"real-data directories not found:\n  dat: {args.dat_dir}\n  ref: {args.ref_dir}"
        )
    if not args.config_dir.exists():
        raise SystemExit(f"config directory not found: {args.config_dir}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)

    datasets, missing_refs, missing_data = discover_datasets(args.dat_dir, args.ref_dir)
    if not datasets:
        raise SystemExit(
            f"no matching .dat/.out pairs found:\n  dat: {args.dat_dir}\n  ref: {args.ref_dir}"
        )
    if missing_refs:
        print(f"warning: no GNOM reference for: {', '.join(missing_refs)}")
    if missing_data:
        print(f"warning: no .dat input for: {', '.join(missing_data)}")

    refs: dict[str, dict] = {}
    results: list[RunResult] = []

    header = (
        f"{'Dataset':<10}  {'Variant':<12}  {'Rebin':>6}  {'ISE':>10}  "
        f"{'dRg/Rg':>9}  {'Rg':>8}  {'chi2':>9}  {'P(0)':>10}  {'P(Dmax)':>10}  "
        f"{'left dP':>10}  {'right dP':>10}  {'Time':>7}"
    )
    print(header)
    print("-" * len(header))

    for name in datasets:
        dat_file = args.dat_dir / f"{name}.dat"
        ref_file = args.ref_dir / f"{name}.out"

        ref = parse_gnom_out(ref_file)
        d_max = float(ref["d_max"])
        rg_ref = float(ref["rg"])
        pr_ref_arr = ref["pr"]
        r_ref = pr_ref_arr[:, 0]
        pr_ref = pr_ref_arr[:, 1]
        refs[name] = {"r": r_ref, "pr": pr_ref, "d_max": d_max}

        variants = [("spline", 0)]
        n_points = count_data_points(dat_file)
        if args.rebin_large > 0 and n_points > args.rebin_threshold:
            variants.append(("spline-rebin", args.rebin_large))

        for variant, rebin in variants:
            pr, fit, elapsed = run_unfourier(
                args.unfourier,
                dat_file,
                d_max,
                args.outdir,
                name,
                variant,
                args.method,
                args.n_basis,
                rebin,
                args.config_dir,
            )
            result = make_result(name, variant, rebin, pr, fit, elapsed, r_ref, pr_ref, rg_ref)
            results.append(result)

            ise_ok = result.ise < args.ise_threshold
            rg_ok = result.rg_rel_err < args.rg_rel_threshold
            rebin_str = str(rebin) if rebin > 0 else "-"
            print(
                f"{name:<10}  {variant:<12}  {rebin_str:>6}  "
                f"{result.ise:>7.4f} {'Y' if ise_ok else 'N':>2}  "
                f"{result.rg_rel_err:>6.4f} {'Y' if rg_ok else 'N':>2}  "
                f"{result.rg_unfourier:>8.3f}  {result.chi_squared:>9.3f}  {result.p0:>10.2e}  "
                f"{result.p_end:>10.2e}  {result.left_slope:>10.2e}  "
                f"{result.right_slope:>10.2e}  {result.elapsed_s:>7.2f}"
            )

    if refs and results:
        plot_results(results, refs, args.plot)
        print(f"\nValidation plot saved to: {args.plot}")

    primary = [r for r in results if r.variant == "spline" and r.rebin == 0]
    ise_pass = sum(r.ise < args.ise_threshold for r in primary)
    rg_all_pass = all(r.rg_rel_err < args.rg_rel_threshold for r in primary)
    required_ise = len(primary) if args.min_ise_passes is None else args.min_ise_passes
    required_ise = max(0, min(required_ise, len(primary)))
    overall_ok = (
        len(primary) == len(datasets)
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
