#!/usr/bin/env python3
"""
validate_spline.py - spline-only regression checks for unFourier 0.9.

This replaces the historical basis-comparison validator. The active product
surface is cubic B-spline only, so this script checks:

  1. Rust spline kernel sanity via cargo test.
  2. Endpoint values for value_zero and value_slope_zero.
  3. Endpoint slopes for value_slope_zero.
  4. Debye-chain P(r) recovery against the numerical reference.
  5. GCV, L-curve, and Bayesian modes all complete with positive P(r).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unfourier_matplotlib")

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def mark(ok: bool) -> str:
    return PASS if ok else FAIL


def generate_debye(tmp: Path, rg: float, k: float) -> tuple[Path, Path]:
    dat = tmp / "debye_k5.dat"
    ref = tmp / "debye_pr_ref.dat"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "gen_debye.py"),
        "--rg",
        str(rg),
        "--k",
        str(k),
        "--output",
        str(dat),
        "--pr-reference",
        str(ref),
    ]
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR, capture_output=True, text=True)
    return dat, ref


def write_config(path: Path, boundary: str, d1: float = 0.0, d2: float = 1.0) -> None:
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
    dat: Path,
    outdir: Path,
    label: str,
    rmax: float,
    method: str,
    n_basis: int,
    boundary: str = "value_zero",
) -> np.ndarray:
    run_dir = outdir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(run_dir / "unfourier.toml", boundary)

    pr_out = run_dir / "pr.dat"
    cmd = [
        str(binary),
        str(dat),
        "--rmax",
        str(rmax),
        "--method",
        method,
        "--n-basis",
        str(n_basis),
        "--output",
        str(pr_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=run_dir)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stderr}")

    pr = np.loadtxt(pr_out, comments="#")
    if pr.ndim != 2 or pr.shape[1] < 2:
        raise RuntimeError(f"{label} produced malformed P(r) output")
    return pr[:, :2]


def load_two_col(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1]


def ise(r_calc: np.ndarray, pr_calc: np.ndarray, r_ref: np.ndarray, pr_ref: np.ndarray) -> float:
    lo = max(float(r_calc[0]), float(r_ref[0]))
    hi = min(float(r_calc[-1]), float(r_ref[-1]))
    mask = (r_calc >= lo) & (r_calc <= hi)
    r_common = r_calc[mask]
    if r_common.size < 2:
        return float("nan")

    ref_interp = np.interp(r_common, r_ref, pr_ref, left=0.0, right=0.0)
    calc = pr_calc[mask]
    peak_ref = ref_interp.max()
    peak_calc = calc.max()
    if peak_ref <= 0.0 or peak_calc <= 0.0:
        return float("nan")

    ref_n = ref_interp / peak_ref
    calc_n = calc / peak_calc
    denom = np.trapezoid(ref_n**2, r_common)
    return float(np.trapezoid((calc_n - ref_n) ** 2, r_common) / denom)


def endpoint_slope(pr: np.ndarray) -> tuple[float, float]:
    r = pr[:, 0]
    p = pr[:, 1]
    left = (p[1] - p[0]) / (r[1] - r[0])
    right = (p[-1] - p[-2]) / (r[-1] - r[-2])
    return float(left), float(right)


def check_kernel() -> bool:
    result = subprocess.run(
        ["cargo", "test", "kernel_saxs_realistic", "--", "--nocapture"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    ok = result.returncode == 0
    print(f"  kernel_saxs_realistic cargo test: {mark(ok)}")
    if not ok:
        print(result.stdout[-1000:])
        print(result.stderr[-1000:])
    return ok


def check_methods(
    binary: Path,
    dat: Path,
    tmp: Path,
    r_ref: np.ndarray,
    pr_ref: np.ndarray,
    rmax: float,
    n_basis: int,
    ise_threshold: float,
) -> tuple[bool, dict[str, np.ndarray]]:
    print("\n--- Method and Debye recovery checks ---")
    curves: dict[str, np.ndarray] = {}
    all_ok = True
    for method in ("gcv", "lcurve", "bayes"):
        pr = run_unfourier(binary, dat, tmp, f"method_{method}", rmax, method, n_basis)
        curves[method] = pr
        score = ise(pr[:, 0], pr[:, 1], r_ref, pr_ref)
        endpoints_ok = abs(pr[0, 1]) < 1e-12 and abs(pr[-1, 1]) < 1e-12
        positive = pr[:, 1].max() > 0.0
        score_ok = score < ise_threshold
        ok = endpoints_ok and positive and score_ok
        all_ok = all_ok and ok
        print(
            f"  {method:<7} ISE={score:.4f}  endpoint_zero={endpoints_ok}  "
            f"positive={positive}  {mark(ok)}"
        )
    return all_ok, curves


def check_boundaries(binary: Path, dat: Path, tmp: Path, rmax: float, n_basis: int) -> bool:
    print("\n--- Boundary mode checks ---")
    value_zero = run_unfourier(
        binary, dat, tmp, "boundary_value_zero", rmax, "gcv", n_basis, "value_zero"
    )
    value_slope_zero = run_unfourier(
        binary,
        dat,
        tmp,
        "boundary_value_slope_zero",
        rmax,
        "gcv",
        n_basis,
        "value_slope_zero",
    )

    endpoints_ok = (
        abs(value_zero[0, 1]) < 1e-12
        and abs(value_zero[-1, 1]) < 1e-12
        and abs(value_slope_zero[0, 1]) < 1e-12
        and abs(value_slope_zero[-1, 1]) < 1e-12
    )
    left, right = endpoint_slope(value_slope_zero)
    slope_ok = abs(left) < 1e-3 and abs(right) < 1e-3
    plateau_ok = np.count_nonzero(np.isclose(value_zero[:8, 1], 0.0, atol=1e-14)) == 1

    print(f"  endpoint values zero:              {mark(endpoints_ok)}")
    print(f"  value_slope_zero endpoint slopes:  left={left:.3e}, right={right:.3e}  {mark(slope_ok)}")
    print(f"  value_zero avoids small-r plateau: {mark(plateau_ok)}")
    return endpoints_ok and slope_ok and plateau_ok


def plot_curves(curves: dict[str, np.ndarray], r_ref: np.ndarray, pr_ref: np.ndarray, out: Path) -> None:
    if not HAS_MPL:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(r_ref, pr_ref / pr_ref.max(), "k--", lw=2, label="Debye reference")
    for method, pr in curves.items():
        p = pr[:, 1]
        plt.plot(pr[:, 0], p / max(p.max(), 1e-15), label=method)
    plt.xlabel("r (A)")
    plt.ylabel("P(r), peak normalised")
    plt.title("Spline validation: Debye recovery")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unfourier", type=Path, default=REPO_ROOT / "target/release/unfourier")
    parser.add_argument("--rmax", type=float, default=150.0)
    parser.add_argument("--rg", type=float, default=30.0)
    parser.add_argument("--k", type=float, default=5.0)
    parser.add_argument("--n-basis", type=int, default=20)
    parser.add_argument("--ise-threshold", type=float, default=0.5)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if not args.unfourier.exists():
        raise SystemExit(f"unfourier binary not found at {args.unfourier}. Run: cargo build --release")

    with tempfile.TemporaryDirectory(prefix="unfourier_spline_validate_") as tmpdir:
        tmp = Path(tmpdir)
        dat, ref = generate_debye(tmp, args.rg, args.k)
        r_ref, pr_ref = load_two_col(ref)

        print("--- Kernel sanity ---")
        kernel_ok = check_kernel()
        boundary_ok = check_boundaries(args.unfourier, dat, tmp, args.rmax, args.n_basis)
        methods_ok, curves = check_methods(
            args.unfourier, dat, tmp, r_ref, pr_ref, args.rmax, args.n_basis, args.ise_threshold
        )

        if not args.no_plot:
            out = SCRIPT_DIR / "validate_spline.png"
            plot_curves(curves, r_ref, pr_ref, out)
            if HAS_MPL:
                print(f"\nPlot saved to {out}")

    all_ok = kernel_ok and boundary_ok and methods_ok
    print("\nSpline Validation Summary")
    print("=" * 50)
    print(f"  {mark(kernel_ok)}  Kernel sanity")
    print(f"  {mark(boundary_ok)}  Boundary modes")
    print(f"  {mark(methods_ok)}  Debye recovery and methods")
    print("=" * 50)
    print(f"  Overall: {mark(all_ok)}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
