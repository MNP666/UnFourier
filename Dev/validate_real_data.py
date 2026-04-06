"""
validate_real_data.py — Compare unFourier P(r) against GNOM reference outputs.

For each of the three real SAXS datasets, this script:
  1. Parses D_max and Rg from the GNOM .out reference file.
  2. Runs unfourier (rect and spline bases) and captures P(r).
  3. For SASDYU3 (1696 pts) also runs with --rebin 200 and reports runtime.
  4. Computes ISE (integrated squared error, peak-normalised) and Rg agreement.

Pass criteria (per todo_m6.md):
  - ISE < 0.15
  - Rg within 15% of GNOM reference

Usage:
    python Dev/validate_real_data.py
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Make parse_gnom importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).parent))
from parse_gnom import parse_gnom_out

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).parent.parent
BIN = REPO / "target" / "release" / "unfourier"
DAT_DIR = REPO / "data" / "dat_ref"
REF_DIR = REPO / "data" / "prs_ref"

DATASETS = ["SASDME2", "SASDF42", "SASDYU3"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def run_unfourier(
    dat_file: Path,
    rmax: float,
    basis: str = "rect",
    rebin: int = 0,
    extra_args: list[str] | None = None,
) -> tuple[np.ndarray, float]:
    """Run unfourier and return (pr_array shape Nx2, elapsed_seconds).

    pr_array columns: r, P(r).  For Bayesian output a third column (sigma)
    is present but ignored here.
    """
    cmd = [
        str(BIN),
        str(dat_file),
        "--rmax", str(rmax),
        "--negative-handling", "clip",
        "--method", "gcv",
        "--basis", basis,
    ]
    if rebin > 0:
        cmd += ["--rebin", str(rebin)]
    if extra_args:
        cmd += extra_args

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"unfourier failed (exit {result.returncode}):\n{result.stderr}"
        )

    rows = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass

    if not rows:
        raise RuntimeError(f"unfourier produced no P(r) output:\n{result.stdout}")

    return np.array(rows), elapsed


def rg_from_pr(r: np.ndarray, pr: np.ndarray) -> float:
    """Estimate Rg from P(r): Rg² = ∫r²P(r)dr / (2∫P(r)dr)."""
    denom = np.trapezoid(pr, r)
    if denom <= 0:
        return float("nan")
    numer = np.trapezoid(r**2 * pr, r)
    return np.sqrt(max(numer / (2.0 * denom), 0.0))


def ise_normalised(
    r_ref: np.ndarray,
    pr_ref: np.ndarray,
    r_new: np.ndarray,
    pr_new: np.ndarray,
) -> float:
    """Integrated squared error between two P(r) curves, both normalised to
    their respective peak values.  Interpolates pr_new onto r_ref grid."""
    peak_ref = pr_ref.max()
    peak_new = pr_new.max()
    if peak_ref <= 0 or peak_new <= 0:
        return float("nan")

    pr_ref_n = pr_ref / peak_ref
    pr_new_interp = np.interp(r_ref, r_new, pr_new / peak_new, left=0.0, right=0.0)

    return float(np.trapezoid((pr_ref_n - pr_new_interp) ** 2, r_ref))


def fmt_pass(value: float, threshold: float) -> str:
    mark = "PASS" if value < threshold else "FAIL"
    return f"{value:.4f}  [{mark}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not BIN.exists():
        print(f"ERROR: binary not found at {BIN}", file=sys.stderr)
        print("Run: cargo build --release", file=sys.stderr)
        sys.exit(1)

    ISE_THRESHOLD = 0.15
    RG_REL_THRESHOLD = 0.15

    # Per-dataset pass tracking (ISE criterion: at least 2/3 datasets must pass
    # on rect basis, per todo_m6.md expected outcomes).
    dataset_ise_pass: dict[str, bool] = {}
    rg_all_pass = True

    header_fields = [
        f"{'Dataset':<14}",
        f"{'Basis':<8}",
        f"{'Rebin':>6}",
        f"{'ISE':>10}",
        f"{'ΔRg/Rg':>10}",
        f"{'Rg_unf':>8}",
        f"{'Rg_ref':>8}",
        f"{'Time(s)':>8}",
    ]
    header = "  ".join(header_fields)
    print(header)
    print("-" * len(header))

    for name in DATASETS:
        dat_file = DAT_DIR / f"{name}.dat"
        ref_file = REF_DIR / f"{name}.out"

        ref = parse_gnom_out(ref_file)
        d_max: float = ref["d_max"]
        rg_ref: float = ref["rg"]
        pr_ref_arr: np.ndarray = ref["pr"]  # columns: R, P(R), ERROR

        r_ref = pr_ref_arr[:, 0]
        pr_ref = pr_ref_arr[:, 1]

        # Collect runs: (basis, rebin)
        runs: list[tuple[str, int]] = [("rect", 0), ("spline", 0)]
        if name == "SASDYU3":
            runs.append(("rect", 200))

        for basis, rebin in runs:
            try:
                pr_arr, elapsed = run_unfourier(dat_file, d_max, basis=basis, rebin=rebin)
            except RuntimeError as exc:
                row_fields = [
                    f"{name:<14}",
                    f"{basis:<8}",
                    f"{rebin:>6}",
                    f"{'ERROR':>10}",
                    f"{'ERROR':>10}",
                    f"{'—':>8}",
                    f"{rg_ref:>8.3f}",
                    f"{'—':>8}",
                ]
                print("  ".join(row_fields))
                print(f"  >> {exc}", file=sys.stderr)
                rg_all_pass = False
                continue

            r_new = pr_arr[:, 0]
            pr_new = pr_arr[:, 1]

            rg_new = rg_from_pr(r_new, pr_new)
            ise = ise_normalised(r_ref, pr_ref, r_new, pr_new)
            rg_rel_err = abs(rg_new - rg_ref) / rg_ref if rg_ref > 0 else float("nan")

            ise_ok = ise < ISE_THRESHOLD
            rg_ok = rg_rel_err < RG_REL_THRESHOLD

            # Track per-dataset ISE (use rect basis as the primary indicator).
            # Rebin runs are diagnostic only — excluded from pass criteria.
            if basis == "rect" and rebin == 0:
                dataset_ise_pass[name] = ise_ok
            if rebin == 0 and not rg_ok:
                rg_all_pass = False

            rebin_str = str(rebin) if rebin > 0 else "—"
            row_fields = [
                f"{name:<14}",
                f"{basis:<8}",
                f"{rebin_str:>6}",
                f"{ise:>7.4f} {'✓' if ise_ok else '✗':>2}",
                f"{rg_rel_err:>7.4f} {'✓' if rg_ok else '✗':>2}",
                f"{rg_new:>8.3f}",
                f"{rg_ref:>8.3f}",
                f"{elapsed:>8.2f}",
            ]
            print("  ".join(row_fields))

    n_ise_pass = sum(dataset_ise_pass.values())
    n_datasets = len(DATASETS)
    ise_criterion_ok = n_ise_pass >= 2  # at least 2/3 datasets must pass ISE

    print()
    print(f"Pass criteria: ISE < {ISE_THRESHOLD} for ≥2/{n_datasets} datasets  |  ΔRg/Rg < {RG_REL_THRESHOLD} for all")
    print(f"  ISE (rect basis): {n_ise_pass}/{n_datasets} datasets pass  {'✓' if ise_criterion_ok else '✗'}")
    print(f"  Rg all datasets:  {'✓ PASS' if rg_all_pass else '✗ FAIL'}")

    if ise_criterion_ok and rg_all_pass:
        print("Overall: ALL CRITERIA MET")
    else:
        print("Overall: SOME CRITERIA NOT MET")
        sys.exit(1)


if __name__ == "__main__":
    main()
