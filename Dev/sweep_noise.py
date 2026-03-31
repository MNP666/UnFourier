#!/usr/bin/env python3
"""
sweep_noise.py — M3 validation: run unFourier at multiple noise levels and compare.

For each noise level k in [3, 5, 10, 20], generates Debye data (if not already
present), runs unFourier with GCV and L-curve selection, and compares the
recovered P(r) against the numerical reference.

Outputs
-------
- Console table: noise level, selected λ, χ²_red, ISE vs reference
- Plot 1: L-curve for each noise level (log RSS_w vs log ‖Lc‖²)  — NOT shown
          here (would need unfourier to dump the full grid; reserved for later)
- Plot 2: P(r) comparison across noise levels (GCV selection)
- Plot 3: ISE vs noise level for GCV and L-curve

Usage
-----
    python Dev/sweep_noise.py [--rmax 180] [--unfourier ./target/release/unfourier]

Requirements
------------
    numpy, matplotlib, scipy  (all standard scientific Python)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_debye(rg: float, k: float | None, outfile: Path, pr_ref: Path | None = None,
                   seed: int = 42) -> None:
    """Call gen_debye.py to create a synthetic data file."""
    gen = SCRIPT_DIR / "gen_debye.py"
    cmd = [sys.executable, str(gen), "--rg", str(rg), "--output", str(outfile),
           "--seed", str(seed)]
    if k is not None:
        cmd += ["--k", str(k)]
    if pr_ref is not None:
        cmd += ["--pr-reference", str(pr_ref)]
    subprocess.run(cmd, check=True)


def run_unfourier(unfourier_bin: str, dat_file: Path, pr_out: Path,
                  fit_out: Path, rmax: float, method: str,
                  npoints: int = 200) -> dict:
    """
    Run unfourier and return selected λ, χ²_red parsed from verbose output.

    Returns a dict with keys: lambda_selected, chi_sq
    """
    cmd = [
        unfourier_bin,
        str(dat_file),
        "--rmax", str(rmax),
        "--npoints", str(npoints),
        "--method", method,
        "--output", str(pr_out),
        "--fit-output", str(fit_out),
        "--verbose",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR running unfourier:\n{result.stderr}", file=sys.stderr)
        return {"lambda_selected": float("nan"), "chi_sq": float("nan")}

    # Parse 'selected λ = X.XXe-X  (λ_eff = ...)' from stderr
    lam = float("nan")
    chi = float("nan")
    for line in result.stderr.splitlines():
        if "selected λ" in line:
            try:
                lam = float(line.split("=")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        if "χ²_red" in line and "selected" not in line:
            try:
                parts = line.split("χ²_red")
                if len(parts) > 1:
                    chi = float(parts[1].strip().split()[0].strip(":= "))
            except (IndexError, ValueError):
                pass

    return {"lambda_selected": lam, "chi_sq": chi}


def load_two_col(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1]


def integrated_squared_error(r_calc: np.ndarray, pr_calc: np.ndarray,
                              r_ref: np.ndarray,  pr_ref: np.ndarray) -> float:
    """
    ISE = ∫ [P_calc(r) − P_ref(r)]² dr, normalised so that ∫ P_ref² dr = 1.

    Interpolates P_ref onto r_calc grid using the overlapping range.
    """
    r_lo = max(r_calc[0],  r_ref[0])
    r_hi = min(r_calc[-1], r_ref[-1])
    mask = (r_calc >= r_lo) & (r_calc <= r_hi)
    r_common = r_calc[mask]
    if r_common.size < 2:
        return float("nan")

    f_ref  = interp1d(r_ref, pr_ref, kind="linear", bounds_error=False, fill_value=0.0)
    pr_ref_interp = f_ref(r_common)
    pr_calc_interp = pr_calc[mask]

    # Normalise both to their peak (avoids ISE being dominated by scale difference)
    peak_ref  = pr_ref_interp.max()
    peak_calc = pr_calc_interp.max()
    if peak_ref < 1e-12 or peak_calc < 1e-12:
        return float("nan")
    pr_ref_n  = pr_ref_interp  / peak_ref
    pr_calc_n = pr_calc_interp / peak_calc

    ise_num   = np.trapz((pr_calc_n - pr_ref_n) ** 2, r_common)
    ise_denom = np.trapz(pr_ref_n ** 2, r_common)
    return ise_num / ise_denom if ise_denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rg",       type=float, default=30.0,  help="Rg of Debye chain (Å)")
    parser.add_argument("--rmax",     type=float, default=180.0, help="r_max for unFourier (Å)")
    parser.add_argument("--npoints",  type=int,   default=200,   help="r grid points")
    parser.add_argument("--unfourier", default=str(REPO_ROOT / "target/release/unfourier"),
                        help="Path to the unfourier binary")
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[3, 5, 10, 20],
                        help="Noise parameter k values to sweep (σ = I/k)")
    parser.add_argument("--no-plot",  action="store_true", help="Skip matplotlib output")
    args = parser.parse_args()

    unfourier_bin = args.unfourier
    if not Path(unfourier_bin).exists():
        sys.exit(
            f"ERROR: unfourier binary not found at '{unfourier_bin}'.\n"
            "Build it first with: cargo build --release"
        )

    tmp = SCRIPT_DIR / "_sweep_tmp"
    tmp.mkdir(exist_ok=True)

    # ---- Generate noiseless reference P(r) ----
    pr_ref_file = tmp / "debye_pr_ref.dat"
    if not pr_ref_file.exists():
        print("Generating reference P(r)...", flush=True)
        noiseless_file = tmp / "debye_noiseless.dat"
        generate_debye(args.rg, None, noiseless_file, pr_ref=pr_ref_file)

    r_ref, pr_ref = load_two_col(pr_ref_file)

    print(
        f"\n{'k':>4}  {'method':>8}  {'λ_sel':>10}  {'χ²_red':>8}  {'ISE':>10}"
    )
    print("-" * 48)

    results: dict[str, list] = {
        "k":          [],
        "lam_gcv":    [],
        "lam_lcurve": [],
        "ise_gcv":    [],
        "ise_lcurve": [],
        "chi_gcv":    [],
        "chi_lcurve": [],
    }

    for k in args.noise_levels:
        dat_file = tmp / f"debye_k{k:.0f}.dat"
        if not dat_file.exists():
            print(f"  Generating debye_k{k:.0f}.dat ...", flush=True)
            generate_debye(args.rg, k, dat_file)

        for method in ("gcv", "lcurve"):
            pr_out  = tmp / f"pr_{method}_k{k:.0f}.dat"
            fit_out = tmp / f"fit_{method}_k{k:.0f}.dat"

            info = run_unfourier(unfourier_bin, dat_file, pr_out, fit_out,
                                 args.rmax, method, args.npoints)

            if pr_out.exists():
                r_calc, pr_calc = load_two_col(pr_out)
                ise = integrated_squared_error(r_calc, pr_calc, r_ref, pr_ref)
            else:
                ise = float("nan")

            lam = info["lambda_selected"]
            chi = info["chi_sq"]

            print(f"{k:>4.0f}  {method:>8}  {lam:>10.3e}  {chi:>8.4f}  {ise:>10.4e}")

            results["k"].append(k)
            if method == "gcv":
                results["lam_gcv"].append(lam)
                results["ise_gcv"].append(ise)
                results["chi_gcv"].append(chi)
            else:
                results["lam_lcurve"].append(lam)
                results["ise_lcurve"].append(ise)
                results["chi_lcurve"].append(chi)

    print()

    if args.no_plot:
        return

    # ---- Plots ----
    noise_levels = sorted(set(results["k"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"M3 sweep — Debye chain Rg = {args.rg} Å, r_max = {args.rmax} Å",
        fontsize=12
    )

    # Panel 1: P(r) comparison for GCV method
    ax = axes[0]
    ax.plot(r_ref, pr_ref / pr_ref.max(), "k--", lw=2, label="Reference", zorder=5)
    cmap = plt.cm.viridis
    for i, k in enumerate(noise_levels):
        pr_out = tmp / f"pr_gcv_k{k:.0f}.dat"
        if not pr_out.exists():
            continue
        r_calc, pr_calc = load_two_col(pr_out)
        peak = pr_calc.max()
        color = cmap(i / max(len(noise_levels) - 1, 1))
        ax.plot(r_calc, pr_calc / max(peak, 1e-12), color=color, label=f"k={k:.0f}")
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("P(r)  [normalised]")
    ax.set_title("P(r) — GCV selection")
    ax.legend(fontsize=8)
    ax.set_xlim(0, args.rmax)
    ax.set_ylim(bottom=-0.05)

    # Panel 2: selected λ vs noise level
    ax = axes[1]
    ks_unique = noise_levels
    ax.loglog(ks_unique, results["lam_gcv"],    "o-", label="GCV",     color="steelblue")
    ax.loglog(ks_unique, results["lam_lcurve"], "s-", label="L-curve", color="darkorange")
    ax.set_xlabel("Noise level k  (σ = I/k)")
    ax.set_ylabel("Selected λ")
    ax.set_title("Auto-selected λ vs noise")
    ax.legend()
    ax.invert_xaxis()  # decreasing k = increasing noise, left to right

    # Panel 3: ISE vs noise level
    ax = axes[2]
    ax.semilogy(ks_unique, results["ise_gcv"],    "o-", label="GCV",     color="steelblue")
    ax.semilogy(ks_unique, results["ise_lcurve"], "s-", label="L-curve", color="darkorange")
    ax.set_xlabel("Noise level k  (σ = I/k)")
    ax.set_ylabel("ISE (normalised)")
    ax.set_title("Integrated Squared Error vs noise")
    ax.legend()
    ax.invert_xaxis()

    plt.tight_layout()
    out_fig = tmp / "sweep_noise.png"
    plt.savefig(out_fig, dpi=150)
    print(f"Plot saved to '{out_fig}'")
    plt.show()


if __name__ == "__main__":
    main()
