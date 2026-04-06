"""
validate_real_data.py — Compare unFourier P(r) against GNOM reference outputs.

For each of the three real SAXS datasets, this script:
  1. Parses D_max and Rg from the GNOM .out reference file.
  2. Runs unfourier (rect and spline bases) and captures P(r).
  3. For SASDYU3 (1696 pts) also runs with --rebin 200 and reports runtime.
  4. Computes ISE (integrated squared error, peak-normalised) and Rg agreement.
  5. Saves a 3×3 validation plot (validation_plot.png).

Pass criteria (per todo_m6.md):
  - ISE < 0.15
  - Rg within 15% of GNOM reference

Usage:
    python Dev/validate_real_data.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

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
PLOT_OUT = Path(__file__).parent / "validation_plot.png"

DATASETS = ["SASDME2", "SASDF42", "SASDYU3"]

# ---------------------------------------------------------------------------
# Basis visualisation constants (must match unfourier defaults)
# ---------------------------------------------------------------------------

NPOINTS_RECT = 100   # --npoints default
NBASIS_SPLINE = 20   # --n-basis default for spline
DEGREE_SPLINE = 3

# ---------------------------------------------------------------------------
# Helpers — validation
# ---------------------------------------------------------------------------


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
# Helpers — basis function visualisation
# ---------------------------------------------------------------------------

def _spline_knots(r_max: float, n_basis: int = NBASIS_SPLINE) -> np.ndarray:
    """Clamped cubic B-spline knot vector — mirrors Rust CubicBSpline::new()."""
    n_interior = n_basis - 2
    t_int = [r_max * i / (n_interior + 1) for i in range(1, n_interior + 1)]
    return np.array([0.0] * (DEGREE_SPLINE + 1) + t_int + [r_max] * (DEGREE_SPLINE + 1))


def _eval_bspline_j(knots: np.ndarray, j: int, r_vals: np.ndarray) -> np.ndarray:
    """Evaluate the j-th B-spline (0-indexed over full knot-vector set)."""
    n_total = len(knots) - DEGREE_SPLINE - 1
    c = np.zeros(n_total)
    c[j] = 1.0
    vals = BSpline(knots, c, DEGREE_SPLINE, extrapolate=False)(r_vals)
    return np.where(np.isfinite(vals), vals, 0.0)


def _spline_greville(knots: np.ndarray, n_basis: int = NBASIS_SPLINE) -> np.ndarray:
    """Greville abscissae for the free basis functions (indices 1..n_basis)."""
    return np.array([
        np.mean(knots[j + 1: j + DEGREE_SPLINE + 1])
        for j in range(1, n_basis + 1)
    ])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_validation_plot(
    plot_results: dict,
    plot_refs: dict,
    out_path: Path,
) -> None:
    """
    3×3 figure:
      col  → dataset (SASDME2 / SASDF42 / SASDYU3)
      row0 → P(r) comparison: GNOM ref, rect, spline (all peak-normalised)
      row1 → Rect basis: bar chart where each bar is one weighted top-hat
      row2 → Spline basis: B-spline functions + interior knots + Greville pts
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle(
        "unFourier validation — P(r) vs GNOM reference",
        fontsize=13, fontweight="bold", y=1.001,
    )

    row_ylabels = [
        "P(r) [peak-norm.]\ncomparison",
        "P(r) [peak-norm.]\nrect basis functions",
        "P(r) [peak-norm.]\nspline basis functions",
    ]
    for row, label in enumerate(row_ylabels):
        axes[row, 0].set_ylabel(label, fontsize=8)

    for col, name in enumerate(DATASETS):
        ref = plot_refs[name]
        d_max: float = ref["d_max"]
        r_ref: np.ndarray = ref["r"]
        pr_ref_n: np.ndarray = ref["pr"] / ref["pr"].max()

        rect_arr: np.ndarray | None = plot_results[name].get("rect")
        spline_arr: np.ndarray | None = plot_results[name].get("spline")

        # ------------------------------------------------------------------ #
        # Row 0 — P(r) comparison                                             #
        # ------------------------------------------------------------------ #
        ax = axes[0, col]
        ax.plot(r_ref, pr_ref_n, "k-", lw=2, label="GNOM ref", zorder=3)
        if rect_arr is not None and rect_arr[:, 1].max() > 0:
            r, p = rect_arr[:, 0], rect_arr[:, 1]
            ax.plot(r, p / p.max(), color="C0", lw=1.5, label="rect", zorder=2)
        if spline_arr is not None and spline_arr[:, 1].max() > 0:
            r, p = spline_arr[:, 0], spline_arr[:, 1]
            ax.plot(r, p / p.max(), color="C1", lw=1.5, label="spline", zorder=2)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, d_max * 1.05)
        ax.set_ylim(-0.08, 1.25)
        ax.axhline(0, color="k", lw=0.5)

        # ------------------------------------------------------------------ #
        # Row 1 — Rect: each bar is one weighted top-hat basis function        #
        # ------------------------------------------------------------------ #
        ax = axes[1, col]
        if rect_arr is not None and rect_arr[:, 1].max() > 0:
            r, p = rect_arr[:, 0], rect_arr[:, 1]
            n_pts = len(r)
            dr = d_max / n_pts
            p_n = p / p.max()
            cmap_r = plt.cm.viridis
            bar_colors = [cmap_r(j / max(n_pts - 1, 1)) for j in range(n_pts)]
            ax.bar(r, p_n, width=dr, color=bar_colors, edgecolor="none", align="center")
        else:
            ax.text(0.5, 0.5, "run failed", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=9)
        ax.set_xlim(0, d_max * 1.05)
        ax.set_ylim(-0.08, 1.25)
        ax.axhline(0, color="k", lw=0.5)

        # ------------------------------------------------------------------ #
        # Row 2 — Spline: B-splines, interior knots, Greville pts, P(r)       #
        # ------------------------------------------------------------------ #
        ax = axes[2, col]
        if spline_arr is not None and spline_arr[:, 1].max() > 0:
            r, p = spline_arr[:, 0], spline_arr[:, 1]
            p_n = p / p.max()

            knots = _spline_knots(d_max)
            # Interior knots are those between the clamped boundary repetitions
            interior_knots = knots[DEGREE_SPLINE + 1: -(DEGREE_SPLINE + 1)]
            greville = _spline_greville(knots)
            r_dense = np.linspace(0, d_max, 600)

            cmap_s = plt.cm.tab20
            n_free = NBASIS_SPLINE
            basis_scale = 0.35  # display basis fns at 35 % of plot height

            # Free B-spline basis functions (indices 1..n_free in full set)
            for k in range(n_free):
                j = k + 1
                b = _eval_bspline_j(knots, j, r_dense)
                ax.plot(r_dense, b * basis_scale, color=cmap_s(k / n_free),
                        lw=0.9, alpha=0.55)

            # Interior knot positions
            for kt in interior_knots:
                ax.axvline(kt, color="#888888", lw=0.6, ls="--", alpha=0.55, zorder=1)

            # Greville abscissae as downward triangles on the x-axis
            ax.plot(greville, np.full_like(greville, -0.055), "kv", ms=4,
                    zorder=5, clip_on=False, label="Greville pts")

            # P(r) curve on top
            ax.plot(r, p_n, color="#CC3311", lw=2, zorder=4, label="P(r)")
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.text(0.5, 0.5, "run failed", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=9)

        ax.set_xlim(0, d_max * 1.05)
        ax.set_ylim(-0.12, 1.25)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("r (Å)", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nValidation plot saved to: {out_path}")


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

    dataset_ise_pass: dict[str, bool] = {}
    rg_all_pass = True

    # Accumulators for the plot
    plot_results: dict[str, dict[str, np.ndarray | None]] = {
        name: {"rect": None, "spline": None} for name in DATASETS
    }
    plot_refs: dict[str, dict] = {}

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

        plot_refs[name] = {"r": r_ref, "pr": pr_ref, "d_max": d_max}

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

            if basis == "rect" and rebin == 0:
                dataset_ise_pass[name] = ise_ok
                plot_results[name]["rect"] = pr_arr
            if basis == "spline" and rebin == 0:
                plot_results[name]["spline"] = pr_arr
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
    ise_criterion_ok = n_ise_pass >= 2

    print()
    print(f"Pass criteria: ISE < {ISE_THRESHOLD} for ≥2/{n_datasets} datasets  |  ΔRg/Rg < {RG_REL_THRESHOLD} for all")
    print(f"  ISE (rect basis): {n_ise_pass}/{n_datasets} datasets pass  {'✓' if ise_criterion_ok else '✗'}")
    print(f"  Rg all datasets:  {'✓ PASS' if rg_all_pass else '✗ FAIL'}")

    if ise_criterion_ok and rg_all_pass:
        print("Overall: ALL CRITERIA MET")
    else:
        print("Overall: SOME CRITERIA NOT MET")

    # Generate the 3×3 validation plot regardless of pass/fail outcome
    make_validation_plot(plot_results, plot_refs, PLOT_OUT)

    if not (ise_criterion_ok and rg_all_pass):
        sys.exit(1)


if __name__ == "__main__":
    main()
