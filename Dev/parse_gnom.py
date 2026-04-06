"""
parse_gnom.py — Parser for GNOM .out files (v4.5 and v5.0).

Usage
-----
    from parse_gnom import parse_gnom_out

    result = parse_gnom_out("data/prs_ref/SASDME2.out")
    # result["pr"]    : np.ndarray, shape (N, 3) — columns R, P(R), ERROR
    # result["rg"]    : float
    # result["i0"]    : float
    # result["d_max"] : float

Standalone validation
---------------------
    python Dev/parse_gnom.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np


def parse_gnom_out(path: str | Path) -> dict:
    """Parse a GNOM .out file and return a dict with keys:
    ``pr`` (ndarray Nx3), ``rg`` (float), ``i0`` (float), ``d_max`` (float).

    Supports GNOM v4.5 and v5.0 output formats.
    """
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()

    rg: float | None = None
    i0: float | None = None
    d_max: float | None = None
    q_min: float | None = None
    q_max: float | None = None
    pr_rows: list[list[float]] = []

    # ------------------------------------------------------------------ #
    # Pass 1: extract scalar metadata (Rg, I(0), D_max)                  #
    # ------------------------------------------------------------------ #
    # Strip ANSI escape codes that grep embeds when it highlights matches.
    _ansi = re.compile(r"\x1b\[[0-9;]*m|\[K|\[01;31m|\[m")

    def clean(s: str) -> str:
        return _ansi.sub("", s)

    for line in lines:
        cl = clean(line)

        # --- Angular range (q_min, q_max) ---
        # v5.0: "Angular range:   0.0029 to   0.2999"
        # v4.5: "Angular range   :     from    0.0152   to    0.1968"
        if q_min is None and re.search(r"Angular\s+range", cl, re.IGNORECASE):
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cl)
            if len(nums) >= 2:
                q_min = float(nums[-2])
                q_max = float(nums[-1])

        # --- D_max ---
        if d_max is None:
            # v5.0: "Real space range:   0.0000 to   45.1900"
            # v4.5: "Real space range   :     from      0.00   to    139.20"
            if re.search(r"Real space range", cl, re.IGNORECASE):
                # Extract all floats; D_max is the last one.
                nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cl)
                if len(nums) >= 2:
                    d_max = float(nums[-1])
            # v5.0 alternative header: "Maximum characteristic size:  80.0000"
            elif re.search(r"Maximum characteristic size", cl, re.IGNORECASE):
                nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cl)
                if nums:
                    d_max = float(nums[-1])

        # --- Rg and I(0) ---
        if rg is None:
            # v5.0: "Real space Rg:   0.1420E+02 +-   0.4941E-01"
            m = re.search(
                r"Real space Rg\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                cl, re.IGNORECASE,
            )
            if m:
                rg = float(m.group(1))

        if i0 is None:
            # v5.0: "Real space I(0):   0.8560E+02 +-   0.2125E+00"
            m = re.search(
                r"Real space I\(0\)\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                cl, re.IGNORECASE,
            )
            if m:
                i0 = float(m.group(1))

        # v4.5: "Real space: Rg =   41.21 +- 0.114  I(0) =   0.6834E+01 +-  0.1801E-01"
        if rg is None or i0 is None:
            if re.search(r"Real space\s*:\s*Rg\s*=", cl, re.IGNORECASE):
                if rg is None:
                    m = re.search(
                        r"Rg\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                        cl, re.IGNORECASE,
                    )
                    if m:
                        rg = float(m.group(1))
                if i0 is None:
                    m = re.search(
                        r"I\(0\)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                        cl, re.IGNORECASE,
                    )
                    if m:
                        i0 = float(m.group(1))

    # ------------------------------------------------------------------ #
    # Pass 2: extract P(r) section                                        #
    # ------------------------------------------------------------------ #
    in_pr = False
    data_started = False

    for line in lines:
        cl = clean(line)

        if not in_pr:
            if "Distance distribution" in cl:
                in_pr = True
            continue

        # Inside the P(r) section: collect lines that parse as three floats.
        # Strip CR characters that \r\r\n endings inject into the cleaned line.
        cl = cl.strip()
        parts = cl.split()
        if len(parts) >= 3:
            try:
                row = [float(parts[0]), float(parts[1]), float(parts[2])]
                pr_rows.append(row)
                data_started = True
                continue
            except ValueError:
                pass

        # Skip blank lines — they appear between data rows in files with \r\r\n
        # endings and must not trigger the early-stop.
        if not cl:
            continue

        # Stop at the first non-empty, non-parseable line once data has started.
        if data_started:
            break

    pr = np.array(pr_rows, dtype=float) if pr_rows else np.empty((0, 3))

    return {
        "pr": pr,
        "rg": rg,
        "i0": i0,
        "d_max": d_max,
        "q_min": q_min,
        "q_max": q_max,
    }


# --------------------------------------------------------------------------- #
# Standalone validation                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    repo = Path(__file__).parent.parent
    ref_dir = repo / "data" / "prs_ref"

    files = sorted(ref_dir.glob("*.out"))
    if not files:
        print(f"No .out files found in {ref_dir}", file=sys.stderr)
        sys.exit(1)

    header = f"{'File':<16}  {'Rg':>10}  {'I(0)':>12}  {'D_max':>8}  {'P(r) pts':>9}"
    print(header)
    print("-" * len(header))

    all_ok = True
    for f in files:
        result = parse_gnom_out(f)
        rg = result["rg"]
        i0 = result["i0"]
        d_max = result["d_max"]
        n_pr = len(result["pr"])

        rg_str = f"{rg:.4f}" if rg is not None else "MISSING"
        i0_str = f"{i0:.4e}" if i0 is not None else "MISSING"
        dm_str = f"{d_max:.2f}" if d_max is not None else "MISSING"

        print(f"{f.name:<16}  {rg_str:>10}  {i0_str:>12}  {dm_str:>8}  {n_pr:>9}")

        if rg is None or i0 is None or d_max is None or n_pr == 0:
            print(f"  WARNING: incomplete parse for {f.name}", file=sys.stderr)
            all_ok = False

    if all_ok:
        print("\nAll files parsed successfully.")
    else:
        print("\nSome files had missing fields — check the warnings above.", file=sys.stderr)
        sys.exit(1)
