use crate::solver::Solution;
use anyhow::Result;
use std::io::{self, Write};
use std::path::Path;

/// Write a `Solution` as a whitespace-delimited text file (or to any `Write`r).
///
/// # Columns
///
/// Always present:
/// - Column 1: r (Å)
/// - Column 2: P(r)
///
/// Present when `solution.p_r_err` is `Some`:
/// - Column 3: σ_P(r)  (posterior standard deviation, added in M4)
///
/// # Header
///
/// A single comment line beginning with `#` describes the columns.
/// This format is intentionally simple and human-readable. Machine-readable
/// formats (JSON, XML) can be added as an alternative `--format` in M6.
pub fn write_pr<W: Write>(writer: &mut W, solution: &Solution) -> Result<()> {
    // Header
    match &solution.p_r_err {
        Some(_) => writeln!(writer, "# {:>14}  {:>14}  {:>14}", "r(A)", "P(r)", "sigma_P(r)")?,
        None => writeln!(writer, "# {:>14}  {:>14}", "r(A)", "P(r)")?,
    }

    for (i, (&r, &pr)) in solution.r.iter().zip(solution.p_r.iter()).enumerate() {
        match &solution.p_r_err {
            Some(err) => writeln!(writer, "  {:>14.6e}  {:>14.6e}  {:>14.6e}", r, pr, err[i])?,
            None => writeln!(writer, "  {:>14.6e}  {:>14.6e}", r, pr)?,
        }
    }

    Ok(())
}

/// Write P(r) to a file at the given path.
pub fn write_pr_to_file<P: AsRef<Path>>(path: P, solution: &Solution) -> Result<()> {
    let mut file = std::fs::File::create(path.as_ref())?;
    write_pr(&mut file, solution)
}

/// Write P(r) to stdout.
pub fn write_pr_to_stdout(solution: &Solution) -> Result<()> {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    write_pr(&mut handle, solution)
}

/// Print a brief diagnostic summary to stderr.
pub fn print_summary(solution: &Solution) {
    let p_max = solution.p_r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let r_at_peak = solution
        .r
        .iter()
        .zip(solution.p_r.iter())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(&r, _)| r)
        .unwrap_or(0.0);
    let d_max = solution
        .r
        .iter()
        .zip(solution.p_r.iter())
        .filter(|(_, &p)| p > 0.0)
        .last()
        .map(|(&r, _)| r)
        .unwrap_or(0.0);

    eprintln!("  χ²_red  = {:.4}", solution.chi_squared);
    eprintln!("  P(r) peak at r = {:.2} Å  (P_max = {:.4e})", r_at_peak, p_max);
    eprintln!("  D_max estimate = {:.2} Å  (last r with P(r) > 0)", d_max);
}
