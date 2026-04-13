use crate::solver::Solution;
use anyhow::{anyhow, Result};
use std::io::{self, Write};
use std::path::Path;

/// Evaluated P(r) output table.
///
/// This is deliberately separate from [`Solution`]: spline coefficients are
/// control weights, while this struct stores sampled values of the real-space
/// function after basis evaluation.
#[derive(Debug, Clone)]
pub struct PrCurve {
    pub r: Vec<f64>,
    pub p_r: Vec<f64>,
    pub p_r_err: Option<Vec<f64>>,
}

/// Write an evaluated P(r) curve as whitespace-delimited text.
///
/// # Columns
///
/// Always present:
/// - Column 1: r (Å)
/// - Column 2: P(r)
///
/// Present when `curve.p_r_err` is `Some`:
/// - Column 3: σ_P(r)  (posterior standard deviation, added in M4)
///
/// # Header
///
/// A single comment line beginning with `#` describes the columns.
/// This format is intentionally simple and human-readable. Machine-readable
/// formats (JSON, XML) can be added as an alternative `--format` in M6.
pub fn write_pr<W: Write>(writer: &mut W, curve: &PrCurve) -> Result<()> {
    if curve.r.len() != curve.p_r.len() {
        return Err(anyhow!(
            "P(r) output length mismatch: {} r values for {} P(r) values",
            curve.r.len(),
            curve.p_r.len()
        ));
    }
    if let Some(err) = &curve.p_r_err {
        if err.len() != curve.r.len() {
            return Err(anyhow!(
                "P(r) output length mismatch: {} r values for {} sigma values",
                curve.r.len(),
                err.len()
            ));
        }
    }

    // Header
    match &curve.p_r_err {
        Some(_) => writeln!(
            writer,
            "# {:>14}  {:>14}  {:>14}",
            "r(A)", "P(r)", "sigma_P(r)"
        )?,
        None => writeln!(writer, "# {:>14}  {:>14}", "r(A)", "P(r)")?,
    }

    for (i, (&r, &pr)) in curve.r.iter().zip(curve.p_r.iter()).enumerate() {
        match &curve.p_r_err {
            Some(err) => writeln!(writer, "  {:>14.6e}  {:>14.6e}  {:>14.6e}", r, pr, err[i])?,
            None => writeln!(writer, "  {:>14.6e}  {:>14.6e}", r, pr)?,
        }
    }

    Ok(())
}

/// Write P(r) to a file at the given path.
pub fn write_pr_to_file<P: AsRef<Path>>(path: P, curve: &PrCurve) -> Result<()> {
    let mut file = std::fs::File::create(path.as_ref())?;
    write_pr(&mut file, curve)
}

/// Write P(r) to stdout.
pub fn write_pr_to_stdout(curve: &PrCurve) -> Result<()> {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    write_pr(&mut handle, curve)
}

/// Write the back-calculated fit (q, I_obs, I_calc, sigma) to a writer.
///
/// This lets you check how well the solution reproduces the measured data.
/// A good fit to I(q) with a bad P(r) is a sign of over-fitting (M1 without
/// regularisation). A slightly worse fit with a smooth P(r) is expected and
/// desirable once Tikhonov regularisation is added in M2.
pub fn write_fit<W: Write>(
    writer: &mut W,
    solution: &Solution,
    q: &[f64],
    i_obs: &[f64],
    sigma: &[f64],
) -> Result<()> {
    writeln!(
        writer,
        "# {:>14}  {:>14}  {:>14}  {:>14}",
        "q(1/A)", "I_obs", "I_calc", "sigma"
    )?;
    for i in 0..q.len() {
        writeln!(
            writer,
            "  {:>14.6e}  {:>14.6e}  {:>14.6e}  {:>14.6e}",
            q[i], i_obs[i], solution.i_calc[i], sigma[i]
        )?;
    }
    Ok(())
}

/// Write the fit to a file at the given path.
pub fn write_fit_to_file<P: AsRef<Path>>(
    path: P,
    solution: &Solution,
    q: &[f64],
    i_obs: &[f64],
    sigma: &[f64],
) -> Result<()> {
    let mut file = std::fs::File::create(path.as_ref())?;
    write_fit(&mut file, solution, q, i_obs, sigma)
}

/// Print a brief diagnostic summary to stderr.
pub fn print_summary(solution: &Solution, curve: &PrCurve) {
    let p_max = curve.p_r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let r_at_peak = curve
        .r
        .iter()
        .zip(curve.p_r.iter())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(&r, _)| r)
        .unwrap_or(0.0);
    let d_max = curve
        .r
        .iter()
        .zip(curve.p_r.iter())
        .filter(|&(_, &p)| p > 0.0)
        .last()
        .map(|(&r, _)| r)
        .unwrap_or(0.0);

    eprintln!("  χ²_red  = {:.4}", solution.chi_squared);
    eprintln!(
        "  P(r) peak at r = {:.2} Å  (P_max = {:.4e})",
        r_at_peak, p_max
    );
    eprintln!("  D_max estimate = {:.2} Å  (last r with P(r) > 0)", d_max);
}
