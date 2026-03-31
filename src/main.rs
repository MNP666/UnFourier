use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use unfourier::{
    basis::{BasisSet, UniformGrid},
    data::parse_dat,
    kernel::build_weighted_system,
    lambda_select::{
        estimate_lambda_range, evaluate_lambda_grid, log_lambda_grid, GcvSelector,
        GridMatrices, LCurveSelector, LambdaSelector,
    },
    output::{print_summary, write_fit_to_file, write_pr_to_file, write_pr_to_stdout},
    preprocess::{Identity, PreprocessingPipeline},
    solver::{Solver, TikhonovSolver},
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// How to choose the regularisation strength λ.
#[derive(Debug, Clone, ValueEnum)]
enum Method {
    /// Minimise the Generalised Cross-Validation score (default).
    Gcv,
    /// Find the corner of the L-curve (log residual vs log solution norm).
    Lcurve,
    /// Supply λ manually via --lambda. Requires --lambda to be set.
    Manual,
}

#[derive(Parser, Debug)]
#[command(
    name = "unfourier",
    about = "Indirect Fourier Transformation of SAXS data\n\nReads a 3-column .dat file (q, I(q), σ(q)) and computes the\npair-distance distribution function P(r).",
    version
)]
struct Args {
    /// Input .dat file: whitespace-delimited columns q  I(q)  σ(q).
    /// Lines beginning with '#' are treated as comments.
    input: PathBuf,

    /// λ selection method.
    /// 'gcv'    — minimise generalised cross-validation score (default).
    /// 'lcurve' — find corner of maximum curvature on the L-curve.
    /// 'manual' — use the value supplied by --lambda directly.
    ///
    /// If --lambda is given without --method, 'manual' is implied.
    #[arg(long, value_enum)]
    method: Option<Method>,

    /// Tikhonov regularisation strength λ (manual mode only).
    /// Larger λ → smoother P(r) but worse fit to I(q).
    /// If --method is not also given, implies --method manual.
    #[arg(long)]
    lambda: Option<f64>,

    /// Maximum r value in Å.
    /// Defaults to π / q_min (a rough upper bound on D_max).
    #[arg(long)]
    rmax: Option<f64>,

    /// Number of r grid points.
    #[arg(long, default_value_t = 100)]
    npoints: usize,

    /// Number of λ candidates in the automatic search grid.
    /// More points = finer search but slower. 60 is usually sufficient.
    #[arg(long, default_value_t = 60)]
    lambda_count: usize,

    /// Lower bound of the λ search grid (user-facing, before internal scaling).
    /// Defaults to 1e-6. Decrease if the auto-selected λ hits this floor.
    #[arg(long)]
    lambda_min: Option<f64>,

    /// Upper bound of the λ search grid (user-facing, before internal scaling).
    /// Defaults to 1e3. Increase if the auto-selected λ hits this ceiling.
    #[arg(long)]
    lambda_max: Option<f64>,

    /// Write P(r) to this file. Defaults to stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Write back-calculated fit (q, I_obs, I_calc, sigma) to this file.
    #[arg(long)]
    fit_output: Option<PathBuf>,

    /// Print diagnostic summary (χ², selected λ, D_max estimate) to stderr.
    #[arg(short, long)]
    verbose: bool,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    // Resolve the effective method: explicit --method, or infer from --lambda.
    let method = match (&args.method, args.lambda) {
        (Some(m), _) => m.clone(),
        (None, Some(_)) => Method::Manual,
        (None, None) => Method::Gcv, // default
    };

    if matches!(method, Method::Manual) && args.lambda.is_none() {
        return Err(anyhow!(
            "--method manual requires --lambda to be specified"
        ));
    }

    // ---- 1. Parse -------------------------------------------------------
    let raw_data = parse_dat(&args.input)?;
    if args.verbose {
        eprintln!(
            "Loaded {} data points from '{}'",
            raw_data.len(),
            args.input.display()
        );
        eprintln!(
            "  q range: [{:.4e}, {:.4e}] Å⁻¹",
            raw_data.q_min(),
            raw_data.q_max()
        );
    }

    // ---- 2. Preprocess ---------------------------------------------------
    let pipeline = PreprocessingPipeline::new().add(Box::new(Identity));
    let data = pipeline.run(raw_data);

    // ---- 3. r grid -------------------------------------------------------
    let r_max = args.rmax.unwrap_or_else(|| {
        let estimated = std::f64::consts::PI / data.q_min();
        if args.verbose {
            eprintln!("  r_max = {:.2} Å  (auto: π / q_min)", estimated);
        }
        estimated
    });
    if args.verbose && args.rmax.is_some() {
        eprintln!("  r_max = {:.2} Å  (user-specified)", r_max);
    }

    let basis = UniformGrid::new(r_max, args.npoints);
    if args.verbose {
        eprintln!(
            "  r grid: {} points, Δr = {:.4} Å",
            args.npoints,
            basis.delta_r()
        );
    }

    // ---- 4. Weighted system ---------------------------------------------
    let (k_weighted, i_weighted) = build_weighted_system(&basis, &data);
    let k_unweighted = basis.build_kernel_matrix(&data.q);

    // ---- 5. Solve --------------------------------------------------------
    let solution = match method {
        // -- Manual: single Tikhonov solve with user-supplied λ ---------------
        Method::Manual => {
            let lambda = args.lambda.unwrap();
            if args.verbose {
                eprintln!("  method: manual  λ = {:.3e}", lambda);
            }
            let solver = TikhonovSolver::new(lambda);
            solver.solve(
                &k_weighted,
                &i_weighted,
                &k_unweighted,
                &data.intensity,
                &data.error,
                basis.r_values(),
            )?
        }

        // -- Automatic: evaluate λ grid, select best --------------------------
        Method::Gcv | Method::Lcurve => {
            let matrices = GridMatrices::build(
                &k_weighted,
                &i_weighted,
                &k_unweighted,
                &data.intensity,
                &data.error,
                basis.r_values(),
            );

            let (default_min, default_max) = estimate_lambda_range(&matrices);
            let lam_min = args.lambda_min.unwrap_or(default_min);
            let lam_max = args.lambda_max.unwrap_or(default_max);
            let grid = log_lambda_grid(lam_min, lam_max, args.lambda_count);

            let method_name = match method {
                Method::Gcv => "gcv",
                Method::Lcurve => "lcurve",
                Method::Manual => unreachable!(),
            };

            if args.verbose {
                eprintln!(
                    "  method: {}  grid: {} pts in [{:.2e}, {:.2e}]",
                    method_name, args.lambda_count, lam_min, lam_max
                );
            }

            let evaluations = evaluate_lambda_grid(&grid, &matrices)?;

            let selector: Box<dyn LambdaSelector> = match method {
                Method::Gcv => Box::new(GcvSelector),
                Method::Lcurve => Box::new(LCurveSelector),
                Method::Manual => unreachable!(),
            };

            let best_idx = selector.select(&evaluations);
            let best = &evaluations[best_idx];

            if args.verbose {
                eprintln!(
                    "  selected λ = {:.3e}  (λ_eff = {:.3e})",
                    best.lambda, best.lambda_eff
                );
                eprintln!(
                    "  {:>8}  {:>12}  {:>12}  {:>10}  {:>10}",
                    "λ", "λ_eff", "GCV", "RSS_w", "χ²_red"
                );
                for (i, e) in evaluations.iter().enumerate() {
                    let marker = if i == best_idx { " ←" } else { "" };
                    eprintln!(
                        "  {:>8.2e}  {:>12.3e}  {:>12.4e}  {:>10.3e}  {:>10.4}{marker}",
                        e.lambda, e.lambda_eff, e.gcv, e.rss_weighted, e.chi_squared
                    );
                }
            }

            best.solution.clone()
        }
    };

    if args.verbose {
        if let Some(lam_eff) = solution.lambda_effective {
            eprintln!(
                "  λ_effective = {:.3e}  (λ_user × tr(KᵀK)/tr(LᵀL))",
                lam_eff
            );
        }
        print_summary(&solution);
    }

    // ---- 6. Output -------------------------------------------------------
    match &args.output {
        Some(path) => {
            write_pr_to_file(path, &solution)?;
            if args.verbose {
                eprintln!("P(r) written to '{}'", path.display());
            }
        }
        None => write_pr_to_stdout(&solution)?,
    }

    if let Some(fit_path) = &args.fit_output {
        write_fit_to_file(fit_path, &solution, &data.q, &data.intensity, &data.error)?;
        if args.verbose {
            eprintln!("Fit written to '{}'", fit_path.display());
        }
    }

    Ok(())
}
