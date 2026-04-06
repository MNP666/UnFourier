mod config;

use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use unfourier::{
    basis::{BasisSet, CubicBSpline, UniformGrid},
    data::{parse_dat, SaxsData},
    kernel::build_weighted_system,
    lambda_select::{
        estimate_lambda_range, evaluate_lambda_grid, log_lambda_grid, posterior_sigma,
        BayesianEvidence, GcvSelector, GridMatrices, LCurveSelector, LambdaSelector,
    },
    output::{print_summary, write_fit_to_file, write_pr_to_file, write_pr_to_stdout},
    preprocess::{ClipNegative, LogRebin, OmitNonPositive, Preprocessor, QRangeSelector},
    solver::{Solution, Solver, TikhonovSolver},
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// How to handle non-positive intensities produced by background subtraction.
#[derive(Debug, Clone, ValueEnum, Default)]
enum NegativeHandling {
    /// Replace non-positive I with the minimum positive value; inflate σ so the
    /// point contributes negligibly to the fit (default).
    #[default]
    Clip,
    /// Discard all points where I ≤ 0 entirely.
    Omit,
    /// Leave non-positive intensities unchanged.
    Keep,
}

/// Which basis to use for representing P(r).
#[derive(Debug, Clone, ValueEnum, Default)]
enum BasisChoice {
    /// Piecewise-constant rectangular bins (default, matches M1–M4 behaviour).
    #[default]
    Rect,
    /// Cubic B-spline basis with zero boundary conditions (M5).
    Spline,
}

/// How to choose the regularisation strength λ.
#[derive(Debug, Clone, ValueEnum)]
enum Method {
    /// Minimise the Generalised Cross-Validation score (default).
    Gcv,
    /// Find the corner of the L-curve (log residual vs log solution norm).
    Lcurve,
    /// Maximise the Bayesian log-evidence (BayesApp-style).
    Bayes,
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
    /// 'bayes'  — maximise Bayesian log-evidence; also produces error bars (M4).
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

    /// Basis set for representing P(r).
    /// 'rect'   — piecewise-constant rectangular bins (default).
    /// 'spline' — cubic B-splines with zero boundary conditions (M5).
    #[arg(long, value_enum, default_value_t = BasisChoice::Rect)]
    basis: BasisChoice,

    /// Number of free basis parameters.
    /// For 'rect'   defaults to --npoints (100).
    /// For 'spline' defaults to 20.
    /// Overrides --npoints when provided.
    #[arg(long)]
    n_basis: Option<usize>,

    /// Number of r grid points (rectangular basis only; superseded by --n-basis).
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

    /// How to handle non-positive intensities (from background subtraction).
    /// 'clip' — replace with minimum positive I and inflate σ (default).
    /// 'omit' — discard such points entirely.
    /// 'keep' — leave unchanged (only safe if you are sure all I > 0).
    #[arg(long, value_enum, default_value_t = NegativeHandling::Clip)]
    negative_handling: NegativeHandling,

    /// Discard data points with q below this value (Å⁻¹).
    #[arg(long)]
    qmin: Option<f64>,

    /// Discard data points with q above this value (Å⁻¹).
    #[arg(long)]
    qmax: Option<f64>,

    /// Trim the high-q tail: discard points (from the high-q end inward)
    /// where I/σ < this threshold. 0.0 = disabled (default).
    #[arg(long, default_value_t = 0.0)]
    snr_cutoff: f64,

    /// Rebin data into N logarithmically-spaced q bins before solving.
    /// 0 = disabled (default). Useful for large datasets (e.g. SASDYU3, 1696 pts).
    #[arg(long, default_value_t = 0)]
    rebin: usize,

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
// Solve helper
// ---------------------------------------------------------------------------

/// Build the weighted system from `data` and `basis`, run the configured
/// λ-selection method, and return the resulting `Solution`.
///
/// `verbose` controls whether the λ grid table is printed to stderr.
/// Pass `false` for the secondary (auto-rebin) solve to keep output clean.
fn run_solve(
    data: &SaxsData,
    basis: &dyn BasisSet,
    method: &Method,
    args: &Args,
    verbose: bool,
) -> Result<Solution> {
    let (k_weighted, i_weighted) = build_weighted_system(basis, data);
    let k_unweighted = basis.build_kernel_matrix(&data.q);

    match method {
        Method::Manual => {
            let lambda = args.lambda.unwrap();
            if verbose {
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
            )
        }

        Method::Gcv | Method::Lcurve | Method::Bayes => {
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

            let selector: Box<dyn LambdaSelector> = match method {
                Method::Gcv => Box::new(GcvSelector),
                Method::Lcurve => Box::new(LCurveSelector),
                Method::Bayes => Box::new(BayesianEvidence),
                Method::Manual => unreachable!(),
            };

            if verbose {
                eprintln!(
                    "  method: {}  grid: {} pts in [{:.2e}, {:.2e}]",
                    selector.name(),
                    args.lambda_count,
                    lam_min,
                    lam_max
                );
            }

            let evaluations = evaluate_lambda_grid(&grid, &matrices)?;
            let best_idx = selector.select(&evaluations);
            let best = &evaluations[best_idx];

            if verbose {
                eprintln!(
                    "  selected λ = {:.3e}  (λ_eff = {:.3e})",
                    best.lambda, best.lambda_eff
                );
                let is_bayes = matches!(method, Method::Bayes);
                let score_label = if is_bayes { "log_evid" } else { "GCV" };
                eprintln!(
                    "  {:>8}  {:>12}  {:>12}  {:>10}  {:>10}",
                    "λ", "λ_eff", score_label, "RSS_w", "χ²_red"
                );
                for (i, e) in evaluations.iter().enumerate() {
                    let marker = if i == best_idx { " ←" } else { "" };
                    let score = if is_bayes { e.log_evidence } else { e.gcv };
                    eprintln!(
                        "  {:>8.2e}  {:>12.3e}  {:>12.4e}  {:>10.3e}  {:>10.4}{marker}",
                        e.lambda, e.lambda_eff, score, e.rss_weighted, e.chi_squared
                    );
                }
            }

            let mut solution = best.solution.clone();

            if matches!(method, Method::Bayes) {
                solution.p_r_err = Some(posterior_sigma(&matrices, best.lambda_eff)?);
            }

            Ok(solution)
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut args = Args::parse();

    // ---- 0. Load optional TOML config and fill unset CLI args ---------------
    if let Some(cfg) = config::UnfourierConfig::load()? {
        if args.verbose {
            eprintln!("  [config] loaded from unfourier.toml");
        }

        // [regularisation]
        if args.method.is_none() {
            if let Some(ref m) = cfg.regularisation.method {
                let parsed = match m.as_str() {
                    "gcv"    => Some(Method::Gcv),
                    "lcurve" => Some(Method::Lcurve),
                    "bayes"  => Some(Method::Bayes),
                    "manual" => Some(Method::Manual),
                    other    => return Err(anyhow!("unfourier.toml: unknown method '{}'", other)),
                };
                args.method = parsed;
                if args.verbose {
                    eprintln!("  [config]   method = {}", m);
                }
            }
        }
        if args.lambda_min.is_none() {
            if let Some(v) = cfg.regularisation.lambda_min {
                args.lambda_min = Some(v);
                if args.verbose {
                    eprintln!("  [config]   lambda_min = {:.3e}", v);
                }
            }
        }
        if args.lambda_max.is_none() {
            if let Some(v) = cfg.regularisation.lambda_max {
                args.lambda_max = Some(v);
                if args.verbose {
                    eprintln!("  [config]   lambda_max = {:.3e}", v);
                }
            }
        }

        // [preprocessing]
        if args.qmin.is_none() {
            if let Some(v) = cfg.preprocessing.qmin {
                args.qmin = Some(v);
                if args.verbose {
                    eprintln!("  [config]   qmin = {:.4e}", v);
                }
            }
        }
        if args.qmax.is_none() {
            if let Some(v) = cfg.preprocessing.qmax {
                args.qmax = Some(v);
                if args.verbose {
                    eprintln!("  [config]   qmax = {:.4e}", v);
                }
            }
        }
        // negative_handling has a clap default so we can't distinguish
        // "user set" vs "default" purely from Option; skip override to keep
        // CLI default behaviour unchanged.
        // [basis]
        // npoints also has a clap default; only apply from TOML if n_basis unset.
        if args.n_basis.is_none() {
            if let Some(n) = cfg.basis.npoints {
                args.n_basis = Some(n);
                if args.verbose {
                    eprintln!("  [config]   n_basis = {}", n);
                }
            }
        }
    }

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
    // Build and run each step individually so verbose output can show
    // per-step point counts and q ranges.
    let mut data = raw_data;

    // Step A: handle non-positive intensities.
    let neg_step: Option<Box<dyn unfourier::preprocess::Preprocessor>> =
        match args.negative_handling {
            NegativeHandling::Clip => Some(Box::new(ClipNegative::default())),
            NegativeHandling::Omit => Some(Box::new(OmitNonPositive)),
            NegativeHandling::Keep => None,
        };

    if let Some(step) = neg_step {
        let before = data.len();
        data = step.process(data)?;
        if args.verbose {
            eprintln!(
                "  [{}] {} → {} pts  q=[{:.4e}, {:.4e}]",
                step.name(),
                before,
                data.len(),
                data.q_min(),
                data.q_max()
            );
        }
    }

    // Step B: q-range / SNR filtering.
    let use_qrange = args.qmin.is_some()
        || args.qmax.is_some()
        || args.snr_cutoff > 0.0;
    if use_qrange {
        let step = QRangeSelector {
            q_min: args.qmin,
            q_max: args.qmax,
            snr_threshold: if args.snr_cutoff > 0.0 { Some(args.snr_cutoff) } else { None },
        };
        let before = data.len();
        data = step.process(data)?;
        if args.verbose {
            eprintln!(
                "  [{}] {} → {} pts  q=[{:.4e}, {:.4e}]",
                step.name(),
                before,
                data.len(),
                data.q_min(),
                data.q_max()
            );
        }
    }

    // Step C: log-rebinning.
    if args.rebin > 0 {
        let step = LogRebin { n_bins: args.rebin };
        let before = data.len();
        data = step.process(data)?;
        if args.verbose {
            eprintln!(
                "  [{}] {} → {} pts  q=[{:.4e}, {:.4e}]",
                step.name(),
                before,
                data.len(),
                data.q_min(),
                data.q_max()
            );
        }
    }

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

    let basis: Box<dyn BasisSet> = match args.basis {
        BasisChoice::Rect => {
            let n = args.n_basis.unwrap_or(args.npoints);
            let b = UniformGrid::new(r_max, n);
            if args.verbose {
                eprintln!(
                    "  basis: rect  n={} bins  Δr={:.4} Å",
                    n,
                    b.delta_r()
                );
            }
            Box::new(b)
        }
        BasisChoice::Spline => {
            let n = args.n_basis.unwrap_or(20);
            let b = CubicBSpline::new(r_max, n);
            if args.verbose {
                eprintln!("  basis: spline  n_basis={}", n);
            }
            Box::new(b)
        }
    };

    // ---- 4 + 5. Build system and solve -----------------------------------
    let solution = run_solve(&data, basis.as_ref(), &method, &args, args.verbose)?;

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
