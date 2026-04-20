mod config;

use anyhow::{Result, anyhow};
use clap::{CommandFactory, FromArgMatches, Parser, ValueEnum, parser::ValueSource};
use std::path::PathBuf;

use unfourier::{
    basis::{BasisSet, CubicBSpline, SplineBoundaryMode},
    data::{SaxsData, parse_dat},
    guinier::{GuinierScanConfig, GuinierScanReport, GuinierWindowFit, scan_guinier},
    kernel::build_weighted_system,
    lambda_select::{
        BayesianEvidence, GcvSelector, GridMatrices, LCurveSelector, LambdaSelector,
        estimate_lambda_range, evaluate_lambda_grid, log_lambda_grid, posterior_coeff_sigma,
    },
    output::{PrCurve, print_summary, write_fit_to_file, write_pr_to_file, write_pr_to_stdout},
    preprocess::{ClipNegative, LogRebin, OmitNonPositive, Preprocessor, QRangeSelector},
    regularise::{ProjectedSplineRegulariser, Regulariser},
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

fn parse_negative_handling_config(value: &str) -> Result<NegativeHandling> {
    NegativeHandling::from_str(value, true).map_err(|_| {
        anyhow!(
            "unfourier.toml: unknown preprocessing.negative_handling '{}'; expected one of: clip, omit, keep",
            value
        )
    })
}

/// Whether low-q preprocessing should be mutated automatically.
#[derive(Debug, Clone, ValueEnum, Default, PartialEq)]
enum AutoQmin {
    /// Do not apply an automatic low-q cutoff.
    #[default]
    Off,
    /// Use the Guinier preflight recommendation as qmin when qmin is unset.
    Guinier,
}

fn parse_auto_qmin_config(value: &str) -> Result<AutoQmin> {
    AutoQmin::from_str(value, true).map_err(|_| {
        anyhow!(
            "unfourier.toml: unknown preprocessing.auto_qmin '{}'; expected one of: off, guinier",
            value
        )
    })
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

const DEFAULT_N_BASIS: usize = 20;
const DEFAULT_MIN_BASIS: usize = 12;
const DEFAULT_MAX_BASIS: usize = 48;
const MIN_N_BASIS: usize = 2;
const DEFAULT_D1_SMOOTHNESS: f64 = 0.1;
const DEFAULT_D2_SMOOTHNESS: f64 = 1.0;

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

    /// Number of free cubic B-spline basis parameters.
    /// Defaults to 20.
    #[arg(long)]
    n_basis: Option<usize>,

    /// Target real-space spacing per free cubic B-spline parameter in Å.
    /// Ignored when --n-basis is set.
    #[arg(long)]
    knot_spacing: Option<f64>,

    /// Minimum n_basis when deriving it from --knot-spacing.
    #[arg(long)]
    min_basis: Option<usize>,

    /// Maximum n_basis when deriving it from --knot-spacing.
    #[arg(long)]
    max_basis: Option<usize>,

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

    /// Print a Guinier low-q preflight report without changing the fit.
    #[arg(long)]
    guinier_report: bool,

    /// Automatically choose qmin. 'off' leaves qmin unchanged; 'guinier'
    /// applies the Guinier recommendation only when qmin is not already set.
    #[arg(long, value_enum, default_value_t = AutoQmin::Off)]
    auto_qmin: AutoQmin,

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
// Basis-size resolution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum BasisCountSource {
    Explicit,
    KnotSpacing {
        knot_spacing: f64,
        min_basis: usize,
        max_basis: usize,
        unclamped: usize,
    },
    Default,
}

#[derive(Debug, Clone, PartialEq)]
struct BasisCount {
    n_basis: usize,
    source: BasisCountSource,
}

fn resolve_basis_count(args: &Args, r_max: f64) -> Result<BasisCount> {
    validate_basis_options(args)?;

    if !r_max.is_finite() || r_max <= 0.0 {
        return Err(anyhow!("r_max must be positive and finite, got {}", r_max));
    }

    if let Some(n_basis) = args.n_basis {
        return Ok(BasisCount {
            n_basis,
            source: BasisCountSource::Explicit,
        });
    }

    if let Some(knot_spacing) = args.knot_spacing {
        let min_basis = args.min_basis.unwrap_or(DEFAULT_MIN_BASIS);
        let max_basis = args.max_basis.unwrap_or(DEFAULT_MAX_BASIS);
        let raw = (r_max / knot_spacing).ceil();
        if raw > usize::MAX as f64 {
            return Err(anyhow!(
                "derived n_basis from r_max/knot_spacing is too large: ceil({:.6e} / {:.6e})",
                r_max,
                knot_spacing
            ));
        }

        let unclamped = raw as usize;
        let n_basis = unclamped.clamp(min_basis, max_basis);
        return Ok(BasisCount {
            n_basis,
            source: BasisCountSource::KnotSpacing {
                knot_spacing,
                min_basis,
                max_basis,
                unclamped,
            },
        });
    }

    Ok(BasisCount {
        n_basis: DEFAULT_N_BASIS,
        source: BasisCountSource::Default,
    })
}

fn validate_basis_options(args: &Args) -> Result<()> {
    if let Some(n_basis) = args.n_basis {
        if n_basis < MIN_N_BASIS {
            return Err(anyhow!(
                "--n-basis must be at least {}, got {}",
                MIN_N_BASIS,
                n_basis
            ));
        }
    }

    if let Some(knot_spacing) = args.knot_spacing {
        if !knot_spacing.is_finite() || knot_spacing <= 0.0 {
            return Err(anyhow!(
                "--knot-spacing must be positive and finite, got {}",
                knot_spacing
            ));
        }
    }

    if args.knot_spacing.is_some() || args.min_basis.is_some() || args.max_basis.is_some() {
        let min_basis = args.min_basis.unwrap_or(DEFAULT_MIN_BASIS);
        let max_basis = args.max_basis.unwrap_or(DEFAULT_MAX_BASIS);

        if min_basis < MIN_N_BASIS {
            return Err(anyhow!(
                "--min-basis must be at least {}, got {}",
                MIN_N_BASIS,
                min_basis
            ));
        }
        if max_basis < MIN_N_BASIS {
            return Err(anyhow!(
                "--max-basis must be at least {}, got {}",
                MIN_N_BASIS,
                max_basis
            ));
        }
        if min_basis > max_basis {
            return Err(anyhow!(
                "--min-basis ({}) must be <= --max-basis ({})",
                min_basis,
                max_basis
            ));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Smoothness resolution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
struct SmoothnessWeights {
    d1_weight: f64,
    d2_weight: f64,
}

fn resolve_smoothness_weights(
    d1_smoothness: Option<f64>,
    d2_smoothness: Option<f64>,
) -> Result<SmoothnessWeights> {
    let d1_weight = match d1_smoothness {
        None => DEFAULT_D1_SMOOTHNESS,
        Some(v) if v == -1.0 => 0.0,
        Some(v) if v == 0.0 => DEFAULT_D1_SMOOTHNESS,
        Some(v) if v.is_finite() && v > 0.0 => v,
        Some(v) => {
            return Err(anyhow!(
                "unfourier.toml: constraints.d1_smoothness must be -1, 0, or a positive finite value, got {}",
                v
            ));
        }
    };

    let d2_weight = match d2_smoothness {
        None => DEFAULT_D2_SMOOTHNESS,
        Some(v) if v.is_finite() && v >= 0.0 => v,
        Some(v) => {
            return Err(anyhow!(
                "unfourier.toml: constraints.d2_smoothness must be a non-negative finite value, got {}",
                v
            ));
        }
    };

    if d1_weight == 0.0 && d2_weight == 0.0 {
        return Err(anyhow!(
            "unfourier.toml: at least one of d1_smoothness or d2_smoothness must be active"
        ));
    }

    Ok(SmoothnessWeights {
        d1_weight,
        d2_weight,
    })
}

// ---------------------------------------------------------------------------
// Guinier preflight
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum GuinierPreflightAction {
    ReportOnly,
    Applied {
        previous_qmin: f64,
        applied_qmin: f64,
    },
    ExplicitQmin {
        qmin: f64,
    },
    NoRecommendation,
}

#[derive(Debug, Clone, PartialEq)]
struct GuinierPreflightResult {
    report: GuinierScanReport,
    action: GuinierPreflightAction,
}

fn run_guinier_preflight(
    data: &SaxsData,
    args: &mut Args,
    config: &GuinierScanConfig,
    qmin_already_set: bool,
) -> Option<GuinierPreflightResult> {
    let auto_requested = matches!(args.auto_qmin, AutoQmin::Guinier);
    if !args.guinier_report && !auto_requested {
        return None;
    }

    let report = scan_guinier(data, config);
    let action = match args.auto_qmin {
        AutoQmin::Off => GuinierPreflightAction::ReportOnly,
        AutoQmin::Guinier if qmin_already_set => GuinierPreflightAction::ExplicitQmin {
            qmin: args
                .qmin
                .expect("qmin_already_set implies args.qmin is Some"),
        },
        AutoQmin::Guinier => {
            if let Some(rec) = &report.recommendation {
                let previous_qmin = data.q_min();
                args.qmin = Some(rec.q_min);
                GuinierPreflightAction::Applied {
                    previous_qmin,
                    applied_qmin: rec.q_min,
                }
            } else {
                GuinierPreflightAction::NoRecommendation
            }
        }
    };

    Some(GuinierPreflightResult { report, action })
}

fn representative_guinier_fits(report: &GuinierScanReport) -> Vec<&GuinierWindowFit> {
    let mut rows = Vec::new();
    let mut start = 0usize;

    while start < report.candidate_fits.len() {
        let skip = report.candidate_fits[start].skip;
        let mut end = start + 1;
        while end < report.candidate_fits.len() && report.candidate_fits[end].skip == skip {
            end += 1;
        }

        let group = &report.candidate_fits[start..end];
        if let Some(fit) = group
            .iter()
            .filter(|fit| fit.valid)
            .max_by_key(|fit| fit.n_points)
            .or_else(|| group.iter().max_by_key(|fit| fit.n_points))
        {
            rows.push(fit);
        }

        start = end;
    }

    rows
}

fn format_optional(value: Option<f64>, width: usize, precision: usize) -> String {
    match value {
        Some(v) => format!("{v:>width$.precision$}"),
        None => format!("{:>width$}", "--"),
    }
}

fn format_guinier_report(preflight: &GuinierPreflightResult) -> String {
    let mut out = String::from("Guinier scan:\n");
    out.push_str("skip  qmin          n   qmax*Rg       Rg          I0       chi2   status\n");

    let rec_skip = preflight.report.recommendation.as_ref().map(|rec| rec.skip);
    for fit in representative_guinier_fits(&preflight.report) {
        let status = if rec_skip == Some(fit.skip) {
            "stable".to_string()
        } else if fit.valid {
            "valid".to_string()
        } else {
            match fit.reject_reason {
                Some(reason) => format!("reject: {reason}"),
                None => "reject".to_string(),
            }
        };

        out.push_str(&format!(
            "{:<4}  {}  {:>3}  {}  {}  {}  {}   {}\n",
            fit.skip,
            format_optional(fit.q_min, 10, 4),
            fit.n_points,
            format_optional(fit.qrg_max, 8, 3),
            format_optional(fit.rg, 8, 3),
            format_optional(fit.i0, 10, 3),
            format_optional(fit.chi2_red, 8, 3),
            status
        ));
    }

    out.push('\n');
    if let Some(rec) = &preflight.report.recommendation {
        out.push_str(&format!(
            "Guinier recommendation: qmin = {:.4e} (skip {})",
            rec.q_min, rec.skip
        ));
    } else {
        out.push_str("Guinier recommendation: none");
    }

    match preflight.action {
        GuinierPreflightAction::ReportOnly => out.push_str(", report-only"),
        GuinierPreflightAction::Applied {
            previous_qmin,
            applied_qmin,
        } => out.push_str(&format!(
            ", applied: low-q edge {:.4e} -> {:.4e}",
            previous_qmin, applied_qmin
        )),
        GuinierPreflightAction::ExplicitQmin { qmin } => {
            out.push_str(&format!(", not applied: qmin already set to {:.4e}", qmin))
        }
        GuinierPreflightAction::NoRecommendation => {
            out.push_str(", auto-qmin requested; qmin left unset")
        }
    }

    out
}

fn format_guinier_note(preflight: &GuinierPreflightResult) -> Option<String> {
    match preflight.action {
        GuinierPreflightAction::ReportOnly => None,
        GuinierPreflightAction::Applied {
            previous_qmin,
            applied_qmin,
        } => {
            let skip = preflight
                .report
                .recommendation
                .as_ref()
                .map(|rec| rec.skip)
                .unwrap_or(0);
            Some(format!(
                "guinier: auto-qmin applied qmin = {:.4e} (skip {}); low-q edge {:.4e} -> {:.4e}",
                applied_qmin, skip, previous_qmin, applied_qmin
            ))
        }
        GuinierPreflightAction::ExplicitQmin { qmin } => Some(format!(
            "guinier: auto-qmin requested but qmin = {:.4e} is already set; recommendation not applied",
            qmin
        )),
        GuinierPreflightAction::NoRecommendation => Some(
            "guinier: auto-qmin requested but no stable recommendation; qmin left unset"
                .to_string(),
        ),
    }
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
    ltl: nalgebra::DMatrix<f64>,
    reg: Box<dyn Regulariser>,
) -> Result<Solution> {
    let (k_weighted, i_weighted) = build_weighted_system(basis, data);
    let k_unweighted = basis.build_kernel_matrix(&data.q);

    match method {
        Method::Manual => {
            let lambda = args.lambda.unwrap();
            if verbose {
                eprintln!("  method: manual  λ = {:.3e}", lambda);
            }
            let mut solver = TikhonovSolver::new(lambda);
            solver.regulariser = reg;
            solver.solve(
                &k_weighted,
                &i_weighted,
                &k_unweighted,
                &data.intensity,
                &data.error,
            )
        }

        Method::Gcv | Method::Lcurve | Method::Bayes => {
            let matrices = GridMatrices::build(
                &k_weighted,
                &i_weighted,
                &k_unweighted,
                &data.intensity,
                &data.error,
                ltl,
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
                solution.coeff_err = Some(posterior_coeff_sigma(&matrices, best.lambda_eff)?);
            }

            Ok(solution)
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let matches = Args::command().get_matches();
    let cli_negative_handling =
        matches.value_source("negative_handling") == Some(ValueSource::CommandLine);
    let cli_auto_qmin = matches.value_source("auto_qmin") == Some(ValueSource::CommandLine);
    let cli_guinier_report =
        matches.value_source("guinier_report") == Some(ValueSource::CommandLine);
    let mut args = Args::from_arg_matches(&matches)?;
    let cli_n_basis = args.n_basis.is_some();
    let cli_knot_spacing = args.knot_spacing.is_some();

    // ---- 0. Load optional TOML config and fill unset CLI args ---------------
    // Smoothness weights for the projected spline regulariser.
    // D1: absent or 0 = default, -1 = disabled, >0 = explicit.
    // D2: absent = default, >=0 = explicit.
    let mut d1_smoothness: Option<f64> = None;
    let mut d2_smoothness: Option<f64> = None;
    let mut spline_boundary = SplineBoundaryMode::default();
    let mut guinier_config = GuinierScanConfig::default();

    if let Some(cfg) = config::UnfourierConfig::load()? {
        if args.verbose {
            eprintln!("  [config] loaded from unfourier.toml");
        }

        // [regularisation]
        if args.method.is_none() {
            if let Some(ref m) = cfg.regularisation.method {
                let parsed = match m.as_str() {
                    "gcv" => Some(Method::Gcv),
                    "lcurve" => Some(Method::Lcurve),
                    "bayes" => Some(Method::Bayes),
                    "manual" => Some(Method::Manual),
                    other => return Err(anyhow!("unfourier.toml: unknown method '{}'", other)),
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
        if !cli_negative_handling {
            if let Some(ref v) = cfg.preprocessing.negative_handling {
                args.negative_handling = parse_negative_handling_config(v)?;
                if args.verbose {
                    eprintln!("  [config]   negative_handling = {}", v);
                }
            }
        }
        if !cli_auto_qmin {
            if let Some(ref v) = cfg.preprocessing.auto_qmin {
                args.auto_qmin = parse_auto_qmin_config(v)?;
                if args.verbose {
                    eprintln!("  [config]   auto_qmin = {}", v);
                }
            }
        }

        // [guinier]
        if !cli_guinier_report {
            if let Some(v) = cfg.guinier.report {
                args.guinier_report = v;
                if args.verbose {
                    eprintln!("  [config]   guinier.report = {}", v);
                }
            }
        }
        if let Some(v) = cfg.guinier.min_points {
            guinier_config.min_points = v;
        }
        if let Some(v) = cfg.guinier.max_points {
            guinier_config.max_points = v;
        }
        if let Some(v) = cfg.guinier.max_skip {
            guinier_config.max_skip = v;
        }
        if let Some(v) = cfg.guinier.max_qrg {
            guinier_config.max_qrg = v;
        }
        if let Some(v) = cfg.guinier.stability_windows {
            guinier_config.stability_windows = v;
        }
        if let Some(v) = cfg.guinier.rg_tolerance {
            guinier_config.rg_tolerance = v;
        }
        if let Some(v) = cfg.guinier.i0_tolerance {
            guinier_config.i0_tolerance = v;
        }
        if let Some(v) = cfg.guinier.max_chi2 {
            guinier_config.max_chi2 = v;
        }
        if args.verbose
            && (cfg.guinier.min_points.is_some()
                || cfg.guinier.max_points.is_some()
                || cfg.guinier.max_skip.is_some()
                || cfg.guinier.max_qrg.is_some()
                || cfg.guinier.stability_windows.is_some()
                || cfg.guinier.rg_tolerance.is_some()
                || cfg.guinier.i0_tolerance.is_some()
                || cfg.guinier.max_chi2.is_some())
        {
            eprintln!("  [config]   guinier scan parameters loaded");
        }

        // [basis]
        if !cli_n_basis && !cli_knot_spacing && args.n_basis.is_none() {
            if let Some(n) = cfg.basis.n_basis {
                args.n_basis = Some(n);
                if args.verbose {
                    eprintln!("  [config]   n_basis = {}", n);
                }
            }
        }
        if !cli_n_basis && args.n_basis.is_none() && args.knot_spacing.is_none() {
            if let Some(v) = cfg.basis.knot_spacing {
                args.knot_spacing = Some(v);
                if args.verbose {
                    eprintln!("  [config]   knot_spacing = {:.4}", v);
                }
            }
        }
        if args.knot_spacing.is_some() && args.min_basis.is_none() {
            if let Some(n) = cfg.basis.min_basis {
                args.min_basis = Some(n);
                if args.verbose {
                    eprintln!("  [config]   min_basis = {}", n);
                }
            }
        }
        if args.knot_spacing.is_some() && args.max_basis.is_none() {
            if let Some(n) = cfg.basis.max_basis {
                args.max_basis = Some(n);
                if args.verbose {
                    eprintln!("  [config]   max_basis = {}", n);
                }
            }
        }

        // [constraints]
        if let Some(mode) = cfg.constraints.spline_boundary {
            spline_boundary = mode;
            if args.verbose {
                eprintln!("  [config]   spline_boundary = {}", mode);
            }
        }
        if let Some(d1) = cfg.constraints.d1_smoothness {
            d1_smoothness = Some(d1);
            if args.verbose {
                eprintln!("  [config]   d1_smoothness = {}", d1);
            }
        }
        if let Some(d2) = cfg.constraints.d2_smoothness {
            d2_smoothness = Some(d2);
            if args.verbose {
                eprintln!("  [config]   d2_smoothness = {}", d2);
            }
        }
    }

    let smoothness = resolve_smoothness_weights(d1_smoothness, d2_smoothness)?;
    let qmin_already_set = args.qmin.is_some();

    // Resolve the effective method: explicit --method, or infer from --lambda.
    let method = match (&args.method, args.lambda) {
        (Some(m), _) => m.clone(),
        (None, Some(_)) => Method::Manual,
        (None, None) => Method::Gcv, // default
    };

    if matches!(method, Method::Manual) && args.lambda.is_none() {
        return Err(anyhow!("--method manual requires --lambda to be specified"));
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

    // Step B: optional Guinier low-q preflight.
    if let Some(preflight) =
        run_guinier_preflight(&data, &mut args, &guinier_config, qmin_already_set)
    {
        let print_report =
            args.guinier_report || (args.verbose && matches!(args.auto_qmin, AutoQmin::Guinier));
        if print_report {
            eprintln!("{}", format_guinier_report(&preflight));
        } else if let Some(note) = format_guinier_note(&preflight) {
            eprintln!("{note}");
        }
    }

    // Step C: q-range / SNR filtering.
    let use_qrange = args.qmin.is_some() || args.qmax.is_some() || args.snr_cutoff > 0.0;
    if use_qrange {
        let step = QRangeSelector {
            q_min: args.qmin,
            q_max: args.qmax,
            snr_threshold: if args.snr_cutoff > 0.0 {
                Some(args.snr_cutoff)
            } else {
                None
            },
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

    // Step D: log-rebinning.
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

    let basis_count = resolve_basis_count(&args, r_max)?;
    let basis = CubicBSpline::with_boundary_mode(r_max, basis_count.n_basis, spline_boundary);
    if args.verbose {
        match basis_count.source {
            BasisCountSource::Explicit => {
                if args.knot_spacing.is_some() {
                    eprintln!(
                        "  basis: cubic-b-spline  n_basis={}  boundary={}  (explicit; knot_spacing ignored)",
                        basis_count.n_basis,
                        basis.boundary_mode()
                    );
                } else {
                    eprintln!(
                        "  basis: cubic-b-spline  n_basis={}  boundary={}  (explicit)",
                        basis_count.n_basis,
                        basis.boundary_mode()
                    );
                }
            }
            BasisCountSource::KnotSpacing {
                knot_spacing,
                min_basis,
                max_basis,
                unclamped,
            } => {
                eprintln!(
                    "  basis: cubic-b-spline  n_basis={}  boundary={}  (derived: ceil({:.2}/{:.4})={} clamped to [{}, {}])",
                    basis_count.n_basis,
                    basis.boundary_mode(),
                    r_max,
                    knot_spacing,
                    unclamped,
                    min_basis,
                    max_basis
                );
            }
            BasisCountSource::Default => {
                eprintln!(
                    "  basis: cubic-b-spline  n_basis={}  boundary={}  (default)",
                    basis_count.n_basis,
                    basis.boundary_mode()
                );
            }
        }
    }

    // ---- 4 + 5. Build system and solve -----------------------------------

    // Build the spline regulariser once. It starts in full clamped coefficient
    // space, then projects through the same free-to-full coefficient map as the
    // kernel and output evaluator.
    let n_basis = basis.r_values().len();
    if args.verbose {
        eprintln!(
            "  [constraints] regulariser: projected-spline-derivative  boundary={}  d1={:.2}  d2={:.2}",
            spline_boundary, smoothness.d1_weight, smoothness.d2_weight
        );
    }
    let reg: Box<dyn Regulariser> = Box::new(ProjectedSplineRegulariser {
        boundary_mode: spline_boundary,
        d1_weight: smoothness.d1_weight,
        d2_weight: smoothness.d2_weight,
    });
    let ltl = reg.gram_matrix(n_basis);

    let solution = run_solve(&data, &basis, &method, &args, args.verbose, ltl, reg)?;

    // ---- Evaluate spline output ------------------------------------------
    // The solver returns spline coefficients.  The published P(r) table is the
    // evaluated spline function on a dense output grid, including the endpoints.
    let output_r = basis.output_grid();
    let output_p = basis.evaluate_pr(&solution.coeffs, &output_r);
    let output_err = solution
        .coeff_err
        .as_ref()
        .map(|coeff_sigma| basis.evaluate_pr_sigma(coeff_sigma, &output_r));
    let pr_curve = PrCurve {
        r: output_r,
        p_r: output_p,
        p_r_err: output_err,
    };

    if args.verbose {
        if let Some(lam_eff) = solution.lambda_effective {
            eprintln!(
                "  λ_effective = {:.3e}  (λ_user × tr(KᵀK)/tr(LᵀL))",
                lam_eff
            );
        }
        print_summary(&solution, &pr_curve);
    }

    // ---- 6. Output -------------------------------------------------------
    match &args.output {
        Some(path) => {
            write_pr_to_file(path, &pr_curve)?;
            if args.verbose {
                eprintln!("P(r) written to '{}'", path.display());
            }
        }
        None => write_pr_to_stdout(&pr_curve)?,
    }

    if let Some(fit_path) = &args.fit_output {
        write_fit_to_file(fit_path, &solution, &data.q, &data.intensity, &data.error)?;
        if args.verbose {
            eprintln!("Fit written to '{}'", fit_path.display());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_for_basis() -> Args {
        Args {
            input: PathBuf::from("data.dat"),
            method: None,
            lambda: None,
            rmax: None,
            n_basis: None,
            knot_spacing: None,
            min_basis: None,
            max_basis: None,
            lambda_count: 60,
            lambda_min: None,
            lambda_max: None,
            negative_handling: NegativeHandling::Clip,
            qmin: None,
            qmax: None,
            snr_cutoff: 0.0,
            rebin: 0,
            guinier_report: false,
            auto_qmin: AutoQmin::Off,
            output: None,
            fit_output: None,
            verbose: false,
        }
    }

    #[test]
    fn basis_count_defaults_to_20() {
        let args = args_for_basis();
        let resolved = resolve_basis_count(&args, 150.0).unwrap();
        assert_eq!(
            resolved,
            BasisCount {
                n_basis: DEFAULT_N_BASIS,
                source: BasisCountSource::Default
            }
        );
    }

    #[test]
    fn negative_handling_config_parses_known_values() {
        assert!(matches!(
            parse_negative_handling_config("clip").unwrap(),
            NegativeHandling::Clip
        ));
        assert!(matches!(
            parse_negative_handling_config("omit").unwrap(),
            NegativeHandling::Omit
        ));
        assert!(matches!(
            parse_negative_handling_config("keep").unwrap(),
            NegativeHandling::Keep
        ));
        assert!(matches!(
            parse_negative_handling_config("OMIT").unwrap(),
            NegativeHandling::Omit
        ));
    }

    #[test]
    fn negative_handling_config_rejects_unknown_values() {
        let err = parse_negative_handling_config("nonnegative-pr").unwrap_err();
        assert!(
            err.to_string()
                .contains("unknown preprocessing.negative_handling"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn auto_qmin_config_parses_known_values() {
        assert_eq!(parse_auto_qmin_config("off").unwrap(), AutoQmin::Off);
        assert_eq!(
            parse_auto_qmin_config("guinier").unwrap(),
            AutoQmin::Guinier
        );
        assert_eq!(
            parse_auto_qmin_config("GUINIER").unwrap(),
            AutoQmin::Guinier
        );
    }

    #[test]
    fn auto_qmin_config_rejects_unknown_values() {
        let err = parse_auto_qmin_config("magic").unwrap_err();
        assert!(
            err.to_string().contains("unknown preprocessing.auto_qmin"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn invalid_cli_auto_qmin_fails_through_clap() {
        let result = Args::try_parse_from(["unfourier", "data.dat", "--auto-qmin", "magic"]);
        assert!(result.is_err());
    }

    fn synthetic_guinier_data() -> SaxsData {
        let rg = 35.0_f64;
        let i0 = 100.0_f64;
        let q: Vec<f64> = (0..32).map(|idx| 0.002 + 0.001 * idx as f64).collect();
        let intensity: Vec<f64> = q
            .iter()
            .map(|&qv| i0 * (-(rg * rg * qv * qv) / 3.0).exp())
            .collect();
        let error: Vec<f64> = intensity.iter().map(|&iv| 0.02 * iv).collect();
        SaxsData::new(q, intensity, error).unwrap()
    }

    #[test]
    fn guinier_report_only_leaves_qmin_unset() {
        let data = synthetic_guinier_data();
        let mut args = args_for_basis();
        args.guinier_report = true;
        let config = GuinierScanConfig {
            min_points: 8,
            max_points: 12,
            max_skip: 4,
            stability_windows: 3,
            ..Default::default()
        };

        let preflight = run_guinier_preflight(&data, &mut args, &config, false)
            .expect("report-only preflight should run");

        assert_eq!(args.qmin, None);
        assert!(matches!(
            preflight.action,
            GuinierPreflightAction::ReportOnly
        ));
    }

    #[test]
    fn guinier_auto_qmin_applies_recommendation_when_qmin_unset() {
        let data = synthetic_guinier_data();
        let mut args = args_for_basis();
        args.auto_qmin = AutoQmin::Guinier;
        let config = GuinierScanConfig {
            min_points: 8,
            max_points: 12,
            max_skip: 4,
            stability_windows: 3,
            ..Default::default()
        };

        let preflight = run_guinier_preflight(&data, &mut args, &config, false)
            .expect("auto-qmin preflight should run");

        assert_eq!(args.qmin, Some(data.q[0]));
        assert!(matches!(
            preflight.action,
            GuinierPreflightAction::Applied { .. }
        ));
    }

    #[test]
    fn guinier_auto_qmin_respects_existing_qmin() {
        let data = synthetic_guinier_data();
        let mut args = args_for_basis();
        args.auto_qmin = AutoQmin::Guinier;
        args.qmin = Some(0.006);
        let config = GuinierScanConfig {
            min_points: 8,
            max_points: 12,
            max_skip: 4,
            stability_windows: 3,
            ..Default::default()
        };

        let preflight = run_guinier_preflight(&data, &mut args, &config, true)
            .expect("auto-qmin preflight should run");

        assert_eq!(args.qmin, Some(0.006));
        assert!(matches!(
            preflight.action,
            GuinierPreflightAction::ExplicitQmin { qmin }
                if (qmin - 0.006).abs() < 1e-12
        ));
    }

    #[test]
    fn guinier_report_mentions_explicit_qmin_precedence() {
        let data = synthetic_guinier_data();
        let mut args = args_for_basis();
        args.auto_qmin = AutoQmin::Guinier;
        args.qmin = Some(0.006);
        let config = GuinierScanConfig {
            min_points: 8,
            max_points: 12,
            max_skip: 4,
            stability_windows: 3,
            ..Default::default()
        };
        let preflight = run_guinier_preflight(&data, &mut args, &config, true).unwrap();

        let report = format_guinier_report(&preflight);

        assert!(report.contains("Guinier scan:"));
        assert!(report.contains("not applied: qmin already set"));
    }

    #[test]
    fn explicit_n_basis_wins_over_knot_spacing() {
        let mut args = args_for_basis();
        args.n_basis = Some(24);
        args.knot_spacing = Some(7.5);
        args.min_basis = Some(12);
        args.max_basis = Some(48);

        let resolved = resolve_basis_count(&args, 150.0).unwrap();
        assert_eq!(
            resolved,
            BasisCount {
                n_basis: 24,
                source: BasisCountSource::Explicit
            }
        );
    }

    #[test]
    fn knot_spacing_derives_basis_count_from_dmax() {
        let mut args = args_for_basis();
        args.knot_spacing = Some(7.5);
        args.min_basis = Some(12);
        args.max_basis = Some(48);

        let resolved = resolve_basis_count(&args, 150.0).unwrap();
        assert_eq!(
            resolved,
            BasisCount {
                n_basis: 20,
                source: BasisCountSource::KnotSpacing {
                    knot_spacing: 7.5,
                    min_basis: 12,
                    max_basis: 48,
                    unclamped: 20
                }
            }
        );
    }

    #[test]
    fn knot_spacing_clamps_to_min_and_max() {
        let mut args = args_for_basis();
        args.knot_spacing = Some(10.0);
        args.min_basis = Some(12);
        args.max_basis = Some(48);

        let min_clamped = resolve_basis_count(&args, 50.0).unwrap();
        assert_eq!(min_clamped.n_basis, 12);

        args.knot_spacing = Some(5.0);
        let max_clamped = resolve_basis_count(&args, 400.0).unwrap();
        assert_eq!(max_clamped.n_basis, 48);
    }

    #[test]
    fn basis_resolution_rejects_invalid_values() {
        let mut args = args_for_basis();
        args.n_basis = Some(1);
        assert!(resolve_basis_count(&args, 150.0).is_err());

        args = args_for_basis();
        args.knot_spacing = Some(0.0);
        assert!(resolve_basis_count(&args, 150.0).is_err());

        args = args_for_basis();
        args.knot_spacing = Some(7.5);
        args.min_basis = Some(50);
        args.max_basis = Some(12);
        assert!(resolve_basis_count(&args, 150.0).is_err());
    }

    #[test]
    fn smoothness_defaults_enable_d1_and_d2() {
        let weights = resolve_smoothness_weights(None, None).unwrap();
        assert_eq!(
            weights,
            SmoothnessWeights {
                d1_weight: DEFAULT_D1_SMOOTHNESS,
                d2_weight: DEFAULT_D2_SMOOTHNESS,
            }
        );
    }

    #[test]
    fn smoothness_d1_minus_one_disables_only_d1() {
        let weights = resolve_smoothness_weights(Some(-1.0), None).unwrap();
        assert_eq!(
            weights,
            SmoothnessWeights {
                d1_weight: 0.0,
                d2_weight: DEFAULT_D2_SMOOTHNESS,
            }
        );
    }

    #[test]
    fn smoothness_d1_zero_uses_default_weight() {
        let weights = resolve_smoothness_weights(Some(0.0), Some(2.0)).unwrap();
        assert_eq!(
            weights,
            SmoothnessWeights {
                d1_weight: DEFAULT_D1_SMOOTHNESS,
                d2_weight: 2.0,
            }
        );
    }

    #[test]
    fn smoothness_positive_values_are_explicit_weights() {
        let weights = resolve_smoothness_weights(Some(0.25), Some(0.75)).unwrap();
        assert_eq!(
            weights,
            SmoothnessWeights {
                d1_weight: 0.25,
                d2_weight: 0.75,
            }
        );
    }

    #[test]
    fn smoothness_rejects_invalid_values() {
        assert!(resolve_smoothness_weights(Some(-0.5), None).is_err());
        assert!(resolve_smoothness_weights(Some(f64::NAN), None).is_err());
        assert!(resolve_smoothness_weights(None, Some(-1.0)).is_err());
        assert!(resolve_smoothness_weights(None, Some(f64::INFINITY)).is_err());
        assert!(resolve_smoothness_weights(Some(-1.0), Some(0.0)).is_err());
    }
}
