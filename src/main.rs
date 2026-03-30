use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use unfourier::{
    basis::{BasisSet, UniformGrid},
    data::parse_dat,
    kernel::build_weighted_system,
    output::{print_summary, write_pr_to_file, write_pr_to_stdout},
    preprocess::{Identity, PreprocessingPipeline},
    solver::{LeastSquaresSvd, Solver},
};

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

    /// Maximum r value in Å.
    /// Defaults to π / q_min (a rough upper bound on D_max).
    #[arg(long)]
    rmax: Option<f64>,

    /// Number of r grid points.
    #[arg(long, default_value_t = 100)]
    npoints: usize,

    /// Write P(r) to this file. Defaults to stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Print diagnostic summary (χ², D_max estimate, peak position) to stderr.
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

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

    // ---- 2. Preprocess (identity for M1) ---------------------------------
    //
    // Swap in real preprocessors here in M6:
    //   .add(Box::new(QRangeSelector::auto()))
    //   .add(Box::new(LogRebin::new(200)))
    //   .add(Box::new(ClipNegative))
    let pipeline = PreprocessingPipeline::new().add(Box::new(Identity));
    let data = pipeline.run(raw_data);

    // ---- 3. Set up r grid ------------------------------------------------
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

    // ---- 4. Build weighted system ----------------------------------------
    let (k_weighted, i_weighted) = build_weighted_system(&basis, &data);

    // Also keep the unweighted kernel for back-calculation and chi-squared
    let k_unweighted = basis.build_kernel_matrix(&data.q);

    // ---- 5. Solve --------------------------------------------------------
    //
    // Swap in a regularising solver in M2:
    //   let solver = TikhonovSvd::new(lambda);
    let solver = LeastSquaresSvd::new();
    let solution = solver.solve(
        &k_weighted,
        &i_weighted,
        &k_unweighted,
        &data.intensity,
        &data.error,
        basis.r_values(),
    )?;

    if args.verbose {
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

    Ok(())
}
