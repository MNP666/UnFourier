use crate::data::SaxsData;

/// A single data-preprocessing step.
///
/// Preprocessors take ownership of a `SaxsData`, transform it, and return the
/// result. They are designed to be pure (no side effects) and stateless so that
/// they can be safely used across threads.
///
/// # Composing preprocessors
///
/// Use [`PreprocessingPipeline`] to chain multiple steps in sequence.
///
/// # Future implementations
///
/// Milestones beyond M1 will add concrete implementations for:
/// - `LogRebin` / `LinearRebin` — reduce data density at high q
/// - `ClipNegative` — handle negative intensities
/// - `QRangeSelector` — Guinier-based q_min and SNR-based q_max cutoffs
/// - `ErrorEstimator` — auto-estimate σ when the input file lacks error values
pub trait Preprocessor: Send + Sync {
    /// Human-readable name used in verbose output.
    fn name(&self) -> &str;

    /// Apply this preprocessing step and return the transformed data.
    fn process(&self, data: SaxsData) -> SaxsData;
}

// ---------------------------------------------------------------------------
// Identity (no-op) — the only implementation needed for M1
// ---------------------------------------------------------------------------

/// Pass-through preprocessor: returns the data unchanged.
///
/// This is the default for M1. It satisfies the `Preprocessor` trait so that
/// the pipeline infrastructure is in place for future steps.
pub struct Identity;

impl Preprocessor for Identity {
    fn name(&self) -> &str {
        "identity"
    }

    fn process(&self, data: SaxsData) -> SaxsData {
        data
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Chains multiple preprocessors in sequence, passing the output of each step
/// as the input to the next.
///
/// ```rust,ignore
/// let pipeline = PreprocessingPipeline::new()
///     .add(Box::new(Identity));  // M1: no-op
///     // .add(Box::new(LogRebin::new(200)))  // future: rebin to 200 points
///     // .add(Box::new(ClipNegative))         // future: remove negative I values
///
/// let processed = pipeline.run(raw_data);
/// ```
#[derive(Default)]
pub struct PreprocessingPipeline {
    steps: Vec<Box<dyn Preprocessor>>,
}

impl PreprocessingPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a preprocessor step. Returns `self` for chaining.
    pub fn add(mut self, step: Box<dyn Preprocessor>) -> Self {
        self.steps.push(step);
        self
    }

    /// Run all steps in order and return the transformed data.
    pub fn run(&self, data: SaxsData) -> SaxsData {
        self.steps.iter().fold(data, |d, step| step.process(d))
    }

    /// Number of steps in the pipeline.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}
