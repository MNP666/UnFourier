use anyhow::{bail, Result};

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
    fn process(&self, data: SaxsData) -> Result<SaxsData>;
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

    fn process(&self, data: SaxsData) -> Result<SaxsData> {
        Ok(data)
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
    pub fn run(&self, data: SaxsData) -> Result<SaxsData> {
        self.steps.iter().try_fold(data, |d, step| step.process(d))
    }

    /// Number of steps in the pipeline.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ClipNegative
// ---------------------------------------------------------------------------

/// Replace non-positive intensities with a floor value and inflate their
/// errors so that those points contribute negligibly to the fit.
///
/// Specifically:
/// - `i_floor  = min(I[i] for I[i] > 0)` — the smallest positive intensity.
/// - `sigma_ceil = sigma_inflate_factor × max(σ[i])` — inflated error.
/// - Every point with `I[i] ≤ 0` is set to `(i_floor, sigma_ceil)`.
///
/// The inflated weight `1/sigma_ceil²` is a factor `sigma_inflate_factor²`
/// below the maximum weight, making clipped points effectively invisible to
/// the solver.
///
/// Returns `Err` if the dataset contains no positive intensities at all.
pub struct ClipNegative {
    pub sigma_inflate_factor: f64,
}

impl Default for ClipNegative {
    fn default() -> Self {
        Self { sigma_inflate_factor: 1000.0 }
    }
}

impl Preprocessor for ClipNegative {
    fn name(&self) -> &str {
        "clip-negative"
    }

    fn process(&self, mut data: SaxsData) -> Result<SaxsData> {
        let i_floor = data
            .intensity
            .iter()
            .cloned()
            .filter(|&v| v > 0.0)
            .fold(f64::INFINITY, f64::min);

        if i_floor.is_infinite() {
            bail!("clip-negative: dataset contains no positive intensities");
        }

        let sigma_ceil = self.sigma_inflate_factor
            * data.error.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut n_clipped = 0usize;
        for i in 0..data.intensity.len() {
            if data.intensity[i] <= 0.0 {
                data.intensity[i] = i_floor;
                data.error[i] = sigma_ceil;
                n_clipped += 1;
            }
        }

        if n_clipped > 0 {
            eprintln!(
                "clip-negative: clipped {n_clipped} non-positive point(s) \
                 (i_floor={i_floor:.4e}, sigma_ceil={sigma_ceil:.4e})"
            );
        }

        Ok(data)
    }
}

// ---------------------------------------------------------------------------
// OmitNonPositive
// ---------------------------------------------------------------------------

/// Remove all data points where `I ≤ 0`.
///
/// Simpler than `ClipNegative` but loses q-coverage at the affected points.
/// Returns `Err` if no points remain after filtering.
pub struct OmitNonPositive;

impl Preprocessor for OmitNonPositive {
    fn name(&self) -> &str {
        "omit-non-positive"
    }

    fn process(&self, data: SaxsData) -> Result<SaxsData> {
        let indices: Vec<usize> = (0..data.intensity.len())
            .filter(|&i| data.intensity[i] > 0.0)
            .collect();

        if indices.is_empty() {
            bail!("omit-non-positive: no points with positive intensity remain");
        }

        let q = indices.iter().map(|&i| data.q[i]).collect();
        let intensity = indices.iter().map(|&i| data.intensity[i]).collect();
        let error = indices.iter().map(|&i| data.error[i]).collect();

        SaxsData::new(q, intensity, error)
            .map_err(|e| anyhow::anyhow!("omit-non-positive: {e}"))
    }
}

// ---------------------------------------------------------------------------
// QRangeSelector
// ---------------------------------------------------------------------------

/// Trim the dataset to a q-range and/or SNR cutoff.
///
/// Applied in order:
/// 1. Discard points with `q < q_min` (if set).
/// 2. Discard points with `q > q_max` (if set).
/// 3. Trim the noisy high-q tail: scan from the highest-q point inward and
///    remove leading points where `I/σ < snr_threshold` (if set).
///
/// Returns `Err` if no points remain after filtering.
pub struct QRangeSelector {
    pub q_min: Option<f64>,
    pub q_max: Option<f64>,
    pub snr_threshold: Option<f64>,
}

impl Preprocessor for QRangeSelector {
    fn name(&self) -> &str {
        "q-range-selector"
    }

    fn process(&self, data: SaxsData) -> Result<SaxsData> {
        // Step 1 & 2: filter by explicit q bounds.
        let mut indices: Vec<usize> = (0..data.q.len())
            .filter(|&i| {
                self.q_min.map_or(true, |lo| data.q[i] >= lo)
                    && self.q_max.map_or(true, |hi| data.q[i] <= hi)
            })
            .collect();

        // Step 3: trim noisy high-q tail by SNR.
        if let Some(threshold) = self.snr_threshold {
            // Walk from the high-q end inward; find the last index (in `indices`)
            // whose SNR meets the threshold and drop everything beyond it.
            let cutoff = indices.iter().rposition(|&i| {
                let snr = data.intensity[i] / data.error[i];
                snr >= threshold
            });
            match cutoff {
                Some(pos) => indices.truncate(pos + 1),
                None => indices.clear(), // nothing meets the threshold
            }
        }

        if indices.is_empty() {
            bail!("q-range-selector: no points remain after filtering");
        }

        let q = indices.iter().map(|&i| data.q[i]).collect();
        let intensity = indices.iter().map(|&i| data.intensity[i]).collect();
        let error = indices.iter().map(|&i| data.error[i]).collect();

        SaxsData::new(q, intensity, error)
            .map_err(|e| anyhow::anyhow!("q-range-selector: {e}"))
    }
}

// ---------------------------------------------------------------------------
// LogRebin
// ---------------------------------------------------------------------------

/// Reduce data density by averaging into logarithmically-spaced q bins.
///
/// For each non-empty bin:
/// - `q_new   = mean(q[j])`
/// - `I_new   = mean(I[j])`
/// - `σ_new   = sqrt(Σ σ[j]²) / n`  (standard error of the mean)
///
/// Empty bins are discarded. Returns `Err` if `n_bins < 1` or no bins are
/// non-empty (e.g. the dataset is empty after upstream filtering).
pub struct LogRebin {
    pub n_bins: usize,
}

impl Preprocessor for LogRebin {
    fn name(&self) -> &str {
        "log-rebin"
    }

    fn process(&self, data: SaxsData) -> Result<SaxsData> {
        if self.n_bins == 0 {
            bail!("log-rebin: n_bins must be at least 1");
        }
        if data.is_empty() {
            bail!("log-rebin: input dataset is empty");
        }

        let q_lo = data.q[0];
        let q_hi = data.q[data.len() - 1];

        if q_lo <= 0.0 {
            bail!("log-rebin: q values must be positive for logarithmic binning (got q_min={q_lo})");
        }

        // n_bins + 1 log-spaced edges covering [q_lo, q_hi].
        let log_lo = q_lo.ln();
        let log_hi = q_hi.ln();
        let edges: Vec<f64> = (0..=self.n_bins)
            .map(|k| (log_lo + (log_hi - log_lo) * k as f64 / self.n_bins as f64).exp())
            .collect();

        let mut q_out = Vec::with_capacity(self.n_bins);
        let mut i_out = Vec::with_capacity(self.n_bins);
        let mut e_out = Vec::with_capacity(self.n_bins);

        for b in 0..self.n_bins {
            let lo = edges[b];
            let hi = edges[b + 1];

            // Include the right edge only in the last bin to avoid dropping q_hi.
            let in_bin: Vec<usize> = (0..data.len())
                .filter(|&j| {
                    if b + 1 == self.n_bins {
                        data.q[j] >= lo && data.q[j] <= hi
                    } else {
                        data.q[j] >= lo && data.q[j] < hi
                    }
                })
                .collect();

            if in_bin.is_empty() {
                continue;
            }

            let n = in_bin.len() as f64;
            let q_mean = in_bin.iter().map(|&j| data.q[j]).sum::<f64>() / n;
            let i_mean = in_bin.iter().map(|&j| data.intensity[j]).sum::<f64>() / n;
            let sigma_se = (in_bin.iter().map(|&j| data.error[j].powi(2)).sum::<f64>()).sqrt() / n;

            q_out.push(q_mean);
            i_out.push(i_mean);
            e_out.push(sigma_se);
        }

        if q_out.is_empty() {
            bail!("log-rebin: all bins are empty");
        }

        SaxsData::new(q_out, i_out, e_out)
            .map_err(|e| anyhow::anyhow!("log-rebin: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(q: &[f64], intensity: &[f64], error: &[f64]) -> SaxsData {
        SaxsData::new(q.to_vec(), intensity.to_vec(), error.to_vec()).unwrap()
    }

    #[test]
    fn clip_negative_replaces_non_positive_with_floor_and_inflated_sigma() {
        // 5 points: indices 1 and 3 are non-positive.
        let data = make_data(
            &[0.01, 0.02, 0.03, 0.04, 0.05],
            &[10.0, -5.0, 8.0, 0.0, 6.0],
            &[0.1, 0.1, 0.1, 0.1, 0.2],
        );

        let clipper = ClipNegative::default(); // factor = 1000
        let out = clipper.process(data).unwrap();

        // i_floor = min positive = 6.0
        let i_floor = 6.0_f64;
        // sigma_ceil = 1000 * max(sigma) = 1000 * 0.2 = 200.0
        let sigma_ceil = 200.0_f64;

        // Non-positive points replaced.
        assert!((out.intensity[1] - i_floor).abs() < 1e-12);
        assert!((out.intensity[3] - i_floor).abs() < 1e-12);
        assert!((out.error[1] - sigma_ceil).abs() < 1e-12);
        assert!((out.error[3] - sigma_ceil).abs() < 1e-12);

        // Positive points untouched.
        assert!((out.intensity[0] - 10.0).abs() < 1e-12);
        assert!((out.intensity[2] - 8.0).abs() < 1e-12);
        assert!((out.intensity[4] - 6.0).abs() < 1e-12);
        assert!((out.error[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn clip_negative_errors_when_no_positive_intensities() {
        let data = make_data(&[0.01, 0.02], &[-1.0, -2.0], &[0.1, 0.1]);
        let err = ClipNegative::default().process(data).unwrap_err();
        assert!(err.to_string().contains("no positive intensities"));
    }

    #[test]
    fn q_range_selector_manual_range_returns_expected_subset() {
        // 6 points at q = 0.01, 0.02, 0.03, 0.04, 0.05, 0.06
        let data = make_data(
            &[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            &[10.0; 6],
            &[0.1; 6],
        );
        let sel = QRangeSelector { q_min: Some(0.02), q_max: Some(0.05), snr_threshold: None };
        let out = sel.process(data).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out.q[0] - 0.02).abs() < 1e-12);
        assert!((out.q[3] - 0.05).abs() < 1e-12);
    }

    #[test]
    fn q_range_selector_snr_threshold_trims_noisy_tail() {
        // 5 points; the last two have SNR = 1.0 / 5.0 = 0.2 < 2.0 threshold.
        // Point at index 2 has SNR = 8.0 / 0.5 = 16 — should be the new last point.
        let data = make_data(
            &[0.01, 0.02, 0.03, 0.04, 0.05],
            &[20.0, 15.0, 8.0, 1.0, 1.0],
            &[0.1, 0.1, 0.5, 5.0, 5.0],
        );
        let sel = QRangeSelector { q_min: None, q_max: None, snr_threshold: Some(2.0) };
        let out = sel.process(data).unwrap();
        // Points at index 3 and 4 (SNR = 0.2) should be trimmed.
        assert_eq!(out.len(), 3);
        assert!((out.q[2] - 0.03).abs() < 1e-12);
    }

    #[test]
    fn log_rebin_100_points_to_20_bins_correct_averages() {
        // 100 log-spaced points from q=0.01 to q=1.0 with constant I=10, σ=0.5.
        let n = 100usize;
        let q: Vec<f64> = (0..n)
            .map(|i| (0.01_f64.ln() + (1.0_f64.ln() - 0.01_f64.ln()) * i as f64 / (n - 1) as f64).exp())
            .collect();
        let intensity = vec![10.0_f64; n];
        let error = vec![0.5_f64; n];
        let data = make_data(&q, &intensity, &error);

        let rebinner = LogRebin { n_bins: 20 };
        let out = rebinner.process(data).unwrap();

        // Should produce exactly 20 non-empty bins (input is dense and log-spaced).
        assert_eq!(out.len(), 20);

        // Mean intensity in each bin must be 10.0 (all inputs are 10.0).
        for i in 0..out.len() {
            assert!((out.intensity[i] - 10.0).abs() < 1e-10,
                "bin {i}: expected I=10.0, got {}", out.intensity[i]);
        }

        // q values must be strictly increasing (monotone output).
        for i in 0..out.len() - 1 {
            assert!(out.q[i] < out.q[i + 1],
                "q not monotone at bin {i}: {} >= {}", out.q[i], out.q[i + 1]);
        }

        // σ_new = sqrt(n_in * 0.25) / n_in = 0.5 / sqrt(n_in).
        // For each bin with n_in points: σ_new = 0.5 / sqrt(n_in) < 0.5.
        for i in 0..out.len() {
            assert!(out.error[i] <= 0.5 + 1e-10,
                "bin {i}: error should not exceed single-point σ=0.5, got {}", out.error[i]);
            assert!(out.error[i] > 0.0, "bin {i}: error should be positive");
        }
    }

    #[test]
    fn omit_non_positive_keeps_only_positive_points() {
        let data = make_data(
            &[0.01, 0.02, 0.03, 0.04, 0.05],
            &[5.0, -1.0, 3.0, 0.0, 7.0],
            &[0.1; 5],
        );
        let out = OmitNonPositive.process(data).unwrap();
        assert_eq!(out.len(), 3);
        // Remaining q values should be the positive-intensity ones.
        assert!((out.q[0] - 0.01).abs() < 1e-12);
        assert!((out.q[1] - 0.03).abs() < 1e-12);
        assert!((out.q[2] - 0.05).abs() < 1e-12);
        // All intensities must be positive.
        assert!(out.intensity.iter().all(|&v| v > 0.0));
    }
}
