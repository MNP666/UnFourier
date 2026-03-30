use anyhow::{bail, Context, Result};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("data vectors have inconsistent lengths")]
    InconsistentLengths,
    #[error("data contains no points")]
    Empty,
}

/// Raw SAXS data: scattering vector q, measured intensity I(q), and error σ(q).
///
/// All three vectors are guaranteed to have the same length and at least one
/// element.
#[derive(Debug, Clone)]
pub struct SaxsData {
    pub q: Vec<f64>,
    pub intensity: Vec<f64>,
    pub error: Vec<f64>,
}

impl SaxsData {
    pub fn new(q: Vec<f64>, intensity: Vec<f64>, error: Vec<f64>) -> Result<Self> {
        if q.len() != intensity.len() || q.len() != error.len() {
            bail!(DataError::InconsistentLengths);
        }
        if q.is_empty() {
            bail!(DataError::Empty);
        }
        Ok(Self { q, intensity, error })
    }

    pub fn len(&self) -> usize {
        self.q.len()
    }

    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }

    /// Minimum q value in the dataset.
    pub fn q_min(&self) -> f64 {
        self.q.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Maximum q value in the dataset.
    pub fn q_max(&self) -> f64 {
        self.q.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Parse a 3-column whitespace-delimited `.dat` file into `SaxsData`.
///
/// # Format
/// - Lines beginning with `#` are treated as comments and skipped.
/// - Lines with fewer than 3 parseable floats are silently skipped (headers, etc.).
/// - Columns are: q  I(q)  σ(q)
///
/// This parser is intentionally lenient about headers and formatting. A more
/// sophisticated parser (handling ATSAS-style metadata blocks, 2-column files
/// with auto-estimated errors, etc.) will be added in a later milestone.
pub fn parse_dat<P: AsRef<Path>>(path: P) -> Result<SaxsData> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read '{}'", path.display()))?;

    let mut q = Vec::new();
    let mut intensity = Vec::new();
    let mut error = Vec::new();
    let mut skipped = 0usize;

    for (line_num, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 3 {
            skipped += 1;
            continue;
        }

        match (
            cols[0].parse::<f64>(),
            cols[1].parse::<f64>(),
            cols[2].parse::<f64>(),
        ) {
            (Ok(q_val), Ok(i_val), Ok(e_val)) => {
                q.push(q_val);
                intensity.push(i_val);
                error.push(e_val);
            }
            _ => {
                // Line has 3+ tokens but they're not all floats — likely a header row.
                skipped += 1;
                if skipped <= 3 {
                    eprintln!(
                        "note: skipped non-numeric line {}: {:?}",
                        line_num + 1,
                        raw_line
                    );
                }
            }
        }
    }

    if q.is_empty() {
        bail!(
            "no valid data found in '{}' (all lines were comments, headers, or unparseable)",
            path.display()
        );
    }

    SaxsData::new(q, intensity, error).context("parsed data failed validation")
}
