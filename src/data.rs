use anyhow::{Context, Result, bail};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("data vectors have inconsistent lengths")]
    InconsistentLengths,
    #[error("data contains no points")]
    Empty,
    #[error(
        "q values are not strictly increasing: first violation at source line {line_num} (q[i]={prev} >= q[i+1]={next})"
    )]
    NonMonotoneQ {
        line_num: usize,
        prev: f64,
        next: f64,
    },
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
        Ok(Self {
            q,
            intensity,
            error,
        })
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
    // Tracks the 1-based source line number for each accepted data row.
    let mut source_lines: Vec<usize> = Vec::new();
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
                source_lines.push(line_num + 1);
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

    // Verify q is strictly increasing.
    for i in 0..q.len() - 1 {
        if q[i] >= q[i + 1] {
            bail!(DataError::NonMonotoneQ {
                line_num: source_lines[i + 1],
                prev: q[i],
                next: q[i + 1],
            });
        }
    }

    SaxsData::new(q, intensity, error).context("parsed data failed validation")
}

/// Parse a `.dat`-format string directly (for testing without touching the
/// filesystem).  Uses the same logic as `parse_dat` but operates on a
/// `&str` instead of a file path.
pub fn parse_dat_str(content: &str) -> Result<SaxsData> {
    let mut q = Vec::new();
    let mut intensity = Vec::new();
    let mut error = Vec::new();
    let mut source_lines: Vec<usize> = Vec::new();

    for (line_num, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 3 {
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
                source_lines.push(line_num + 1);
            }
            _ => {}
        }
    }

    if q.is_empty() {
        bail!("no valid data found (all lines were comments, headers, or unparseable)");
    }

    for i in 0..q.len() - 1 {
        if q[i] >= q[i + 1] {
            bail!(DataError::NonMonotoneQ {
                line_num: source_lines[i + 1],
                prev: q[i],
                next: q[i + 1],
            });
        }
    }

    SaxsData::new(q, intensity, error).context("parsed data failed validation")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_monotone_q_returns_error_with_line_number() {
        // Lines 1-2 are a header (skipped), data starts at line 3.
        // The third data row (source line 7) has a q value less than the previous.
        let input = "\
# Example header
# q  I  sigma
0.01  100.0  1.0
0.02  90.0   1.0
0.03  80.0   1.0
0.025 70.0   1.0
0.04  60.0   1.0
";
        let err = parse_dat_str(input).unwrap_err();
        let msg = err.to_string();
        // The offending row is source line 6 (0-indexed line 5 → +1 = 6).
        assert!(
            msg.contains("line 6"),
            "expected 'line 6' in error, got: {msg}"
        );
        assert!(
            msg.contains("NonMonotoneQ")
                || msg.contains("strictly increasing")
                || msg.contains("violation"),
            "expected monotonicity error, got: {msg}"
        );
    }

    #[test]
    fn leading_whitespace_and_tab_delimiters_parse_correctly() {
        let input = "  0.01\t100.0\t1.0\n\t0.02\t90.0\t0.9\n   0.03  80.0  0.8\n";
        let data = parse_dat_str(input).expect("should parse");
        assert_eq!(data.len(), 3);
        assert!((data.q[0] - 0.01).abs() < 1e-12);
        assert!((data.q[1] - 0.02).abs() < 1e-12);
        assert!((data.intensity[1] - 90.0).abs() < 1e-12);
        assert!((data.error[2] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn crlf_line_endings_parse_correctly() {
        let input = "0.01 100.0 1.0\r\n0.02 90.0 0.9\r\n0.03 80.0 0.8\r\n";
        let data = parse_dat_str(input).expect("should parse CRLF");
        assert_eq!(data.len(), 3);
        assert!((data.q[2] - 0.03).abs() < 1e-12);
    }
}
