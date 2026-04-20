use std::fmt;

use crate::data::SaxsData;

/// Configuration for the low-q Guinier preflight scan.
///
/// This type is independent of CLI/TOML parsing so tests and future callers can
/// run the scanner directly.
#[derive(Debug, Clone, PartialEq)]
pub struct GuinierScanConfig {
    pub min_points: usize,
    pub max_points: usize,
    pub max_skip: usize,
    pub max_qrg: f64,
    pub stability_windows: usize,
    pub rg_tolerance: f64,
    pub i0_tolerance: f64,
    pub max_chi2: f64,
}

impl Default for GuinierScanConfig {
    fn default() -> Self {
        Self {
            min_points: 8,
            max_points: 25,
            max_skip: 8,
            max_qrg: 1.3,
            stability_windows: 3,
            rg_tolerance: 0.02,
            i0_tolerance: 0.03,
            max_chi2: 3.0,
        }
    }
}

/// Explicit reason a candidate Guinier window was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuinierRejectReason {
    TooFewPoints,
    NonFiniteInput,
    NonPositiveIntensity,
    InvalidSigma,
    SingularFit,
    PositiveSlope,
    InvalidDerivedValue,
    QrgTooHigh,
    PoorChiSquared,
}

impl fmt::Display for GuinierRejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::TooFewPoints => "too few points",
            Self::NonFiniteInput => "non-finite input",
            Self::NonPositiveIntensity => "non-positive intensity",
            Self::InvalidSigma => "invalid sigma",
            Self::SingularFit => "singular fit",
            Self::PositiveSlope => "positive slope",
            Self::InvalidDerivedValue => "invalid derived value",
            Self::QrgTooHigh => "qmax*Rg too high",
            Self::PoorChiSquared => "poor chi-squared",
        };
        f.write_str(label)
    }
}

/// Result for one attempted Guinier window.
#[derive(Debug, Clone, PartialEq)]
pub struct GuinierWindowFit {
    pub skip: usize,
    pub n_points: usize,
    pub q_min: Option<f64>,
    pub q_max: Option<f64>,
    pub qrg_max: Option<f64>,
    pub rg: Option<f64>,
    pub i0: Option<f64>,
    pub slope: Option<f64>,
    pub intercept: Option<f64>,
    pub chi2_red: Option<f64>,
    pub valid: bool,
    pub reject_reason: Option<GuinierRejectReason>,
}

impl GuinierWindowFit {
    fn new(skip: usize, n_points: usize, q_min: Option<f64>, q_max: Option<f64>) -> Self {
        Self {
            skip,
            n_points,
            q_min,
            q_max,
            qrg_max: None,
            rg: None,
            i0: None,
            slope: None,
            intercept: None,
            chi2_red: None,
            valid: false,
            reject_reason: None,
        }
    }

    fn reject(mut self, reason: GuinierRejectReason) -> Self {
        self.valid = false;
        self.reject_reason = Some(reason);
        self
    }

    fn accept(mut self) -> Self {
        self.valid = true;
        self.reject_reason = None;
        self
    }
}

/// Stable low-q cutoff suggested by the scan.
#[derive(Debug, Clone, PartialEq)]
pub struct GuinierRecommendation {
    pub skip: usize,
    pub q_min: f64,
    pub rg: f64,
    pub i0: f64,
    pub chi2_red: f64,
    pub confidence_label: String,
}

/// Complete side-effect-free scan output.
#[derive(Debug, Clone, PartialEq)]
pub struct GuinierScanReport {
    pub config: GuinierScanConfig,
    pub candidate_fits: Vec<GuinierWindowFit>,
    pub best_fits: Vec<GuinierWindowFit>,
    pub recommendation: Option<GuinierRecommendation>,
}

/// Run the Guinier preflight scan over increasing low-q truncations.
pub fn scan_guinier(data: &SaxsData, config: &GuinierScanConfig) -> GuinierScanReport {
    let mut candidate_fits = Vec::new();
    let mut best_fits = Vec::new();

    if !data.is_empty() && config.max_points >= config.min_points {
        let max_skip = config.max_skip.min(data.len().saturating_sub(1));
        for skip in 0..=max_skip {
            let remaining = data.len() - skip;
            if remaining < config.min_points {
                candidate_fits.push(fit_guinier_window(data, skip, remaining, config));
                continue;
            }

            let upper = config.max_points.min(remaining);
            let first_fit = candidate_fits.len();
            for n_points in config.min_points..=upper {
                candidate_fits.push(fit_guinier_window(data, skip, n_points, config));
            }

            if let Some(best) = candidate_fits[first_fit..]
                .iter()
                .filter(|fit| fit.valid)
                .max_by_key(|fit| fit.n_points)
                .cloned()
            {
                best_fits.push(best);
            }
        }
    }

    let recommendation = recommend_from_best_fits(&best_fits, config);

    GuinierScanReport {
        config: config.clone(),
        candidate_fits,
        best_fits,
        recommendation,
    }
}

/// Fit one window of `ln(I)` against `q^2` using weighted least squares.
pub fn fit_guinier_window(
    data: &SaxsData,
    skip: usize,
    n_points: usize,
    config: &GuinierScanConfig,
) -> GuinierWindowFit {
    let q_min = data.q.get(skip).copied();
    let end = skip.saturating_add(n_points);
    let q_max = if skip < data.len() {
        data.q.get(end.min(data.len()).saturating_sub(1)).copied()
    } else {
        None
    };
    let mut fit = GuinierWindowFit::new(skip, n_points, q_min, q_max);

    let min_required = config.min_points.max(3);
    if skip >= data.len() || n_points < min_required || end > data.len() {
        return fit.reject(GuinierRejectReason::TooFewPoints);
    }

    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wy = 0.0;
    let mut sum_wxx = 0.0;
    let mut sum_wxy = 0.0;

    for idx in skip..end {
        let q = data.q[idx];
        let intensity = data.intensity[idx];
        let sigma = data.error[idx];

        if !q.is_finite() || !intensity.is_finite() || !sigma.is_finite() {
            return fit.reject(GuinierRejectReason::NonFiniteInput);
        }
        if intensity <= 0.0 {
            return fit.reject(GuinierRejectReason::NonPositiveIntensity);
        }
        if sigma <= 0.0 {
            return fit.reject(GuinierRejectReason::InvalidSigma);
        }

        let x = q * q;
        let y = intensity.ln();
        let sigma_y = sigma / intensity;
        let weight = 1.0 / (sigma_y * sigma_y);

        if !x.is_finite() || !y.is_finite() || !sigma_y.is_finite() || !weight.is_finite() {
            return fit.reject(GuinierRejectReason::NonFiniteInput);
        }
        if sigma_y <= 0.0 || weight <= 0.0 {
            return fit.reject(GuinierRejectReason::InvalidSigma);
        }

        sum_w += weight;
        sum_wx += weight * x;
        sum_wy += weight * y;
        sum_wxx += weight * x * x;
        sum_wxy += weight * x * y;
    }

    let denom = sum_w * sum_wxx - sum_wx * sum_wx;
    if !denom.is_finite() || denom.abs() <= f64::EPSILON * sum_w.max(1.0) * sum_wxx.abs().max(1.0) {
        return fit.reject(GuinierRejectReason::SingularFit);
    }

    let slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
    let intercept = (sum_wy - slope * sum_wx) / sum_w;

    fit.slope = Some(slope);
    fit.intercept = Some(intercept);

    if !slope.is_finite() || !intercept.is_finite() {
        return fit.reject(GuinierRejectReason::SingularFit);
    }
    if slope >= 0.0 {
        return fit.reject(GuinierRejectReason::PositiveSlope);
    }

    let rg = (-3.0 * slope).sqrt();
    let i0 = intercept.exp();
    if !rg.is_finite() || !i0.is_finite() || rg <= 0.0 || i0 <= 0.0 {
        return fit.reject(GuinierRejectReason::InvalidDerivedValue);
    }

    let q_max = data.q[end - 1];
    let qrg_max = q_max * rg;
    fit.rg = Some(rg);
    fit.i0 = Some(i0);
    fit.qrg_max = Some(qrg_max);

    let max_qrg = if config.max_qrg.is_finite() {
        config.max_qrg
    } else {
        f64::INFINITY
    };
    if qrg_max > max_qrg {
        return fit.reject(GuinierRejectReason::QrgTooHigh);
    }

    let mut chi2 = 0.0;
    for idx in skip..end {
        let q = data.q[idx];
        let intensity = data.intensity[idx];
        let sigma = data.error[idx];
        let x = q * q;
        let y = intensity.ln();
        let sigma_y = sigma / intensity;
        let residual = y - (intercept + slope * x);
        chi2 += (residual / sigma_y).powi(2);
    }

    let dof = n_points - 2;
    let chi2_red = chi2 / dof as f64;
    fit.chi2_red = Some(chi2_red);

    let max_chi2 = if config.max_chi2.is_finite() {
        config.max_chi2
    } else {
        f64::INFINITY
    };
    if !chi2_red.is_finite() {
        return fit.reject(GuinierRejectReason::InvalidDerivedValue);
    }
    if chi2_red > max_chi2 {
        return fit.reject(GuinierRejectReason::PoorChiSquared);
    }

    fit.accept()
}

fn recommend_from_best_fits(
    best_fits: &[GuinierWindowFit],
    config: &GuinierScanConfig,
) -> Option<GuinierRecommendation> {
    let plateau_len = config.stability_windows.max(1);
    if best_fits.len() < plateau_len {
        return None;
    }

    for start in 0..=best_fits.len() - plateau_len {
        let group = &best_fits[start..start + plateau_len];
        if !is_consecutive_by_skip(group) || !is_stable_plateau(group, config) {
            continue;
        }

        let first = &group[0];
        return Some(GuinierRecommendation {
            skip: first.skip,
            q_min: first.q_min?,
            rg: first.rg?,
            i0: first.i0?,
            chi2_red: first.chi2_red?,
            confidence_label: "stable".to_string(),
        });
    }

    None
}

fn is_consecutive_by_skip(fits: &[GuinierWindowFit]) -> bool {
    fits.windows(2).all(|pair| pair[1].skip == pair[0].skip + 1)
}

fn is_stable_plateau(fits: &[GuinierWindowFit], config: &GuinierScanConfig) -> bool {
    let first_rg = match fits.first().and_then(|fit| fit.rg) {
        Some(value) if value.is_finite() && value > 0.0 => value,
        _ => return false,
    };
    let first_i0 = match fits.first().and_then(|fit| fit.i0) {
        Some(value) if value.is_finite() && value > 0.0 => value,
        _ => return false,
    };

    let rg_tol = config.rg_tolerance.max(0.0);
    let i0_tol = config.i0_tolerance.max(0.0);

    fits.iter().all(|fit| {
        fit.valid
            && fit
                .rg
                .is_some_and(|rg| relative_delta(rg, first_rg) <= rg_tol)
            && fit
                .i0
                .is_some_and(|i0| relative_delta(i0, first_i0) <= i0_tol)
    })
}

fn relative_delta(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs()).max(f64::EPSILON);
    (a - b).abs() / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_guinier(rg: f64, i0: f64, n: usize, q0: f64, dq: f64, rel_sigma: f64) -> SaxsData {
        let q: Vec<f64> = (0..n).map(|idx| q0 + dq * idx as f64).collect();
        let intensity: Vec<f64> = q
            .iter()
            .map(|&qv| i0 * (-(rg * rg * qv * qv) / 3.0).exp())
            .collect();
        let error: Vec<f64> = intensity.iter().map(|&iv| rel_sigma * iv).collect();
        SaxsData::new(q, intensity, error).unwrap()
    }

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {actual} to be within {tol} of {expected}"
        );
    }

    #[test]
    fn weighted_fit_recovers_synthetic_guinier_parameters() {
        let rg = 42.0;
        let i0 = 123.0;
        let data = synthetic_guinier(rg, i0, 20, 0.002, 0.001, 0.02);
        let config = GuinierScanConfig::default();

        let fit = fit_guinier_window(&data, 0, 12, &config);

        assert!(fit.valid, "fit was rejected: {:?}", fit.reject_reason);
        assert_close(fit.rg.unwrap(), rg, 1e-10);
        assert_close(fit.i0.unwrap(), i0, 1e-10);
        assert!(fit.chi2_red.unwrap() < 1e-20);
    }

    #[test]
    fn positive_slope_is_rejected() {
        let q = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let intensity: Vec<f64> = q
            .iter()
            .map(|qv| (5.0_f64 + 1000.0_f64 * qv * qv).exp())
            .collect();
        let error: Vec<f64> = intensity.iter().map(|&iv| 0.01 * iv).collect();
        let data = SaxsData::new(q, intensity, error).unwrap();
        let config = GuinierScanConfig {
            min_points: 3,
            max_points: 5,
            ..Default::default()
        };

        let fit = fit_guinier_window(&data, 0, 5, &config);

        assert!(!fit.valid);
        assert_eq!(fit.reject_reason, Some(GuinierRejectReason::PositiveSlope));
    }

    #[test]
    fn non_positive_intensity_rejects_affected_window() {
        let mut data = synthetic_guinier(30.0, 80.0, 10, 0.002, 0.001, 0.02);
        data.intensity[1] = 0.0;
        let config = GuinierScanConfig {
            min_points: 3,
            max_points: 5,
            ..Default::default()
        };

        let bad = fit_guinier_window(&data, 0, 4, &config);
        let good = fit_guinier_window(&data, 2, 4, &config);

        assert_eq!(
            bad.reject_reason,
            Some(GuinierRejectReason::NonPositiveIntensity)
        );
        assert!(
            good.valid,
            "later window was rejected: {:?}",
            good.reject_reason
        );
    }

    #[test]
    fn zero_sigma_rejects_without_panicking() {
        let mut data = synthetic_guinier(30.0, 80.0, 10, 0.002, 0.001, 0.02);
        data.error[2] = 0.0;
        let config = GuinierScanConfig {
            min_points: 3,
            max_points: 5,
            ..Default::default()
        };

        let fit = fit_guinier_window(&data, 0, 4, &config);

        assert!(!fit.valid);
        assert_eq!(fit.reject_reason, Some(GuinierRejectReason::InvalidSigma));
    }

    #[test]
    fn window_below_configured_min_points_is_rejected() {
        let data = synthetic_guinier(30.0, 80.0, 10, 0.002, 0.001, 0.02);
        let config = GuinierScanConfig {
            min_points: 8,
            max_points: 10,
            ..Default::default()
        };

        let fit = fit_guinier_window(&data, 0, 7, &config);

        assert!(!fit.valid);
        assert_eq!(fit.reject_reason, Some(GuinierRejectReason::TooFewPoints));
    }

    #[test]
    fn scan_explores_candidate_windows_and_keeps_largest_valid_per_skip() {
        let data = synthetic_guinier(20.0, 50.0, 6, 0.002, 0.001, 0.02);
        let config = GuinierScanConfig {
            min_points: 3,
            max_points: 5,
            max_skip: 2,
            stability_windows: 2,
            ..Default::default()
        };

        let report = scan_guinier(&data, &config);

        assert_eq!(report.candidate_fits.len(), 8);
        assert_eq!(report.best_fits.len(), 3);
        assert_eq!(report.best_fits[0].n_points, 5);
        assert_eq!(report.best_fits[1].n_points, 5);
        assert_eq!(report.best_fits[2].n_points, 4);
    }

    #[test]
    fn scan_rejects_windows_beyond_guinier_qrg_limit() {
        let data = synthetic_guinier(20.0, 50.0, 12, 0.005, 0.005, 0.02);
        let config = GuinierScanConfig {
            min_points: 3,
            max_points: 8,
            max_skip: 0,
            max_qrg: 0.51,
            stability_windows: 1,
            ..Default::default()
        };

        let report = scan_guinier(&data, &config);

        assert!(
            report
                .candidate_fits
                .iter()
                .any(|fit| fit.reject_reason == Some(GuinierRejectReason::QrgTooHigh))
        );
        assert!(
            report
                .best_fits
                .iter()
                .all(|fit| fit.qrg_max.unwrap() <= config.max_qrg)
        );
        assert_eq!(report.best_fits[0].n_points, 5);
    }

    #[test]
    fn stable_synthetic_series_recommends_first_point() {
        let data = synthetic_guinier(35.0, 100.0, 32, 0.002, 0.001, 0.02);
        let config = GuinierScanConfig {
            min_points: 8,
            max_points: 12,
            max_skip: 4,
            stability_windows: 3,
            ..Default::default()
        };

        let report = scan_guinier(&data, &config);

        let rec = report
            .recommendation
            .expect("expected stable recommendation");
        assert_eq!(rec.skip, 0);
        assert_close(rec.q_min, data.q[0], 1e-12);
        assert_close(rec.rg, 35.0, 1e-10);
    }

    #[test]
    fn corrupted_first_point_recommends_later_qmin() {
        let mut data = synthetic_guinier(35.0, 100.0, 32, 0.002, 0.001, 0.01);
        data.intensity[0] *= 2.0;
        data.error[0] = 0.01 * data.intensity[0];
        let config = GuinierScanConfig {
            min_points: 8,
            max_points: 12,
            max_skip: 5,
            stability_windows: 3,
            max_chi2: 3.0,
            ..Default::default()
        };

        let report = scan_guinier(&data, &config);

        assert!(report.best_fits.iter().all(|fit| fit.skip != 0));
        let rec = report
            .recommendation
            .expect("expected recommendation after skipping bad point");
        assert_eq!(rec.skip, 1);
        assert_close(rec.q_min, data.q[1], 1e-12);
    }

    #[test]
    fn chaotic_low_q_series_returns_no_recommendation() {
        let mut data = synthetic_guinier(30.0, 100.0, 36, 0.002, 0.001, 0.05);
        for idx in 0..12 {
            let factor = if idx % 2 == 0 { 1.35 } else { 0.72 };
            data.intensity[idx] *= factor;
            data.error[idx] = 0.05 * data.intensity[idx];
        }
        let config = GuinierScanConfig {
            min_points: 8,
            max_points: 12,
            max_skip: 5,
            max_qrg: 10.0,
            stability_windows: 3,
            rg_tolerance: 0.005,
            i0_tolerance: 0.005,
            max_chi2: 1.0e12,
        };

        let report = scan_guinier(&data, &config);

        assert!(
            report.recommendation.is_none(),
            "unexpected recommendation: {:?}",
            report.recommendation
        );
    }
}
