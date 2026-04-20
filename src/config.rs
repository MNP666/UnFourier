use serde::Deserialize;
use std::fs;
use unfourier::basis::SplineBoundaryMode;

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RegularisationConfig {
    pub method: Option<String>,
    pub lambda_min: Option<f64>,
    pub lambda_max: Option<f64>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PreprocessingConfig {
    pub qmin: Option<f64>,
    pub qmax: Option<f64>,
    pub negative_handling: Option<String>,
    pub auto_qmin: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct GuinierConfig {
    pub report: Option<bool>,
    pub min_points: Option<usize>,
    pub max_points: Option<usize>,
    pub max_skip: Option<usize>,
    pub max_qrg: Option<f64>,
    pub stability_windows: Option<usize>,
    pub rg_tolerance: Option<f64>,
    pub i0_tolerance: Option<f64>,
    pub max_chi2: Option<f64>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct BasisConfig {
    pub n_basis: Option<usize>,
    pub knot_spacing: Option<f64>,
    pub min_basis: Option<usize>,
    pub max_basis: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ConstraintsConfig {
    pub spline_boundary: Option<SplineBoundaryMode>,
    /// -1 = disabled, 0 = default relative weight, >0 = explicit relative weight
    pub d1_smoothness: Option<f64>,
    pub d2_smoothness: Option<f64>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct UnfourierConfig {
    #[serde(default)]
    pub regularisation: RegularisationConfig,
    #[serde(default)]
    pub preprocessing: PreprocessingConfig,
    #[serde(default)]
    pub guinier: GuinierConfig,
    #[serde(default)]
    pub basis: BasisConfig,
    #[serde(default)]
    pub constraints: ConstraintsConfig,
}

impl UnfourierConfig {
    /// Look for `unfourier.toml` in the current working directory.
    /// Returns `None` if absent, `Err` if present but unparseable.
    pub fn load() -> anyhow::Result<Option<Self>> {
        let path = std::path::Path::new("unfourier.toml");
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read unfourier.toml: {}", e))?;
        let cfg: UnfourierConfig = toml::from_str(&text)
            .map_err(|e| anyhow::anyhow!("failed to parse unfourier.toml: {}", e))?;
        Ok(Some(cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_toml() {
        let text = r#"
[regularisation]
lambda_min = 1.0
lambda_max = 1e3

[preprocessing]
qmax = 0.30
"#;
        let cfg: UnfourierConfig = toml::from_str(text).unwrap();
        assert_eq!(cfg.regularisation.lambda_min, Some(1.0));
        assert_eq!(cfg.regularisation.lambda_max, Some(1e3));
        assert_eq!(cfg.preprocessing.qmax, Some(0.30));
        assert!(cfg.preprocessing.qmin.is_none());
        assert!(cfg.regularisation.method.is_none());
    }

    #[test]
    fn parse_full_toml() {
        let text = r#"
[regularisation]
method = "lcurve"
lambda_min = 1e-4
lambda_max = 1e2

[preprocessing]
qmin = 0.01
qmax = 0.50
negative_handling = "omit"
auto_qmin = "guinier"

[guinier]
report = true
min_points = 8
max_points = 25
max_skip = 8
max_qrg = 1.3
stability_windows = 3
rg_tolerance = 0.02
i0_tolerance = 0.03
max_chi2 = 3.0

[basis]
n_basis = 30
knot_spacing = 7.5
min_basis = 12
max_basis = 48

[constraints]
spline_boundary = "value_slope_zero"
d1_smoothness = 0.0
d2_smoothness = 1.5
"#;
        let cfg: UnfourierConfig = toml::from_str(text).unwrap();
        assert_eq!(cfg.regularisation.method.as_deref(), Some("lcurve"));
        assert_eq!(cfg.preprocessing.qmin, Some(0.01));
        assert_eq!(cfg.preprocessing.negative_handling.as_deref(), Some("omit"));
        assert_eq!(cfg.preprocessing.auto_qmin.as_deref(), Some("guinier"));
        assert_eq!(cfg.guinier.report, Some(true));
        assert_eq!(cfg.guinier.min_points, Some(8));
        assert_eq!(cfg.guinier.max_points, Some(25));
        assert_eq!(cfg.guinier.max_skip, Some(8));
        assert_eq!(cfg.guinier.max_qrg, Some(1.3));
        assert_eq!(cfg.guinier.stability_windows, Some(3));
        assert_eq!(cfg.guinier.rg_tolerance, Some(0.02));
        assert_eq!(cfg.guinier.i0_tolerance, Some(0.03));
        assert_eq!(cfg.guinier.max_chi2, Some(3.0));
        assert_eq!(cfg.basis.n_basis, Some(30));
        assert_eq!(cfg.basis.knot_spacing, Some(7.5));
        assert_eq!(cfg.basis.min_basis, Some(12));
        assert_eq!(cfg.basis.max_basis, Some(48));
        assert_eq!(
            cfg.constraints.spline_boundary,
            Some(SplineBoundaryMode::ValueSlopeZero)
        );
        assert_eq!(cfg.constraints.d1_smoothness, Some(0.0));
        assert_eq!(cfg.constraints.d2_smoothness, Some(1.5));
    }

    #[test]
    fn empty_toml_is_all_none() {
        let cfg: UnfourierConfig = toml::from_str("").unwrap();
        assert!(cfg.regularisation.lambda_min.is_none());
        assert!(cfg.preprocessing.qmax.is_none());
        assert!(cfg.guinier.report.is_none());
    }

    #[test]
    fn old_basis_kind_is_rejected() {
        let text = r#"
[basis]
type = "rect"
"#;
        let err = toml::from_str::<UnfourierConfig>(text).unwrap_err();
        assert!(
            err.to_string().contains("unknown field"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn old_basis_npoints_is_rejected() {
        let text = r#"
[basis]
npoints = 100
"#;
        let err = toml::from_str::<UnfourierConfig>(text).unwrap_err();
        assert!(
            err.to_string().contains("unknown field"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parse_knot_spacing_basis_toml() {
        let text = r#"
[basis]
knot_spacing = 7.5
min_basis = 12
max_basis = 48
"#;
        let cfg: UnfourierConfig = toml::from_str(text).unwrap();
        assert_eq!(cfg.basis.n_basis, None);
        assert_eq!(cfg.basis.knot_spacing, Some(7.5));
        assert_eq!(cfg.basis.min_basis, Some(12));
        assert_eq!(cfg.basis.max_basis, Some(48));
    }

    #[test]
    fn parse_constraints_toml() {
        let text = r#"
[constraints]
spline_boundary = "value_zero"
d1_smoothness = -1.0
d2_smoothness = 1.0
"#;
        let cfg: UnfourierConfig = toml::from_str(text).unwrap();
        assert_eq!(
            cfg.constraints.spline_boundary,
            Some(SplineBoundaryMode::ValueZero)
        );
        assert_eq!(cfg.constraints.d1_smoothness, Some(-1.0));
        assert_eq!(cfg.constraints.d2_smoothness, Some(1.0));

        // Explicit weight
        let text2 = r#"
[constraints]
d1_smoothness = 2.5
d2_smoothness = 0.75
"#;
        let cfg2: UnfourierConfig = toml::from_str(text2).unwrap();
        assert_eq!(cfg2.constraints.d1_smoothness, Some(2.5));
        assert_eq!(cfg2.constraints.d2_smoothness, Some(0.75));

        // Absent section → all None
        let cfg3: UnfourierConfig = toml::from_str("").unwrap();
        assert!(cfg3.constraints.d1_smoothness.is_none());
        assert!(cfg3.constraints.d2_smoothness.is_none());
        assert!(cfg3.constraints.spline_boundary.is_none());
    }

    #[test]
    fn invalid_spline_boundary_is_rejected() {
        let text = r#"
[constraints]
spline_boundary = "clamped-ish"
"#;
        let err = toml::from_str::<UnfourierConfig>(text).unwrap_err();
        assert!(
            err.to_string().contains("unknown variant"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unknown_guinier_field_is_rejected() {
        let text = r#"
[guinier]
magic = true
"#;
        let err = toml::from_str::<UnfourierConfig>(text).unwrap_err();
        assert!(
            err.to_string().contains("unknown field"),
            "unexpected error: {err}"
        );
    }
}
