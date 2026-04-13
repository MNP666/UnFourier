use serde::Deserialize;
use std::fs;

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
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct BasisConfig {
    #[serde(rename = "type")]
    pub basis_type: Option<String>,
    pub npoints: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ConstraintsConfig {
    /// -1 = disabled, 0 = default relative weight 1.0, >0 = explicit relative weight
    pub d1_smoothness: Option<f64>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct UnfourierConfig {
    #[serde(default)]
    pub regularisation: RegularisationConfig,
    #[serde(default)]
    pub preprocessing: PreprocessingConfig,
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

[basis]
type = "spline"
npoints = 30
"#;
        let cfg: UnfourierConfig = toml::from_str(text).unwrap();
        assert_eq!(cfg.regularisation.method.as_deref(), Some("lcurve"));
        assert_eq!(cfg.preprocessing.qmin, Some(0.01));
        assert_eq!(cfg.basis.basis_type.as_deref(), Some("spline"));
        assert_eq!(cfg.basis.npoints, Some(30));
    }

    #[test]
    fn empty_toml_is_all_none() {
        let cfg: UnfourierConfig = toml::from_str("").unwrap();
        assert!(cfg.regularisation.lambda_min.is_none());
        assert!(cfg.preprocessing.qmax.is_none());
    }

    #[test]
    fn parse_constraints_toml() {
        let text = r#"
[constraints]
d1_smoothness = -1.0
"#;
        let cfg: UnfourierConfig = toml::from_str(text).unwrap();
        assert_eq!(cfg.constraints.d1_smoothness, Some(-1.0));

        // Explicit weight
        let text2 = r#"
[constraints]
d1_smoothness = 2.5
"#;
        let cfg2: UnfourierConfig = toml::from_str(text2).unwrap();
        assert_eq!(cfg2.constraints.d1_smoothness, Some(2.5));

        // Absent section → all None
        let cfg3: UnfourierConfig = toml::from_str("").unwrap();
        assert!(cfg3.constraints.d1_smoothness.is_none());
    }
}
