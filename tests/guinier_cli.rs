use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_unfourier"))
}

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new(label: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("unfourier-{label}-{nanos}"));
        fs::create_dir(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn join(&self, child: &str) -> PathBuf {
        self.path.join(child)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn write_guinier_data(path: &Path, corrupt_first_point: bool) {
    let rg = 35.0_f64;
    let i0 = 100.0_f64;
    let mut rows = String::new();

    for idx in 0..32 {
        let q = 0.002 + 0.001 * idx as f64;
        let mut intensity = i0 * (-(rg * rg * q * q) / 3.0).exp();
        if corrupt_first_point && idx == 0 {
            intensity *= 2.0;
        }
        let sigma = 0.01 * intensity;
        rows.push_str(&format!("{q:.6} {intensity:.12e} {sigma:.12e}\n"));
    }

    fs::write(path, rows).unwrap();
}

fn run_unfourier(cwd: &Path, input: &Path, extra_args: &[&str]) -> (String, String) {
    let output = Command::new(binary())
        .current_dir(cwd)
        .arg(input)
        .args(["--rmax", "120", "--n-basis", "12"])
        .args(extra_args)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "unfourier failed\nstatus: {}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    (
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
    )
}

#[test]
fn guinier_report_only_does_not_change_pr_output() {
    let dir = TestDir::new("report-only");
    let input = dir.join("synthetic.dat");
    write_guinier_data(&input, false);

    let (baseline_pr, baseline_stderr) = run_unfourier(dir.path(), &input, &[]);
    let (report_pr, report_stderr) = run_unfourier(dir.path(), &input, &["--guinier-report"]);

    assert!(
        baseline_stderr.trim().is_empty(),
        "baseline should not print a Guinier report:\n{baseline_stderr}"
    );
    assert_eq!(
        report_pr, baseline_pr,
        "report-only Guinier mode changed P(r) output"
    );
    assert!(
        report_stderr.contains("Guinier scan:"),
        "report-only mode did not print the report:\n{report_stderr}"
    );
    assert!(
        report_stderr.contains("report-only"),
        "report did not identify report-only mode:\n{report_stderr}"
    );
}

#[test]
fn guinier_auto_qmin_applies_corrupted_first_point_when_unset() {
    let dir = TestDir::new("auto-apply");
    let input = dir.join("synthetic.dat");
    write_guinier_data(&input, true);

    let (_pr, stderr) = run_unfourier(
        dir.path(),
        &input,
        &["--auto-qmin", "guinier", "--guinier-report"],
    );

    assert!(
        stderr.contains("Guinier recommendation: qmin = 3.0000e-3 (skip 1)"),
        "expected skip-1 recommendation for corrupted first point:\n{stderr}"
    );
    assert!(
        stderr.contains("applied: low-q edge 2.0000e-3 -> 3.0000e-3"),
        "auto-qmin recommendation was not reported as applied:\n{stderr}"
    );
}

#[test]
fn guinier_auto_qmin_keeps_explicit_qmin() {
    let dir = TestDir::new("explicit-qmin");
    let input = dir.join("synthetic.dat");
    write_guinier_data(&input, true);

    let (_pr, stderr) = run_unfourier(
        dir.path(),
        &input,
        &[
            "--qmin",
            "0.006",
            "--auto-qmin",
            "guinier",
            "--guinier-report",
        ],
    );

    assert!(
        stderr.contains("not applied: qmin already set to 6.0000e-3"),
        "explicit qmin precedence was not reported:\n{stderr}"
    );
    assert!(
        !stderr.contains("applied: low-q edge"),
        "auto-qmin should not apply when qmin is explicit:\n{stderr}"
    );
}

#[test]
fn cli_auto_qmin_off_overrides_toml_auto_qmin() {
    let no_config_dir = TestDir::new("no-config");
    let config_dir = TestDir::new("config");
    let input = no_config_dir.join("synthetic.dat");
    write_guinier_data(&input, true);

    fs::write(
        config_dir.join("unfourier.toml"),
        r#"
[preprocessing]
auto_qmin = "guinier"
"#,
    )
    .unwrap();

    let (baseline_pr, _) = run_unfourier(no_config_dir.path(), &input, &[]);
    let (disabled_pr, disabled_stderr) =
        run_unfourier(config_dir.path(), &input, &["--auto-qmin", "off"]);
    let (_auto_pr, auto_stderr) = run_unfourier(config_dir.path(), &input, &[]);

    assert_eq!(
        disabled_pr, baseline_pr,
        "--auto-qmin off did not restore the no-auto-qmin behavior"
    );
    assert!(
        disabled_stderr.trim().is_empty(),
        "--auto-qmin off should not run the TOML auto preflight:\n{disabled_stderr}"
    );
    assert!(
        auto_stderr.contains("auto-qmin applied qmin = 3.0000e-3 (skip 1)"),
        "TOML auto_qmin did not apply without the CLI override:\n{auto_stderr}"
    );
}
