mod common;

use std::{path::PathBuf, process::Command};

/// Checks that bundled demos work outside the project directory and produce finite English-language results.
#[test]
fn demo_runs_from_another_directory_and_reports_both_tasks_in_english() {
    let directory = common::TempDir::new();
    let output = Command::new(env!("CARGO_BIN_EXE_neural"))
        .current_dir(&directory.path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Regression"));
    assert!(stdout.contains("Classification"));
    assert!(stdout.contains("Test accuracy: 100.00%"));
    let mse = stdout
        .lines()
        .find_map(|line| line.strip_prefix("Test MSE (standardized): "))
        .unwrap()
        .parse::<f64>()
        .unwrap();
    assert!(mse < 0.01);
    let prediction = stdout
        .lines()
        .find_map(|line| line.strip_prefix("Prediction for [-2.75]: "))
        .unwrap()
        .parse::<f64>()
        .unwrap();
    assert!((prediction + 4.5).abs() < 0.5);
    assert!(stdout.is_ascii());
    assert!(!stdout.contains("Loss plot saved"));
}

/// Returns an absolute path to a small bundled CSV fixture.
fn demo_file(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

/// Checks regression argument parsing and optional SVG output to a path containing spaces.
#[test]
fn regression_arguments_generate_an_optional_plot_with_a_path_containing_spaces() {
    let file = common::TempPlot::new();
    let output = Command::new(env!("CARGO_BIN_EXE_neural"))
        .arg("regression")
        .arg(demo_file("regression_train.csv"))
        .arg(demo_file("regression_test.csv"))
        .args(["10", "4", "--plot"])
        .arg(&file.path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Loss plot saved"));
    let svg = std::fs::read_to_string(&file.path).unwrap();
    assert!(svg.contains("id=\"training-loss\""));
    assert!(svg.contains("id=\"test-loss\""));
}

/// Checks that regression infers multiple input features instead of assuming a single input.
#[test]
fn regression_derives_input_count_from_custom_csv_data() {
    let output = Command::new(env!("CARGO_BIN_EXE_neural"))
        .arg("regression")
        .arg(demo_file("classification_train.csv"))
        .arg(demo_file("classification_test.csv"))
        .args(["10", "4"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Prediction for [-2.2, -2.1]"));
}

/// Checks that an explicit classification command trains and reports the expected class decisions.
#[test]
fn classification_arguments_select_classes_epochs_and_batch_size() {
    let output = Command::new(env!("CARGO_BIN_EXE_neural"))
        .arg("classification")
        .arg(demo_file("classification_train.csv"))
        .arg(demo_file("classification_test.csv"))
        .args(["3", "200", "4"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Test accuracy: 100.00%"));
    assert!(!stdout.contains("Regression"));
}

/// Checks rejection of invalid counts, missing values, duplicate flags, and unsupported arguments.
#[test]
fn invalid_arguments_fail_before_loading_files_or_writing_a_plot() {
    for args in [
        vec!["unknown"],
        vec!["regression"],
        vec!["regression", "missing", "missing", "0"],
        vec!["regression", "missing", "missing", "10", "0"],
        vec!["regression", "missing", "missing", "abc"],
        vec!["classification", "missing", "missing", "1"],
        vec!["regression", "missing", "missing", "--plot"],
        vec!["regression", "missing", "missing", "--plot", "loss.png"],
        vec![
            "regression",
            "missing",
            "missing",
            "--plot",
            "a.svg",
            "--plot",
            "b.svg",
        ],
        vec!["regression", "missing", "missing", "--unknown"],
        vec!["--plot", "a.svg"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_neural"))
            .args(&args)
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "Accepted invalid arguments: {args:?}"
        );
        assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
    }
}

/// Checks that command-line help advertises training counts and optional plot output.
#[test]
fn help_lists_arguments_and_plot_option() {
    let output = Command::new(env!("CARGO_BIN_EXE_neural"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[EPOCHS] [BATCH]"));
    assert!(stdout.contains("--plot FILE.svg"));
}
