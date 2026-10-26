mod common;

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn demo(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data/demo")
        .join(name)
}

fn regression(directory: &Path, output: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_neural"));
    command
        .current_dir(directory)
        .arg("regression")
        .arg(demo("regression_train.csv"))
        .arg(demo("regression_test.csv"))
        .args(["3", "4", "--out-dir"])
        .arg(output);
    command
}

fn success(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn report_contains_all_charts_histories_and_replay_configuration() {
    let directory = common::TempDir::new();
    let output = directory.path.join("regression report");
    success(
        regression(&directory.path, &output)
            .args([
                "--hidden",
                "4,3",
                "--activation",
                "sigmoid",
                "--loss",
                "mae",
                "--seed",
                "0",
                "--bias",
                "false",
                "--snapshot-every",
                "5",
                "--weight-limit",
                "4",
            ])
            .output()
            .unwrap(),
    );
    for name in [
        "index.html",
        "configuration.tsv",
        "metrics.csv",
        "epochs.csv",
        "iterations.csv",
        "weights.csv",
        "training-data.svg",
        "results.svg",
        "epoch-loss.svg",
        "iteration-loss.svg",
        "weights.svg",
    ] {
        assert!(output.join(name).is_file(), "Missing {name}");
    }
    let iterations = fs::read_to_string(output.join("iterations.csv")).unwrap();
    assert_eq!(iterations.lines().count(), 13); // 13 samples, batches 4+4+4+1, three epochs.
    assert!(iterations.lines().last().unwrap().starts_with("12,3,1,"));
    let epochs = fs::read_to_string(output.join("epochs.csv")).unwrap();
    assert_eq!(epochs.lines().count(), 4);
    assert!(epochs.lines().skip(1).all(|line| line.ends_with(','))); // No official test evaluations during training.
    let weights = fs::read_to_string(output.join("weights.csv")).unwrap();
    let updates: std::collections::BTreeSet<_> = weights
        .lines()
        .skip(1)
        .map(|line| line.split(',').next().unwrap().parse::<usize>().unwrap())
        .collect();
    assert_eq!(updates, [0, 5, 10, 12].into_iter().collect());
    assert!(
        weights
            .lines()
            .filter(|line| line.contains(",bias,,"))
            .all(|line| line.ends_with(",0"))
    );
    let svg = fs::read_to_string(output.join("results.svg")).unwrap();
    assert!(svg.contains("id=\"learned-function\""));
    assert!(svg.contains("original units"));
    let html = fs::read_to_string(output.join("index.html")).unwrap();
    assert!(html.contains("evaluated only after training"));
    let before = fs::read(output.join("iterations.csv")).unwrap();
    success(
        Command::new(env!("CARGO_BIN_EXE_neural"))
            .args(["experiments"])
            .arg(output.join("configuration.tsv"))
            .arg("--summary")
            .arg(directory.path.join("replay.csv"))
            .output()
            .unwrap(),
    );
    assert_eq!(fs::read(output.join("iterations.csv")).unwrap(), before);
}

#[test]
fn reports_are_reproducible_and_test_targets_cannot_influence_weights() {
    let directory = common::TempDir::new();
    let first = directory.path.join("first");
    let second = directory.path.join("second");
    let test = directory.path.join("different test.csv");
    fs::write(&test, "x,y\n-2.75,1000\n2.75,-1000\n").unwrap();
    success(
        regression(&directory.path, &first)
            .args(["--hidden", "none", "--seed", "13"])
            .output()
            .unwrap(),
    );
    success(
        Command::new(env!("CARGO_BIN_EXE_neural"))
            .arg("regression")
            .arg(demo("regression_train.csv"))
            .arg(&test)
            .args(["3", "4", "--hidden", "none", "--seed", "13", "--out-dir"])
            .arg(&second)
            .output()
            .unwrap(),
    );
    for name in ["iterations.csv", "epochs.csv", "weights.csv"] {
        assert_eq!(
            fs::read(first.join(name)).unwrap(),
            fs::read(second.join(name)).unwrap(),
            "Test data changed {name}"
        );
    }
    assert_ne!(
        fs::read(first.join("metrics.csv")).unwrap(),
        fs::read(second.join("metrics.csv")).unwrap()
    );
}

#[test]
fn classification_report_visualizes_true_labels_decision_regions_and_errors() {
    let directory = common::TempDir::new();
    let output = directory.path.join("classification");
    success(
        Command::new(env!("CARGO_BIN_EXE_neural"))
            .arg("classification")
            .arg(demo("classification_train.csv"))
            .arg(demo("classification_test.csv"))
            .args(["3", "2", "4", "--learning-rate", "0", "--out-dir"])
            .arg(&output)
            .output()
            .unwrap(),
    );
    let svg = fs::read_to_string(output.join("results.svg")).unwrap();
    assert!(svg.contains("decision regions"));
    assert!(svg.matches("opacity=\"0.18\"").count() >= 2500);
    assert!(svg.contains("Black border: error"));
    assert!(svg.contains("stroke-width=\"3\""));
    let labels = fs::read_to_string(output.join("training-data.svg")).unwrap();
    assert!(labels.contains("Training true class"));
    assert!(labels.contains("Test true class"));
}

#[test]
fn binary_cross_entropy_uses_one_sigmoid_output_without_one_hot_mismatch() {
    let directory = common::TempDir::new();
    let train = directory.path.join("binary.csv");
    fs::write(&train, "x1,x2,class\n-2,-1,0\n-1,-2,0\n1,2,1\n2,1,1\n").unwrap();
    let output = directory.path.join("binary");
    let result = Command::new(env!("CARGO_BIN_EXE_neural"))
        .arg("classification")
        .arg(&train)
        .arg(&train)
        .args([
            "2",
            "100",
            "4",
            "--loss",
            "bce",
            "--hidden",
            "none",
            "--out-dir",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&result.stdout).contains("Test accuracy: 100.00%"));
    success(result);
    let configuration = fs::read_to_string(output.join("configuration.tsv")).unwrap();
    assert!(configuration.contains("\tsigmoid\tbce\t"));
}

#[test]
fn suite_runs_independent_rows_with_relative_paths_and_writes_summary() {
    let directory = common::TempDir::new();
    fs::copy(
        demo("regression_train.csv"),
        directory.path.join("train.csv"),
    )
    .unwrap();
    fs::copy(demo("regression_test.csv"), directory.path.join("test.csv")).unwrap();
    let plan = directory.path.join("plan.tsv");
    fs::write(&plan, "task\ttrain\ttest\thidden\tactivation\tloss\tseed\tepochs\tbatch\tout_dir\nregression\ttrain.csv\ttest.csv\tnone\ttanh\tmse\t0\t2\t4\tfirst run\nregression\ttrain.csv\ttest.csv\t3,3,3,3\trelu\tmae\t1\t2\t4\tsecond run\n").unwrap();
    let summary = directory.path.join("summary.csv");
    success(
        Command::new(env!("CARGO_BIN_EXE_neural"))
            .arg("experiments")
            .arg(&plan)
            .arg("--summary")
            .arg(&summary)
            .output()
            .unwrap(),
    );
    assert_eq!(
        fs::read_to_string(summary)
            .unwrap()
            .matches("\"ok\"")
            .count(),
        2
    );
    assert!(directory.path.join("first run/index.html").exists());
    assert!(directory.path.join("second run/index.html").exists());
}

#[test]
fn suite_continues_after_a_failed_run_and_marks_it_in_summary() {
    let directory = common::TempDir::new();
    let plan = directory.path.join("plan.tsv");
    fs::write(&plan, format!("task\ttrain\ttest\tepochs\tout_dir\nregression\tmissing.csv\tmissing.csv\t1\tfailed\nregression\t{}\t{}\t1\tpassed\n",
        demo("regression_train.csv").display(), demo("regression_test.csv").display())).unwrap();
    let summary = directory.path.join("summary.csv");
    let output = Command::new(env!("CARGO_BIN_EXE_neural"))
        .arg("experiments")
        .arg(&plan)
        .arg("--summary")
        .arg(&summary)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let csv = fs::read_to_string(summary).unwrap();
    assert!(csv.contains("\"error\""));
    assert!(csv.contains("\"ok\""));
    assert!(directory.path.join("passed/index.html").exists());
}

#[test]
fn invalid_options_and_duplicate_report_directories_are_rejected() {
    let directory = common::TempDir::new();
    for flags in [
        vec!["--hidden", "3,0"],
        vec!["--activation", "unknown"],
        vec!["--seed", "-1"],
        vec!["--learning-rate", "NaN"],
        vec!["--bias", "maybe"],
        vec!["--weight-limit", "0"],
        vec!["--snapshot-every", "0"],
        vec!["--output", "sigmoid"],
        vec!["--loss", "cce"],
    ] {
        let output = regression(&directory.path, &directory.path.join("invalid"))
            .args(flags)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(!directory.path.join("invalid").exists());
    }
    let plan = directory.path.join("duplicate.tsv");
    fs::write(&plan, "task\ttrain\ttest\tout_dir\nregression\tmissing\tmissing\tsame\nregression\tmissing\tmissing\tsame\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_neural"))
        .arg("experiments")
        .arg(plan)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("distinct out_dir"));
}
