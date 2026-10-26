mod common;

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

/// Returns an absolute path to a bundled dataset used by experiment tests.
fn demo(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

/// Builds a short regression command with isolated report output for an integration test.
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

/// Requires a subprocess to succeed and includes its standard error when the assertion fails.
fn success(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Checks all report artifacts, sampled parameter updates, and exact iteration-history reproduction on replay.
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

/// Checks that replacing evaluation targets changes metrics but not learned weights or training histories.
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

/// Checks that 2D reports include class regions, true labels, and visibly marked test mistakes.
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

/// Checks the scalar-target sigmoid path for two-class BCE and its saved configuration.
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

/// Checks TSV plans with relative paths, different architectures, and separate successful report directories.
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
    let html = fs::read_to_string(directory.path.join("summary.html")).unwrap();
    assert!(html.contains("Successful: 2. Failed: 0."));
    assert_eq!(html.matches("class=\"dataset\"").count(), 1);
    assert_eq!(html.matches("<svg xmlns=").count(), 3);
    assert_eq!(html.matches("<h2>Shared dataset</h2>").count(), 1);
    assert!(html.contains("id=\"batch-epoch-config-1\""));
    assert!(html.contains("id=\"batch-epoch-config-2\""));
    assert!(html.contains("id=\"batch-iteration-config-1\""));
    assert!(!html.contains("Weight and bias comparisons"));
    assert!(!html.contains("learned-function"));
    assert!(html.contains("first%20run/index.html"));
    assert!(html.contains("hidden=none"));
    assert!(html.contains("hidden=3,3,3,3"));
    assert!(html.contains("learning_rate=0.03"));
}

/// Checks that a failed experiment is recorded while a later valid experiment still runs.
#[test]
fn suite_continues_after_a_failed_run_and_marks_it_in_summary() {
    let directory = common::TempDir::new();
    let plan = directory.path.join("plan.tsv");
    let train = demo("regression_train.csv");
    let test = demo("regression_test.csv");
    fs::write(&plan, format!("task\ttrain\ttest\tepochs\toutput\tout_dir\nregression\t{}\t{}\t1\tsigmoid\tfailed\nregression\t{}\t{}\t1\tlinear\tpassed\n",
        train.display(), test.display(), train.display(), test.display())).unwrap();
    let summary = directory.path.join("summary.csv");
    fs::create_dir(directory.path.join("failed")).unwrap();
    fs::write(directory.path.join("failed/index.html"), "stale report").unwrap();
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
    let html = fs::read_to_string(directory.path.join("summary.html")).unwrap();
    assert!(html.contains("Successful: 1. Failed: 1."));
    assert!(html.contains("class=\"error\""));
    assert!(!html.contains("href=\"failed/index.html\""));
    assert_eq!(html.matches("<svg xmlns=").count(), 3);
}

/// Checks URL encoding of dataset/report names and valid linking from a summary in another directory.
#[test]
fn suite_html_escapes_names_and_links_all_artifacts() {
    let directory = common::TempDir::new();
    let dataset = directory.path.join("train & data.csv");
    fs::copy(demo("regression_train.csv"), &dataset).unwrap();
    let plan = directory.path.join("plan.tsv");
    fs::write(&plan, "task\ttrain\ttest\tepochs\tout_dir\nregression\ttrain & data.csv\ttrain & data.csv\t1\treport #ą\n").unwrap();
    let summary = directory.path.join("summaries/result.csv");
    success(
        Command::new(env!("CARGO_BIN_EXE_neural"))
            .arg("experiments")
            .arg(&plan)
            .arg("--summary")
            .arg(summary)
            .output()
            .unwrap(),
    );
    let html = fs::read_to_string(directory.path.join("summaries/result.html")).unwrap();
    assert!(html.contains("train &amp; data.csv"));
    assert!(html.contains("../report%20%23%C4%85/index.html"));
    assert_eq!(html.matches("<svg xmlns=").count(), 3);
    assert!(!html.contains("<img"));
    assert!(html.contains("href=\"result.csv\""));
}

/// Checks rejection of invalid model flags and experiment plans that reuse an output directory.
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

/// Checks that evaluation MSE stays in original units and is independent of the selected training loss.
#[test]
fn regression_metric_is_distinct_from_lecture_loss() {
    let directory = common::TempDir::new();
    let train = directory.path.join("train.csv");
    let test = directory.path.join("test.csv");
    fs::write(&train, "x,y\n-1,-1\n0,1\n1,3\n").unwrap();
    fs::write(&test, "x,y\n0,2\n").unwrap();
    let mut metrics = Vec::new();
    for loss in ["mse", "mae"] {
        let report = directory.path.join(loss);
        success(
            Command::new(env!("CARGO_BIN_EXE_neural"))
                .arg("regression")
                .arg(&train)
                .arg(&test)
                .args([
                    "1",
                    "2",
                    "--hidden",
                    "none",
                    "--learning-rate",
                    "0",
                    "--loss",
                    loss,
                    "--out-dir",
                ])
                .arg(&report)
                .output()
                .unwrap(),
        );
        let csv = fs::read_to_string(report.join("metrics.csv")).unwrap();
        let values: Vec<f64> = csv
            .lines()
            .nth(1)
            .unwrap()
            .split(',')
            .take(3)
            .map(|value| value.trim_matches('"').parse().unwrap())
            .collect();
        metrics.push(values);
    }
    // Targets [-1, 1, 3] have population variance 8/3. At x=0, initial prediction is 1.
    assert!((metrics[0][2] - 1.0).abs() < 1e-12);
    assert!((metrics[0][2] - 2.0 * metrics[0][1] * (8.0 / 3.0)).abs() < 1e-12);
    assert_eq!(metrics[0][2], metrics[1][2]);
}

/// Checks equivalent zero- and one-based classification, including evaluation splits missing the highest class.
#[test]
fn one_based_labels_match_zero_based_training_and_reports() {
    let directory = common::TempDir::new();
    for (classes, loss) in [(2, "cce"), (2, "mse"), (2, "bce"), (3, "cce"), (3, "mse")] {
        let mut previous = None;
        for offset in [0, 1] {
            let train = directory
                .path
                .join(format!("train-{classes}-{loss}-{offset}.csv"));
            let test = directory
                .path
                .join(format!("test-{classes}-{loss}-{offset}.csv"));
            let mut training = String::from("x,class\n");
            for label in 0..classes {
                training.push_str(&format!("{},{}\n", label, label + offset));
            }
            fs::write(&train, training).unwrap();
            fs::write(&test, format!("x,class\n0,{offset}\n")).unwrap();
            let report = directory
                .path
                .join(format!("report-{classes}-{loss}-{offset}"));
            success(
                Command::new(env!("CARGO_BIN_EXE_neural"))
                    .arg("classification")
                    .arg(&train)
                    .arg(&test)
                    .arg(classes.to_string())
                    .args(["3", "2", "--loss", loss, "--out-dir"])
                    .arg(&report)
                    .output()
                    .unwrap(),
            );
            let metrics = fs::read_to_string(report.join("metrics.csv")).unwrap();
            let result = (
                metrics
                    .lines()
                    .nth(1)
                    .unwrap()
                    .split(',')
                    .take(4)
                    .collect::<Vec<_>>()
                    .join(","),
                fs::read(report.join("iterations.csv")).unwrap(),
                fs::read(report.join("weights.csv")).unwrap(),
            );
            if let Some(previous) = &previous {
                assert_eq!(&result, previous);
            }
            previous = Some(result);
        }
    }
}

/// Checks invalid or inconsistent numbering without allowing evaluation labels to select the training mapping.
#[test]
fn inconsistent_class_label_numbering_is_rejected() {
    let directory = common::TempDir::new();
    for (index, (training, evaluation)) in [
        ("x,class\n0,0\n1,1\n", "x,class\n0,2\n"),
        ("x,class\n0,1\n1,2\n", "x,class\n0,0\n"),
        ("x,class\n0,0\n1,2\n", "x,class\n0,0\n"),
    ]
    .iter()
    .enumerate()
    {
        let train = directory.path.join(format!("train-{index}.csv"));
        let test = directory.path.join(format!("test-{index}.csv"));
        let report = directory.path.join(format!("report-{index}"));
        fs::write(&train, training).unwrap();
        fs::write(&test, evaluation).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_neural"))
            .arg("classification")
            .arg(train)
            .arg(test)
            .args(["2", "1", "2", "--out-dir"])
            .arg(&report)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("label numbering"));
        assert!(!report.exists());
    }
}

/// Checks that mixed file pairs, tasks, or class counts are rejected before any training/report output.
#[test]
fn suite_rejects_mixed_dataset_pairs_before_training() {
    let directory = common::TempDir::new();
    let train = demo("classification_train.csv");
    let test = demo("classification_test.csv");
    for (index, second) in [
        format!("classification\t{}\t{}\t3", test.display(), test.display()),
        format!("classification\t{}\t{}\t2", train.display(), test.display()),
        format!("regression\t{}\t{}\t", train.display(), test.display()),
    ]
    .iter()
    .enumerate()
    {
        let plan = directory.path.join(format!("mixed-{index}.tsv"));
        fs::write(&plan, format!("task\ttrain\ttest\tclasses\tepochs\tout_dir\nclassification\t{}\t{}\t3\t1\tfirst-{index}\n{second}\t1\tsecond-{index}\n", train.display(), test.display())).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_neural"))
            .arg("experiments")
            .arg(plan)
            .arg("--summary")
            .arg(directory.path.join(format!("mixed-{index}.csv")))
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("same training/evaluation files"));
        assert!(!directory.path.join(format!("first-{index}")).exists());
        assert!(!directory.path.join(format!("mixed-{index}.csv")).exists());
    }
}

/// Checks seed means, sample deviations, duplicate-seed exclusion, and plotted mean coordinates against raw histories.
#[test]
fn suite_averages_distinct_seeds_and_plots_the_mean_history() {
    let directory = common::TempDir::new();
    let plan = directory.path.join("seeds.tsv");
    let train = demo("regression_train.csv");
    let test = demo("regression_test.csv");
    let mut contents = String::from("task\ttrain\ttest\thidden\tepochs\tbatch\tseed\tout_dir\n");
    for (index, seed) in [1, 2, 1].iter().enumerate() {
        contents.push_str(&format!(
            "regression\t{}\t{}\tnone\t2\t4\t{seed}\trun-{index}\n",
            train.display(),
            test.display()
        ));
    }
    fs::write(&plan, contents).unwrap();
    success(
        Command::new(env!("CARGO_BIN_EXE_neural"))
            .arg("experiments")
            .arg(&plan)
            .arg("--summary")
            .arg(directory.path.join("summary.csv"))
            .output()
            .unwrap(),
    );
    let html = fs::read_to_string(directory.path.join("summary.html")).unwrap();
    assert!(html.contains("Successful: 3. Failed: 0."));
    assert!(html.contains("n=2: 1, 2"));
    assert!(!html.contains("batch-epoch-config-2"));
    assert_eq!(html.matches("<polygon").count(), 2);
    let histories: Vec<Vec<(f64, f64)>> = (0..2)
        .map(|index| {
            fs::read_to_string(directory.path.join(format!("run-{index}/epochs.csv")))
                .unwrap()
                .lines()
                .skip(1)
                .map(|line| {
                    let values: Vec<_> = line.split(',').collect();
                    (values[0].parse().unwrap(), values[1].parse().unwrap())
                })
                .collect()
        })
        .collect();
    let means: Vec<_> = histories[0]
        .iter()
        .zip(&histories[1])
        .map(|(&(x, a), &(_, b))| (x, (a + b) / 2.0, (a - b).abs() / 2.0_f64.sqrt()))
        .collect();
    let last = means.last().unwrap();
    assert!(html.contains(&format!("{:.6} ± {:.6}", last.1, last.2)));
    let low = means
        .iter()
        .map(|p| (p.1 - p.2).max(0.0))
        .fold(f64::INFINITY, f64::min);
    let high = means
        .iter()
        .map(|p| p.1 + p.2)
        .fold(f64::NEG_INFINITY, f64::max);
    let padding = (high - low) * 0.05;
    let low = low - padding;
    let high = high + padding;
    let coordinates = html
        .split("id=\"batch-epoch-config-1\" points=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap();
    let plotted: Vec<Vec<f64>> = coordinates
        .split_whitespace()
        .map(|point| point.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(plotted.len(), means.len());
    for (point, expected) in plotted.iter().zip(means) {
        let y = 470.0 - (expected.1 - low) / (high - low) * 400.0;
        assert!((point[1] - y).abs() <= 0.00501);
    }
}

/// Checks that an entirely failed batch still saves an error overview without attempting empty comparison plots.
#[test]
fn suite_with_no_successful_runs_still_writes_html() {
    let directory = common::TempDir::new();
    let plan = directory.path.join("failed.tsv");
    fs::write(
        &plan,
        "task\ttrain\ttest\tout_dir\nregression\tmissing.csv\tmissing.csv\tfailed\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_neural"))
        .arg("experiments")
        .arg(plan)
        .arg("--summary")
        .arg(directory.path.join("summary.csv"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let html = fs::read_to_string(directory.path.join("summary.html")).unwrap();
    assert!(html.contains("Successful: 0. Failed: 1."));
    assert!(html.contains("No successful runs are available"));
    assert!(!html.contains("<svg"));
}
