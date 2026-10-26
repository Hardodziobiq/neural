use crate::{
    AppResult,
    cli::{self, Command, Options},
    report,
};
use neural::net::{
    activation_function::ActivationFunction as Activation,
    data_set::{Dataset, load_classification, load_regression},
    history::SnapshotPolicy,
    loss::Loss,
    neural_net::NeuralNet,
};
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Copy, Debug)]
pub enum Task {
    Regression,
    Classification(usize),
}

impl Task {
    /// Returns the task name used in configuration files, summaries, and report headings.
    pub fn name(self) -> &'static str {
        match self {
            Self::Regression => "regression",
            Self::Classification(_) => "classification",
        }
    }
    /// Returns the classification class count, or one output target for regression.
    pub fn classes(self) -> usize {
        match self {
            Self::Regression => 1,
            Self::Classification(count) => count,
        }
    }
}

pub struct Settings {
    pub task: Task,
    pub layers: Vec<usize>,
    pub output: Activation,
    pub loss: Loss,
    pub learning_rate: f64,
    pub snapshot_every: usize,
}

#[derive(Clone)]
pub struct Summary {
    pub train_loss: f64,
    pub test_loss: f64,
    pub test_mse: Option<f64>,
    pub accuracy: Option<f64>,
    pub seconds: f64,
}

/// Loads and prepares data, trains a fresh seeded model, then evaluates it and saves requested reports.
/// Returns final metrics; the test set is never passed to the training loop.
pub fn run(options: Options, task: Task) -> AppResult<Summary> {
    let mut train = load(&options.train_path, task)?;
    let mut test = load(&options.test_path, task)?;
    let inputs = train.features.first().map_or(0, Vec::len);
    train.validate(inputs, 1)?;
    test.validate(inputs, 1)?;
    if let Task::Classification(classes) = task {
        normalize_classification_labels(&mut train, &mut test, classes)?;
    }
    let original_train = train.clone();
    let original_test = test.clone();
    let settings = settings(&options, task, train.features.len())?;
    if let Task::Classification(classes) = task {
        train.one_hot_encode_targets(classes)?;
        test.one_hot_encode_targets(classes)?;
        if settings.output == Activation::Sigmoid {
            for target in train.targets.iter_mut().chain(&mut test.targets) {
                *target = vec![target[1]];
            }
        }
    }
    let features = train.standardize_features()?;
    features.transform(&mut test.features)?;
    let targets = if matches!(task, Task::Regression) {
        let scaler = train.standardize_targets()?;
        scaler.transform(&mut test.targets)?;
        Some(scaler)
    } else {
        None
    };
    let mut net = NeuralNet::new(
        options.seed,
        inputs,
        options.bias,
        settings.layers.clone(),
        options.activation,
        settings.output,
        settings.loss,
        settings.learning_rate,
    )?;
    let policy = if options.out_dir.is_some() {
        SnapshotPolicy::EveryIterations(settings.snapshot_every)
    } else {
        SnapshotPolicy::Disabled
    };
    let start = Instant::now();
    // The official test set is never passed into training or used for model selection.
    let mut history =
        net.train_with_history(&train, None, options.epochs, options.batch_size, policy)?;
    let seconds = start.elapsed().as_secs_f64();
    if options.out_dir.is_some() {
        let last = history
            .iterations
            .last()
            .ok_or("Training produced no updates")?
            .iteration;
        if history
            .parameter_snapshots
            .last()
            .map(|snapshot| snapshot.iteration)
            != Some(last)
        {
            history.record_parameters(last, net.weights_snapshot(), net.biases_snapshot());
        }
    }
    let mut summary = Summary {
        train_loss: net.test(&train)?,
        test_loss: net.test(&test)?,
        test_mse: None,
        accuracy: None,
        seconds,
    };
    let predict = |input: &[f64]| -> AppResult<Vec<f64>> {
        let output = net.predict(&features.transform_sample(input)?)?;
        match &targets {
            Some(scaler) => Ok(scaler.inverse_transform_sample(&output)?),
            None => Ok(output),
        }
    };
    if matches!(task, Task::Regression) {
        let mut mse = 0.0;
        for (input, target) in original_test.features.iter().zip(&original_test.targets) {
            // Evaluation MSE remains a mean, independent of the lecture's training-loss scaling.
            let prediction = predict(input)?;
            mse += prediction
                .iter()
                .zip(target)
                .map(|(prediction, target)| (prediction - target).powi(2))
                .sum::<f64>()
                / target.len() as f64
                / original_test.features.len() as f64;
        }
        summary.test_mse = Some(mse);
        println!("Regression");
        println!("Training loss (standardized): {:.6}", summary.train_loss);
        println!("Test loss (standardized): {:.6}", summary.test_loss);
        // Scalar regression MSE is twice the lecture loss, in standardized units.
        if settings.loss == Loss::MeanSquaredError {
            println!("Test MSE (standardized): {:.6}", 2.0 * summary.test_loss);
        }
        println!("Test MSE (original units): {mse:.6}");
        let sample = &original_test.features[0];
        println!("Prediction for {sample:?}: {:.6}", predict(sample)?[0]);
    } else {
        let mut correct = 0;
        for (input, target) in original_test.features.iter().zip(&original_test.targets) {
            correct +=
                usize::from(class_index(&predict(input)?, task.classes())? == target[0] as usize);
        }
        let accuracy = correct as f64 / original_test.features.len() as f64;
        summary.accuracy = Some(accuracy);
        println!("Classification");
        println!("Training loss: {:.6}", summary.train_loss);
        println!("Test loss: {:.6}", summary.test_loss);
        println!("Test accuracy: {:.2}%", accuracy * 100.0);
    }
    println!(
        "Updates: {}, seed: {}, training seconds: {seconds:.3}",
        history.iterations.len(),
        net.seed()
    );
    if let Some(path) = &options.plot {
        history.save_loss_plot(path)?;
        println!("Loss plot saved to {}", path.display());
    }
    if let Some(directory) = &options.out_dir {
        report::save(
            directory,
            &options,
            &settings,
            &summary,
            &history,
            (&original_train, &original_test),
            predict,
        )?;
        println!("Report saved to {}", directory.join("index.html").display());
    }
    Ok(summary)
}

/// Infers one-based labels only when training contains class K and no zero, then maps both splits consistently.
/// Ambiguous training subsets keep zero-based labels; evaluation labels never determine the mapping.
fn normalize_classification_labels(
    train: &mut Dataset,
    test: &mut Dataset,
    classes: usize,
) -> AppResult<()> {
    let one_based = train.targets.iter().any(|label| label[0] == classes as f64)
        && train.targets.iter().all(|label| label[0] >= 1.0);
    let offset = if one_based { 1.0 } else { 0.0 };
    for (split, dataset) in [("Training", &*train), ("Evaluation", &*test)] {
        if dataset.targets.iter().any(|label| {
            let value = label[0] - offset;
            value < 0.0 || value >= classes as f64 || value.fract() != 0.0
        }) {
            return Err(format!(
                "{split} labels must be in {}..{}; label numbering is inferred from training data only",
                offset as usize,
                classes - 1 + offset as usize
            )
            .into());
        }
    }
    if one_based {
        for label in train.targets.iter_mut().chain(&mut test.targets) {
            label[0] -= 1.0;
        }
    }
    Ok(())
}

/// Resolves defaults, checks output/loss compatibility, and adds the output layer and snapshot interval.
fn settings(options: &Options, task: Task, samples: usize) -> AppResult<Settings> {
    let regression = matches!(task, Task::Regression);
    let loss = options.loss.unwrap_or(if regression {
        Loss::MeanSquaredError
    } else {
        Loss::CategorialCrossEntropy
    });
    let binary = !regression && loss == Loss::BinaryCrossEntropy;
    let output = options.output.unwrap_or(if regression {
        Activation::Linear
    } else if binary {
        Activation::Sigmoid
    } else {
        Activation::SoftMax
    });
    if regression
        && (!matches!(loss, Loss::MeanSquaredError | Loss::MeanAbsoluteError)
            || output != Activation::Linear)
    {
        return Err("Standardized regression requires linear output and mse or mae loss".into());
    }
    if !regression {
        if task.classes() < 2 {
            return Err("Classification requires at least two classes".into());
        }
        if binary && (task.classes() != 2 || output != Activation::Sigmoid) {
            return Err("bce requires two classes and sigmoid output".into());
        }
        if !binary
            && (!matches!(loss, Loss::MeanSquaredError | Loss::CategorialCrossEntropy)
                || output != Activation::SoftMax)
        {
            return Err(
                "Multiclass classification requires softmax output and cce or mse loss".into(),
            );
        }
    }
    let mut layers =
        options
            .hidden
            .clone()
            .unwrap_or(if regression { vec![16, 16] } else { vec![8] });
    layers.push(if binary || regression {
        1
    } else {
        task.classes()
    });
    let updates = samples
        .div_ceil(options.batch_size)
        .checked_mul(options.epochs)
        .ok_or("Update count is too large")?;
    Ok(Settings {
        task,
        layers,
        output,
        loss,
        learning_rate: options
            .learning_rate
            .unwrap_or(if regression { 0.03 } else { 0.05 }),
        snapshot_every: options
            .snapshot_every
            .unwrap_or(updates.div_ceil(200).max(1)),
    })
}

/// Loads a task-specific CSV and represents its last column as one scalar target per sample.
fn load(path: &str, task: Task) -> AppResult<Dataset> {
    match task {
        Task::Regression => Ok(load_regression(path)?),
        Task::Classification(_) => {
            let (features, labels) = load_classification(path)?;
            Ok(Dataset {
                features,
                targets: labels.into_iter().map(|label| vec![label as f64]).collect(),
            })
        }
    }
}

/// Selects a binary class at the 0.5 threshold or the largest multiclass output score.
pub fn class_index(output: &[f64], classes: usize) -> AppResult<usize> {
    if classes == 2 && output.len() == 1 {
        return Ok(usize::from(output[0] >= 0.5));
    }
    if output.len() != classes {
        return Err("Class score count does not match class count".into());
    }
    Ok(output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .ok_or("No class scores")?
        .0)
}

/// Runs independent models from a TSV plan and saves a summary, continuing after individual run failures.
/// Returns an error after saving the summary if any run failed.
pub fn run_suite(path: &Path, summary_path: &Path) -> AppResult<()> {
    let contents = fs::read_to_string(path)?;
    let root = path.parent().unwrap_or(Path::new("."));
    let mut lines = contents
        .trim_start_matches('\u{feff}')
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty() && !line.starts_with('#'));
    let (_, header) = lines.next().ok_or("Experiment plan is empty")?;
    let columns: Vec<_> = header.split('\t').collect();
    let allowed = [
        "task",
        "train",
        "test",
        "classes",
        "hidden",
        "activation",
        "output",
        "loss",
        "learning_rate",
        "seed",
        "bias",
        "epochs",
        "batch",
        "snapshot_every",
        "weight_limit",
        "out_dir",
    ];
    let mut seen = std::collections::HashSet::new();
    for column in &columns {
        if !allowed.contains(column) || !seen.insert(*column) {
            return Err(format!("Unknown or duplicate plan column: {column}").into());
        }
    }
    for required in ["task", "train", "test", "out_dir"] {
        if !columns.contains(&required) {
            return Err(format!("Missing plan column: {required}").into());
        }
    }
    let mut experiments = Vec::new();
    let mut directories = std::collections::HashSet::new();
    let mut dataset_pair = None;
    for (line_number, line) in lines {
        let values: Vec<_> = line.split('\t').collect();
        if values.len() != columns.len() {
            return Err(format!("Plan line {}: column count mismatch", line_number + 1).into());
        }
        let get = |name: &str| {
            columns
                .iter()
                .position(|&column| column == name)
                .map(|index| values[index])
                .unwrap_or("")
        };
        for required in ["task", "train", "test", "out_dir"] {
            if get(required).is_empty() {
                return Err(format!("Plan line {}: empty {required}", line_number + 1).into());
            }
        }
        let mut args = vec![
            get("task").to_owned(),
            rooted(root, get("train"))?,
            rooted(root, get("test"))?,
        ];
        if get("task") == "classification" {
            args.push(get("classes").to_owned());
        }
        args.push(
            if get("epochs").is_empty() {
                "500"
            } else {
                get("epochs")
            }
            .into(),
        );
        args.push(
            if get("batch").is_empty() {
                "16"
            } else {
                get("batch")
            }
            .into(),
        );
        for (&column, &value) in columns.iter().zip(&values) {
            if value.is_empty()
                || ["task", "train", "test", "classes", "epochs", "batch"].contains(&column)
            {
                continue;
            }
            args.push(format!("--{}", column.replace('_', "-")));
            args.push(if column == "out_dir" {
                rooted(root, value)?
            } else {
                value.into()
            });
        }
        let (options, task) = match cli::parse(&args)? {
            Command::Regression(options) => (options, Task::Regression),
            Command::Classification(options, classes) => (options, Task::Classification(classes)),
            _ => return Err("Plan supports only regression and classification".into()),
        };
        let train_path = Path::new(&options.train_path);
        let test_path = Path::new(&options.test_path);
        let pair = (
            fs::canonicalize(train_path).unwrap_or(std::path::absolute(train_path)?),
            fs::canonicalize(test_path).unwrap_or(std::path::absolute(test_path)?),
            task.name(),
            task.classes(),
        );
        if dataset_pair
            .as_ref()
            .is_some_and(|previous| previous != &pair)
        {
            return Err(
                "Every plan row must use the same training/evaluation files, task, and class count"
                    .into(),
            );
        }
        dataset_pair = Some(pair);
        let directory = options.out_dir.clone().ok_or("Plan requires out_dir")?;
        if !directories.insert(directory) {
            return Err("Every plan row must have a distinct out_dir".into());
        }
        experiments.push((line_number + 1, options, task));
    }
    if experiments.is_empty() {
        return Err("Experiment plan has no runs".into());
    }
    let mut csv = String::from(
        "row,task,seed,status,train_loss,test_loss,test_mse,test_accuracy,training_seconds,report_dir,error\n",
    );
    let mut failures = 0;
    let mut suite_runs = Vec::new();
    for (row, options, task) in experiments {
        println!("\nExperiment plan row {row}");
        let directory = options
            .out_dir
            .as_ref()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let seed = options.seed;
        let result = run(options.clone(), task);
        match &result {
            Ok(result) => csv.push_str(&report::csv_row(&[
                row.to_string(),
                task.name().into(),
                seed.to_string(),
                "ok".into(),
                result.train_loss.to_string(),
                result.test_loss.to_string(),
                result.test_mse.map(|v| v.to_string()).unwrap_or_default(),
                result.accuracy.map(|v| v.to_string()).unwrap_or_default(),
                result.seconds.to_string(),
                directory,
                String::new(),
            ])),
            Err(error) => {
                failures += 1;
                eprintln!("Experiment row {row} failed: {error}");
                csv.push_str(&report::csv_row(&[
                    row.to_string(),
                    task.name().into(),
                    seed.to_string(),
                    "error".into(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    directory,
                    error.to_string(),
                ]));
            }
        }
        suite_runs.push(report::SuiteRun {
            row,
            task,
            options,
            result: result.map_err(|error| error.to_string()),
        });
    }
    if let Some(parent) = summary_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(summary_path, csv)?;
    println!("Experiment summary saved to {}", summary_path.display());
    let html_path = report::save_suite(summary_path, path, &suite_runs)?;
    println!("Batch report saved to {}", html_path.display());
    if failures > 0 {
        return Err(format!("{failures} experiments failed; see summary CSV").into());
    }
    Ok(())
}

/// Resolves relative dataset and output paths against the plan directory, preserving absolute paths.
fn rooted(root: &Path, value: &str) -> Result<String, Box<dyn Error>> {
    let path = PathBuf::from(value);
    Ok(if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
    .to_str()
    .ok_or("Path is not UTF-8")?
    .into())
}
