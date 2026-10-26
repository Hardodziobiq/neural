use crate::{
    AppResult,
    charts::{self, COLORS, Chart},
    cli::Options,
    experiment::{Settings, Summary, Task, class_index},
};
use neural::net::{
    activation_function::ActivationFunction as Activation, data_set::Dataset, history::History,
    loss::Loss,
};
use std::{fmt::Write, fs, path::Path};

pub fn csv_row(values: &[String]) -> String {
    values
        .iter()
        .map(|value| format!("\"{}\"", value.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(",")
        + "\n"
}

pub fn save(
    directory: &Path,
    options: &Options,
    settings: &Settings,
    summary: &Summary,
    history: &History,
    datasets: (&Dataset, &Dataset),
    predict: impl Fn(&[f64]) -> AppResult<Vec<f64>>,
) -> AppResult<()> {
    let (train, test) = datasets;
    fs::create_dir_all(directory)?;
    let configuration = configuration(directory, options, settings)?;
    fs::write(directory.join("configuration.tsv"), &configuration)?;
    history.save_loss_plot(directory.join("epoch-loss.svg"))?;
    let iteration_points: Vec<_> = history
        .iterations
        .iter()
        .map(|row| (row.iteration as f64, row.batch_loss))
        .collect();
    let mut iteration_bounds = charts::bounds(&iteration_points)?;
    iteration_bounds[0] = 0.0;
    iteration_bounds[2] = iteration_bounds[2].max(0.0);
    let mut chart = Chart::new(
        "Batch loss before each update",
        "Parameter update",
        "Mean batch loss",
        iteration_bounds,
    );
    chart.legend("Batch loss", COLORS[0]);
    chart.line(&iteration_points, COLORS[0], "iteration-loss");
    fs::write(directory.join("iteration-loss.svg"), chart.finish())?;
    save_weights(directory, history, options.weight_limit)?;
    save_history(directory, history)?;
    let result_plots = match settings.task {
        Task::Regression if train.features[0].len() == 1 => {
            regression_plots(directory, train, test, &predict)?;
            true
        }
        Task::Classification(classes) if train.features[0].len() == 2 => {
            classification_plots(directory, train, test, classes, &predict)?;
            true
        }
        _ => false,
    };
    let metrics = format!(
        "train_loss,test_loss,test_mse_original,test_accuracy,training_seconds,updates\n{}",
        csv_row(&[
            summary.train_loss.to_string(),
            summary.test_loss.to_string(),
            summary.test_mse.map(|v| v.to_string()).unwrap_or_default(),
            summary.accuracy.map(|v| v.to_string()).unwrap_or_default(),
            summary.seconds.to_string(),
            history.iterations.len().to_string()
        ])
    );
    fs::write(directory.join("metrics.csv"), metrics)?;
    let config_lines: Vec<_> = configuration.lines().collect();
    let table = config_lines[0]
        .split('\t')
        .zip(config_lines[1].split('\t'))
        .map(|(key, value)| {
            format!(
                "<tr><th>{}</th><td>{}</td></tr>",
                charts::escape(key),
                charts::escape(value)
            )
        })
        .collect::<String>();
    let mut html = format!(
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Neural network experiment</title><style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#253047}}img{{width:100%;border:1px solid #ddd}}table{{border-collapse:collapse}}td,th{{padding:5px 12px;border:1px solid #ddd;text-align:left}}section{{margin:30px 0}}code{{background:#eee}}a{{color:#2563eb}}</style><h1>{} experiment</h1><p>Training updates: {}. Final training loss: {:.6}. Final test loss: {:.6}. Training time: {:.3} s.</p>",
        settings.task.name(),
        history.iterations.len(),
        summary.train_loss,
        summary.test_loss,
        summary.seconds
    );
    if let Some(mse) = summary.test_mse {
        write!(html, "<p>Test MSE in original units: {mse:.6}.</p>").unwrap();
    }
    if let Some(accuracy) = summary.accuracy {
        write!(html, "<p>Test accuracy: {:.2}%.</p>", accuracy * 100.0).unwrap();
    }
    html.push_str("<p>The test set was evaluated only after training. Epoch loss uses the complete training set after an epoch. Iteration loss uses that update's batch before the update; different batches can have different difficulty.</p>");
    html.push_str(&format!("<details><summary>Configuration and reproduction</summary><p>Replay with <code>neural experiments PATH/TO/configuration.tsv</code>. Data paths are absolute; retain the same data and Cargo.lock.</p><table>{table}</table></details>"));
    for file in [
        "configuration.tsv",
        "metrics.csv",
        "epochs.csv",
        "iterations.csv",
        "weights.csv",
    ] {
        write!(html, "<a href=\"{file}\">{file}</a> &nbsp; ").unwrap();
    }
    let mut plots = vec![
        ("Training loss by epoch", "epoch-loss.svg"),
        ("Batch loss by parameter update", "iteration-loss.svg"),
        ("Selected weights and biases", "weights.svg"),
    ];
    if result_plots {
        plots.insert(0, ("Training and test observations", "training-data.svg"));
        plots.insert(1, ("Learned function or decision regions", "results.svg"));
    } else {
        html.push_str("<p>Result plots require 1D regression or 2D classification inputs. Numeric metrics and training-history plots are still available.</p>");
    }
    for (title, file) in plots {
        write!(
            html,
            "<section><h2>{title}</h2><img src=\"{file}\" alt=\"{title}\"></section>"
        )
        .unwrap();
    }
    html.push_str("<p>The weight chart shows evenly distributed parameter selections, identified by layer (L), neuron (N), weight input (W), or bias (B). All recorded parameters are in weights.csv. Snapshot zero is the initial state; the final state is also retained. Reports replace files with the same names in the chosen directory.</p></html>\n");
    fs::write(directory.join("index.html"), html)?;
    Ok(())
}

fn configuration(directory: &Path, options: &Options, settings: &Settings) -> AppResult<String> {
    let headers = [
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
    let hidden = &settings.layers[..settings.layers.len() - 1];
    let hidden = if hidden.is_empty() {
        "none".into()
    } else {
        hidden
            .iter()
            .map(|width| width.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let values = [
        settings.task.name().to_owned(),
        fs::canonicalize(&options.train_path)?
            .to_string_lossy()
            .into_owned(),
        fs::canonicalize(&options.test_path)?
            .to_string_lossy()
            .into_owned(),
        settings.task.classes().to_string(),
        hidden,
        activation_name(options.activation).into(),
        activation_name(settings.output).into(),
        loss_name(settings.loss).into(),
        settings.learning_rate.to_string(),
        options.seed.to_string(),
        options.bias.to_string(),
        options.epochs.to_string(),
        options.batch_size.to_string(),
        settings.snapshot_every.to_string(),
        options.weight_limit.to_string(),
        fs::canonicalize(directory)?.to_string_lossy().into_owned(),
    ];
    if values
        .iter()
        .any(|value| value.contains(['\t', '\n', '\r']))
    {
        return Err("Configuration paths cannot contain tabs or newlines".into());
    }
    Ok(headers.join("\t") + "\n" + &values.join("\t") + "\n")
}

fn activation_name(activation: Activation) -> &'static str {
    match activation {
        Activation::Linear => "linear",
        Activation::Sigmoid => "sigmoid",
        Activation::Tanh => "tanh",
        Activation::Relu => "relu",
        Activation::SoftMax => "softmax",
    }
}
fn loss_name(loss: Loss) -> &'static str {
    match loss {
        Loss::MeanSquaredError => "mse",
        Loss::MeanAbsoluteError => "mae",
        Loss::BinaryCrossEntropy => "bce",
        Loss::CategorialCrossEntropy => "cce",
    }
}

fn save_history(directory: &Path, history: &History) -> AppResult<()> {
    let mut epochs = String::from("epoch,train_loss,validation_loss\n");
    for row in &history.epochs {
        writeln!(
            epochs,
            "{},{},{}",
            row.epoch,
            row.train_loss,
            row.test_loss
                .map(|loss| loss.to_string())
                .unwrap_or_default()
        )
        .unwrap();
    }
    fs::write(directory.join("epochs.csv"), epochs)?;
    let mut iterations = String::from("iteration,epoch,samples_count,batch_loss_before_update\n");
    for row in &history.iterations {
        writeln!(
            iterations,
            "{},{},{},{}",
            row.iteration, row.epoch, row.samples_count, row.batch_loss
        )
        .unwrap();
    }
    fs::write(directory.join("iterations.csv"), iterations)?;
    let mut weights = String::from("iteration,layer,neuron,parameter,input,value\n");
    for snapshot in &history.parameter_snapshots {
        for (layer, neurons) in snapshot.weights.iter().enumerate() {
            for (neuron, values) in neurons.iter().enumerate() {
                for (input, value) in values.iter().enumerate() {
                    writeln!(
                        weights,
                        "{},{layer},{neuron},weight,{input},{value}",
                        snapshot.iteration
                    )
                    .unwrap();
                }
                writeln!(
                    weights,
                    "{},{layer},{neuron},bias,,{}",
                    snapshot.iteration, snapshot.biases[layer][neuron]
                )
                .unwrap();
            }
        }
    }
    fs::write(directory.join("weights.csv"), weights)?;
    Ok(())
}

fn parameter_values(weights: &[Vec<Vec<f64>>], biases: &[Vec<f64>]) -> Vec<(String, f64)> {
    let mut parameters = Vec::new();
    for (layer, neurons) in weights.iter().enumerate() {
        for (neuron, values) in neurons.iter().enumerate() {
            for (input, &value) in values.iter().enumerate() {
                parameters.push((format!("L{layer}.N{neuron}.W{input}"), value));
            }
            parameters.push((format!("L{layer}.N{neuron}.B"), biases[layer][neuron]));
        }
    }
    parameters
}

fn save_weights(directory: &Path, history: &History, limit: usize) -> AppResult<()> {
    let first = history
        .parameter_snapshots
        .first()
        .ok_or("Weight plots require parameter snapshots")?;
    let parameters = parameter_values(&first.weights, &first.biases);
    let count = limit.min(parameters.len());
    if count == 0 {
        return Err("Weight plots require a positive curve limit and parameters".into());
    }
    let indices: Vec<_> = (0..count)
        .map(|index| {
            if count == 1 {
                0
            } else {
                index * (parameters.len() - 1) / (count - 1)
            }
        })
        .collect();
    let mut curves = vec![Vec::new(); count];
    for snapshot in &history.parameter_snapshots {
        let values = parameter_values(&snapshot.weights, &snapshot.biases);
        if values.len() != parameters.len() {
            return Err("Parameter dimensions changed during training".into());
        }
        for (curve, &index) in curves.iter_mut().zip(&indices) {
            curve.push((snapshot.iteration as f64, values[index].1));
        }
    }
    let points: Vec<_> = curves.iter().flatten().copied().collect();
    let mut weight_bounds = charts::bounds(&points)?;
    weight_bounds[0] = 0.0;
    let mut chart = Chart::new(
        "Weight and bias evolution",
        "Parameter update",
        "Parameter value",
        weight_bounds,
    );
    for (curve, (&index, color)) in curves.iter().zip(indices.iter().zip(COLORS.iter().cycle())) {
        chart.legend(&parameters[index].0, color);
        chart.line(curve, color, &parameters[index].0);
    }
    fs::write(directory.join("weights.svg"), chart.finish())?;
    Ok(())
}

fn regression_plots(
    directory: &Path,
    train: &Dataset,
    test: &Dataset,
    predict: &impl Fn(&[f64]) -> AppResult<Vec<f64>>,
) -> AppResult<()> {
    let observed: Vec<_> = train
        .features
        .iter()
        .zip(&train.targets)
        .chain(test.features.iter().zip(&test.targets))
        .map(|(input, target)| (input[0], target[0]))
        .collect();
    let observed_bounds = charts::bounds(&observed)?;
    let mut curve = Vec::new();
    for index in 0..=200 {
        let x =
            observed_bounds[0] + (observed_bounds[1] - observed_bounds[0]) * index as f64 / 200.0;
        curve.push((x, predict(&[x])?[0]));
    }
    let all: Vec<_> = observed.iter().chain(&curve).copied().collect();
    let bounds = charts::bounds(&all)?;
    let mut results = Chart::new(
        "Regression: observations and learned function",
        "Input x",
        "Target / prediction (original units)",
        bounds,
    );
    let mut data = Chart::new(
        "Regression observations",
        "Input x",
        "Target (original units)",
        observed_bounds,
    );
    for chart in [&mut results, &mut data] {
        chart.legend("Training circles", COLORS[0]);
        chart.legend("Test squares", COLORS[1]);
        for (dataset, color, square, name) in [
            (train, COLORS[0], false, "Training"),
            (test, COLORS[1], true, "Test"),
        ] {
            for (input, target) in dataset.features.iter().zip(&dataset.targets) {
                chart.point(
                    (input[0], target[0]),
                    color,
                    square,
                    &format!("{name}: x={}, y={}", input[0], target[0]),
                    false,
                );
            }
        }
    }
    results.legend("Learned function", COLORS[2]);
    results.line(&curve, COLORS[2], "learned-function");
    fs::write(directory.join("training-data.svg"), data.finish())?;
    fs::write(directory.join("results.svg"), results.finish())?;
    Ok(())
}

fn classification_plots(
    directory: &Path,
    train: &Dataset,
    test: &Dataset,
    classes: usize,
    predict: &impl Fn(&[f64]) -> AppResult<Vec<f64>>,
) -> AppResult<()> {
    let points: Vec<_> = train
        .features
        .iter()
        .chain(&test.features)
        .map(|input| (input[0], input[1]))
        .collect();
    let bounds = charts::bounds(&points)?;
    let mut data = Chart::new(
        "Classification: true labels",
        "Feature x1",
        "Feature x2",
        bounds,
    );
    let mut results = Chart::new(
        "Classification: decision regions and predictions",
        "Feature x1",
        "Feature x2",
        bounds,
    );
    let color = |class: usize| format!("hsl({},65%,45%)", class.wrapping_mul(137) % 360);
    const GRID: usize = 50;
    for row in 0..GRID {
        for column in 0..GRID {
            let x0 = bounds[0] + (bounds[1] - bounds[0]) * column as f64 / GRID as f64;
            let x1 = bounds[0] + (bounds[1] - bounds[0]) * (column + 1) as f64 / GRID as f64;
            let y0 = bounds[2] + (bounds[3] - bounds[2]) * row as f64 / GRID as f64;
            let y1 = bounds[2] + (bounds[3] - bounds[2]) * (row + 1) as f64 / GRID as f64;
            let class = class_index(&predict(&[(x0 + x1) / 2.0, (y0 + y1) / 2.0])?, classes)?;
            results.cell(x0, y0, x1, y1, &color(class));
        }
    }
    for chart in [&mut data, &mut results] {
        for class in 0..classes {
            chart.legend(&format!("Class {class}"), &color(class));
        }
        chart.legend("Circles: training", "#475569");
        chart.legend("Squares: test", "#475569");
    }
    results.legend("Black border: error", "#111827");
    for (dataset, square, name) in [(train, false, "Training"), (test, true, "Test")] {
        for (input, target) in dataset.features.iter().zip(&dataset.targets) {
            let actual = target[0] as usize;
            let predicted = if square {
                class_index(&predict(input)?, classes)?
            } else {
                actual
            };
            data.point(
                (input[0], input[1]),
                &color(actual),
                square,
                &format!("{name} true class {actual}"),
                false,
            );
            results.point(
                (input[0], input[1]),
                &color(predicted),
                square,
                &format!("{name}: predicted {predicted}, true {actual}"),
                predicted != actual,
            );
        }
    }
    fs::write(directory.join("training-data.svg"), data.finish())?;
    fs::write(directory.join("results.svg"), results.finish())?;
    Ok(())
}
