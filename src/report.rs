use crate::{
    AppResult,
    cli::Options,
    experiment::{Settings, Summary, Task, class_index},
};
use neural::net::{
    activation_function::ActivationFunction as Activation,
    charts::{self, COLORS, Chart},
    data_set::Dataset,
    history::History,
    loss::Loss,
};
use std::{
    collections::BTreeMap,
    fmt::Write,
    fs,
    path::{Path, PathBuf},
};

pub struct SuiteRun {
    pub row: usize,
    pub task: Task,
    pub options: Options,
    pub result: Result<Summary, String>,
}

struct ComparisonRun {
    seed: u64,
    configuration: String,
    metrics: Summary,
    epochs: Vec<(f64, f64)>,
    iterations: Vec<(f64, f64)>,
}

/// Reads two numeric columns from an internally generated history CSV, preserving recorded update positions.
fn history_points(path: &Path, x: usize, y: usize) -> AppResult<Vec<(f64, f64)>> {
    fs::read_to_string(path)?
        .lines()
        .skip(1)
        .map(|line| {
            let values: Vec<_> = line.split(',').collect();
            Ok((
                values.get(x).ok_or("Missing history coordinate")?.parse()?,
                values.get(y).ok_or("Missing history value")?.parse()?,
            ))
        })
        .collect()
}

/// Loads one successful run's histories for shared batch comparisons, without retraining the model.
fn comparison_run(run: &SuiteRun, configuration: String) -> AppResult<ComparisonRun> {
    let directory = run.options.out_dir.as_ref().unwrap();
    Ok(ComparisonRun {
        seed: run.options.seed,
        configuration,
        metrics: run.result.as_ref().unwrap().clone(),
        epochs: history_points(&directory.join("epochs.csv"), 0, 1)?,
        iterations: history_points(&directory.join("iterations.csv"), 0, 3)?,
    })
}

type ComparisonGroups<'a> = BTreeMap<&'a str, BTreeMap<u64, &'a ComparisonRun>>;

/// Computes a mean and sample standard deviation; one observation has no estimated spread.
fn mean_std(values: &[f64]) -> (f64, f64) {
    let mut mean = 0.0;
    let mut squares = 0.0;
    for (index, value) in values.iter().enumerate() {
        let delta = value - mean;
        mean += delta / (index + 1) as f64;
        squares += delta * (value - mean);
    }
    (
        mean,
        if values.len() > 1 {
            (squares.max(0.0) / (values.len() - 1) as f64).sqrt()
        } else {
            0.0
        },
    )
}

/// Formats a metric across unique seeds, distinguishing one-seed runs from an estimated standard deviation.
fn metric_summary(values: &[f64], percent: bool) -> String {
    if values.is_empty() {
        return "—".into();
    }
    let (mut mean, mut std) = mean_std(values);
    if percent {
        mean *= 100.0;
        std *= 100.0;
    }
    let unit = if percent { "%" } else { "" };
    if values.len() == 1 {
        format!("{mean:.6}{unit} (single seed)")
    } else {
        format!("{mean:.6} ± {std:.6}{unit}")
    }
}

/// Overlays configuration means with sample-standard-deviation bands on shared, exactly aligned history axes.
fn comparison_chart(
    title: &str,
    x_label: &str,
    y_label: &str,
    id: &str,
    groups: &ComparisonGroups<'_>,
    epochs: bool,
) -> AppResult<String> {
    let mut series = Vec::new();
    for runs in groups.values() {
        let histories: Vec<_> = runs
            .values()
            .map(|run| if epochs { &run.epochs } else { &run.iterations })
            .collect();
        let first = histories[0];
        if histories.iter().any(|history| {
            history.len() != first.len() || history.iter().zip(first).any(|(a, b)| a.0 != b.0)
        }) {
            return Err("Cannot average histories with different epoch/update coordinates".into());
        }
        let mut averages = Vec::new();
        let mut intervals = Vec::new();
        for (index, &(x, _)) in first.iter().enumerate() {
            let values: Vec<_> = histories.iter().map(|history| history[index].1).collect();
            let (mean, std) = mean_std(&values);
            averages.push((x, mean));
            intervals.push((x, (mean - std).max(0.0), mean + std));
        }
        series.push((runs.len(), averages, intervals));
    }
    let points: Vec<_> = series
        .iter()
        .flat_map(|(_, _, intervals)| {
            intervals
                .iter()
                .flat_map(|&(x, low, high)| [(x, low), (x, high)])
        })
        .collect();
    let mut bounds = charts::bounds(&points)?;
    bounds[0] = 0.0;
    let mut chart = Chart::new(title, x_label, y_label, bounds);
    for (index, (count, points, intervals)) in series.into_iter().enumerate() {
        let color = format!("hsl({},65%,40%)", index * 137 % 360);
        chart.legend(&format!("C{} (n={count})", index + 1), &color);
        if count > 1 {
            chart.band(&intervals, &color);
        }
        chart.line(&points, &color, &format!("{id}-config-{}", index + 1));
    }
    Ok(chart.finish())
}

/// Groups identical configurations across distinct seeds, then adds averaged metrics and two shared loss plots.
fn batch_comparisons(html: &mut String, runs: &[ComparisonRun]) -> AppResult<()> {
    if runs.is_empty() {
        html.push_str("<p>No successful runs are available for comparison.</p>");
        return Ok(());
    }
    let mut groups: ComparisonGroups<'_> = BTreeMap::new();
    for run in runs {
        groups
            .entry(&run.configuration)
            .or_default()
            .entry(run.seed)
            .or_insert(run);
    }
    html.push_str("<section><h2>Configuration averages across seeds</h2><p>Results are mean ± sample standard deviation across distinct successful seeds of the same configuration. Repeated runs of the same configuration and seed count once. Failed runs are excluded and listed below; n gives the actual number of successful seeds. With one seed the spread cannot be estimated.</p><div class=\"scroll\"><table><thead><tr><th>ID</th><th>Configuration</th><th>Seeds</th><th>Train loss</th><th>Test loss</th><th>Test MSE</th><th>Test accuracy</th><th>Seconds</th></tr></thead><tbody>");
    for (index, (configuration, group)) in groups.iter().enumerate() {
        write!(
            html,
            "<tr><td>C{}</td><td>{}</td><td>n={}: {}</td>",
            index + 1,
            charts::escape(configuration),
            group.len(),
            group
                .keys()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        for metric in 0..5 {
            let values: Vec<_> = group
                .values()
                .filter_map(|run| match metric {
                    0 => Some(run.metrics.train_loss),
                    1 => Some(run.metrics.test_loss),
                    2 => run.metrics.test_mse,
                    3 => run.metrics.accuracy,
                    _ => Some(run.metrics.seconds),
                })
                .collect();
            write!(html, "<td>{}</td>", metric_summary(&values, metric == 3)).unwrap();
        }
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table></div></section><section><h2>Training loss by epoch</h2><p>Each line is a configuration mean over seeds; shading is ± one sample standard deviation, clipped at zero. It is not a confidence interval. Loss is measured after each epoch. Different loss functions have different scales and should not be ranked by these values.</p>");
    html.push_str(&comparison_chart(
        "Training loss by epoch",
        "Epoch",
        "Training loss",
        "batch-epoch",
        &groups,
        true,
    )?);
    html.push_str("</section><section><h2>Batch loss by parameter update</h2><p>Configuration means and standard deviations use the same update index across seeds. Each value is measured on that run's batch before its update; shuffles mean that the samples can differ across seeds. Batch sizes can differ between configurations.</p>");
    html.push_str(&comparison_chart(
        "Batch loss by parameter update",
        "Parameter update",
        "Mean batch loss",
        "batch-iteration",
        &groups,
        false,
    )?);
    html.push_str("</section>");
    Ok(())
}

/// Percent-encodes a filesystem path for HTML links, preserving URL path separators.
fn encode_path(path: &str) -> String {
    let mut encoded = String::new();
    for byte in path.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~/".contains(&byte) {
            encoded.push(byte as char);
        } else {
            write!(encoded, "%{byte:02X}").unwrap();
        }
    }
    encoded
}

/// Builds a portable relative link, falling back to a file URL for paths on another filesystem root.
fn report_url(base: &Path, target: &Path) -> std::io::Result<String> {
    let base = fs::canonicalize(base)?;
    let target = fs::canonicalize(target)?;
    let from: Vec<_> = base.components().collect();
    let to: Vec<_> = target.components().collect();
    let common = from.iter().zip(&to).take_while(|(a, b)| a == b).count();
    if common == 0 {
        let path = target.to_string_lossy().replace('\\', "/");
        let path = path.strip_prefix("//?/").unwrap_or(&path);
        return Ok(
            if let Some(unc) = path
                .strip_prefix("UNC/")
                .or_else(|| path.strip_prefix("//"))
            {
                format!("file://{}", encode_path(unc))
            } else {
                format!("file:///{}", encode_path(path).replace("%3A", ":"))
            },
        );
    }
    let mut parts = vec!["..".to_owned(); from.len() - common];
    parts.extend(
        to[common..]
            .iter()
            .map(|part| part.as_os_str().to_string_lossy().into_owned()),
    );
    Ok(encode_path(&parts.join("/")))
}

/// Writes a batch comparison for one dataset pair: shared observations, metrics and overlaid loss histories.
/// Failed runs remain visible without linking stale artifacts from earlier runs.
pub fn save_suite(summary_path: &Path, plan: &Path, runs: &[SuiteRun]) -> AppResult<PathBuf> {
    let mut output = summary_path.with_extension("html");
    if output == summary_path {
        output = summary_path.with_file_name(format!(
            "{}.html",
            summary_path.file_name().unwrap().to_string_lossy()
        ));
    }
    let base = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let failed = runs.iter().filter(|run| run.result.is_err()).count();
    let first = runs
        .first()
        .ok_or("Cannot report an empty experiment plan")?;
    let task = first.task.name();
    let classes = first.task.classes();
    let train = &first.options.train_path;
    let test = &first.options.test_path;
    let mut html = format!(
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Experiment batch report</title><style>body{{font:16px system-ui;margin:32px auto;max-width:1400px;padding:0 20px;color:#253047}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #ddd;padding:7px;text-align:left}}.scroll{{overflow:auto}}svg{{width:100%;height:auto;max-width:1100px}}.error{{color:#b91c1c}}code{{overflow-wrap:anywhere}}a{{color:#2563eb}}</style><h1>Experiment batch report</h1><p>Plan: <code>{}</code></p><p>Runs: {}. Successful: {}. Failed: {}.</p><p><a href=\"{}\">Download summary CSV</a></p><p>All runs use one training/evaluation pair. Accuracy is shown as a percentage; regression MSE uses original target units. Compare different losses using accuracy or original-unit MSE. The evaluation set does not influence training.</p>",
        charts::escape(&plan.to_string_lossy()),
        runs.len(),
        runs.len() - failed,
        failed,
        charts::escape(&report_url(base, summary_path)?)
    );
    write!(html, "<section class=\"dataset\"><h2>{task}: {}</h2><p>Train: <code>{}</code><br>Evaluation: <code>{}</code></p>",
            charts::escape(&Path::new(train).file_name().unwrap_or_default().to_string_lossy()), charts::escape(train), charts::escape(test)).unwrap();
    if task == "classification" {
        write!(html, "<p>Classes: {classes}.</p>").unwrap();
    }
    let table_start = html.len();
    html.push_str("<div class=\"scroll\"><table><thead><tr><th>Plan row</th><th>Configuration</th><th>Seed</th><th>Train loss</th><th>Test loss</th><th>Test MSE</th><th>Test accuracy</th><th>Seconds</th><th>Report / error</th></tr></thead><tbody>");
    let mut comparisons = Vec::new();
    let mut observations = None;
    for run in runs {
        write!(html, "<tr><td>{}</td>", run.row).unwrap();
        match &run.result {
            Err(error) => {
                write!(
                    html,
                    "<td>Failed</td><td>{}</td><td colspan=\"6\" class=\"error\">{}</td></tr>",
                    run.options.seed,
                    charts::escape(error)
                )
                .unwrap();
            }
            Ok(metrics) => {
                let directory = run.options.out_dir.as_ref().unwrap();
                let saved = fs::read_to_string(directory.join("configuration.tsv"))?;
                let run_html = fs::read_to_string(directory.join("index.html"))?;
                if observations.is_none() && run_html.contains("src=\"training-data.svg\"") {
                    observations = Some(fs::read_to_string(directory.join("training-data.svg"))?);
                }
                let mut lines = saved.lines();
                let configuration: BTreeMap<_, _> = lines
                    .next()
                    .unwrap()
                    .split('\t')
                    .zip(lines.next().unwrap().split('\t'))
                    .collect();
                let description = [
                    "hidden",
                    "activation",
                    "output",
                    "loss",
                    "learning_rate",
                    "bias",
                    "epochs",
                    "batch",
                ]
                .iter()
                .map(|key| format!("{key}={}", configuration[key]))
                .collect::<Vec<_>>()
                .join(", ");
                comparisons.push(comparison_run(run, description.clone())?);
                let mse = metrics
                    .test_mse
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "—".into());
                let accuracy = metrics
                    .accuracy
                    .map(|value| format!("{:.2}%", value * 100.0))
                    .unwrap_or_else(|| "—".into());
                write!(html, "<td>{}</td><td>{}</td><td>{:.6}</td><td>{:.6}</td><td>{mse}</td><td>{accuracy}</td><td>{:.3}</td><td><a href=\"{}\">Individual report</a></td></tr>",
                        charts::escape(&description), run.options.seed, metrics.train_loss, metrics.test_loss, metrics.seconds,
                        charts::escape(&report_url(base, &directory.join("index.html"))?)).unwrap();
            }
        }
    }
    html.push_str("</tbody></table></div>");
    let individual_table = html.split_off(table_start);
    if let Some(svg) = observations {
        html.push_str("<h2>Shared dataset</h2>");
        html.push_str(&svg);
    }
    html.push_str("</section>");
    batch_comparisons(&mut html, &comparisons)?;
    html.push_str("<details><summary>Individual runs and errors</summary>");
    html.push_str(&individual_table);
    html.push_str("</details>");
    html.push_str("<p>Comparison plots are embedded in this page. Individual reports and the summary CSV are linked separately.</p></html>\n");
    fs::write(&output, html)?;
    Ok(output)
}

/// Formats one CSV row, quoting fields and escaping embedded quotes.
pub fn csv_row(values: &[String]) -> String {
    values
        .iter()
        .map(|value| format!("\"{}\"", value.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(",")
        + "\n"
}

/// Writes configuration, CSV histories, SVG plots, and an HTML overview for a completed run.
/// Datasets and callback inputs use original coordinates; regression predictions use original target units.
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
    html.push_str("<p>MSE loss follows the lecture: half the sum of squared errors over outputs, averaged over samples. Regression losses use standardized targets. Test MSE in original units is a separate mean squared error metric without the factor 0.5.</p>");
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

/// Returns a replayable TSV configuration with resolved model settings and absolute file paths.
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

/// Returns the command-line name used to save and replay an activation choice.
fn activation_name(activation: Activation) -> &'static str {
    match activation {
        Activation::Linear => "linear",
        Activation::Sigmoid => "sigmoid",
        Activation::Tanh => "tanh",
        Activation::Relu => "relu",
        Activation::SoftMax => "softmax",
    }
}
/// Returns the command-line name used to save and replay a loss choice.
fn loss_name(loss: Loss) -> &'static str {
    match loss {
        Loss::MeanSquaredError => "mse",
        Loss::MeanAbsoluteError => "mae",
        Loss::BinaryCrossEntropy => "bce",
        Loss::CategorialCrossEntropy => "cce",
    }
}

/// Exports epoch losses, pre-update batch losses, and every recorded parameter snapshot to CSV.
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

/// Flattens layer parameters into named values such as `L0.N1.W2` and `L0.N1.B`.
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

/// Plots an evenly distributed selection of weight and bias trajectories from saved snapshots.
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

/// Plots original observations and a 201-point prediction curve for one-dimensional regression.
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

/// Plots true 2D labels and a 50-by-50 decision grid, marking incorrect test predictions.
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
