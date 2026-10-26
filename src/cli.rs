use neural::net::{activation_function::ActivationFunction as Activation, loss::Loss};
use std::{collections::HashSet, path::PathBuf};

pub const HELP: &str = "Usage:
  neural
  neural regression TRAIN.csv TEST.csv [EPOCHS] [BATCH] [OPTIONS]
  neural classification TRAIN.csv TEST.csv CLASSES [EPOCHS] [BATCH] [OPTIONS]
  neural experiments PLAN.tsv [--summary FILE.csv]

Options:
  --hidden 16,16|none       Hidden layers (output layer is added automatically)
  --activation sigmoid|tanh|relu|linear
  --output linear|softmax|sigmoid
  --loss mse|mae|cce|bce
  --learning-rate NUMBER    Finite, nonnegative learning rate
  --seed NUMBER            Random seed, including zero (default 42)
  --bias true|false        Enable biases (default true)
  --out-dir DIRECTORY      Save HTML report, plots, history and configuration
  --snapshot-every NUMBER  Record parameters every N updates for reports
  --weight-limit NUMBER    Maximum weight/bias curves (default 12)
  --plot FILE.svg          Save the epoch loss plot separately

Defaults: 500 epochs, batch 16. No arguments: run both demos and save reports.
The final test set is evaluated only after training, never used to select settings.";

pub enum Command {
    Demo,
    Help,
    Regression(Options),
    Classification(Options, usize),
    Experiments(PathBuf, PathBuf),
}

#[derive(Clone)]
pub struct Options {
    pub train_path: String,
    pub test_path: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub plot: Option<PathBuf>,
    pub out_dir: Option<PathBuf>,
    pub hidden: Option<Vec<usize>>,
    pub activation: Activation,
    pub output: Option<Activation>,
    pub loss: Option<Loss>,
    pub learning_rate: Option<f64>,
    pub seed: u64,
    pub bias: bool,
    pub snapshot_every: Option<usize>,
    pub weight_limit: usize,
}

impl Options {
    pub fn demo(task: &str) -> Self {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/data/demo/");
        let mut options = Self::new(
            format!("{root}{task}_train.csv"),
            format!("{root}{task}_test.csv"),
        );
        options.out_dir = Some(PathBuf::from(format!("results/demo-{task}")));
        options
    }

    fn new(train_path: String, test_path: String) -> Self {
        Self {
            train_path,
            test_path,
            epochs: 500,
            batch_size: 16,
            plot: None,
            out_dir: None,
            hidden: None,
            activation: Activation::Tanh,
            output: None,
            loss: None,
            learning_rate: None,
            seed: 42,
            bias: true,
            snapshot_every: None,
            weight_limit: 12,
        }
    }
}

pub fn parse(args: &[String]) -> Result<Command, String> {
    if args.is_empty() {
        return Ok(Command::Demo);
    }
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
        return Ok(Command::Help);
    }
    if args[0] == "experiments" {
        return match args.len() {
            2 => Ok(Command::Experiments(
                PathBuf::from(&args[1]),
                "results/summary.csv".into(),
            )),
            4 if args[2] == "--summary" => Ok(Command::Experiments(
                PathBuf::from(&args[1]),
                PathBuf::from(&args[3]),
            )),
            _ => Err(HELP.into()),
        };
    }
    let mut positional = Vec::new();
    let mut flags = Vec::new();
    let mut seen = HashSet::new();
    let mut index = 0;
    while index < args.len() {
        if args[index].starts_with("--") {
            let flag = args[index].as_str();
            if !seen.insert(flag) {
                return Err(format!("{flag} can only be specified once"));
            }
            index += 1;
            let value = args
                .get(index)
                .filter(|value| !value.starts_with("--"))
                .ok_or_else(|| format!("{flag} requires a value"))?;
            flags.push((flag, value.as_str()));
        } else if args[index].starts_with('-') {
            return Err(format!("Unknown argument: {}", args[index]));
        } else {
            positional.push(args[index].as_str());
        }
        index += 1;
    }
    let task = positional.first().copied().unwrap_or("");
    let required = match task {
        "regression" => 3,
        "classification" => 4,
        _ => return Err(HELP.into()),
    };
    if !(required..=required + 2).contains(&positional.len()) {
        return Err(HELP.into());
    }
    let mut options = Options::new(positional[1].into(), positional[2].into());
    options.epochs = positive(
        positional.get(required).copied().unwrap_or("500"),
        "Epoch count",
    )?;
    options.batch_size = positive(
        positional.get(required + 1).copied().unwrap_or("16"),
        "Batch size",
    )?;
    for (flag, value) in flags {
        match flag {
            "--plot" => {
                let path = PathBuf::from(value);
                if !path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("svg"))
                {
                    return Err("--plot requires a file with the .svg extension".into());
                }
                options.plot = Some(path);
            }
            "--out-dir" if !value.is_empty() => options.out_dir = Some(value.into()),
            "--hidden" => {
                options.hidden = Some(if value == "none" {
                    vec![]
                } else {
                    value
                        .split(',')
                        .map(|item| positive(item, "Hidden layer width"))
                        .collect::<Result<_, _>>()?
                })
            }
            "--activation" => options.activation = activation(value)?,
            "--output" => options.output = Some(activation(value)?),
            "--loss" => {
                options.loss = Some(match value {
                    "mse" => Loss::MeanSquaredError,
                    "mae" => Loss::MeanAbsoluteError,
                    "cce" => Loss::CategorialCrossEntropy,
                    "bce" => Loss::BinaryCrossEntropy,
                    _ => return Err(format!("Unknown loss: {value}")),
                })
            }
            "--learning-rate" => {
                let rate: f64 = value.parse().map_err(|_| "Invalid learning rate")?;
                if !rate.is_finite() || rate < 0.0 {
                    return Err("Learning rate must be finite and nonnegative".into());
                }
                options.learning_rate = Some(rate);
            }
            "--seed" => {
                options.seed = value
                    .parse()
                    .map_err(|_| "Seed must be a nonnegative u64 integer")?
            }
            "--bias" => options.bias = value.parse().map_err(|_| "Bias must be true or false")?,
            "--snapshot-every" => {
                options.snapshot_every = Some(positive(value, "Snapshot interval")?)
            }
            "--weight-limit" => options.weight_limit = positive(value, "Weight curve limit")?,
            _ => return Err(format!("Unknown argument: {flag}")),
        }
    }
    if options.snapshot_every.is_some() && options.out_dir.is_none() {
        return Err("--snapshot-every requires --out-dir".into());
    }
    if task == "regression" {
        Ok(Command::Regression(options))
    } else {
        let classes = positive(positional[3], "Class count")?;
        if classes < 2 {
            return Err("Classification requires at least two classes".into());
        }
        Ok(Command::Classification(options, classes))
    }
}

fn activation(value: &str) -> Result<Activation, String> {
    match value {
        "linear" => Ok(Activation::Linear),
        "sigmoid" => Ok(Activation::Sigmoid),
        "tanh" => Ok(Activation::Tanh),
        "relu" => Ok(Activation::Relu),
        "softmax" => Ok(Activation::SoftMax),
        _ => Err(format!("Unknown activation: {value}")),
    }
}

fn positive(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| format!("{name} must be a positive integer"))
}
