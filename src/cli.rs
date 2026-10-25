use std::path::PathBuf;

pub const HELP: &str = "Usage:
  neural
  neural regression TRAIN.csv TEST.csv [EPOCHS] [BATCH] [--plot FILE.svg]
  neural classification TRAIN.csv TEST.csv CLASSES [EPOCHS] [BATCH] [--plot FILE.svg]

No arguments: run both bundled examples.
Defaults: 500 epochs, batch size 16. Class labels start at zero.
--plot saves training/test loss by epoch as an SVG file.
Network architecture and learning rate are configured in src/main.rs.";

pub enum Command {
    Demo,
    Help,
    Regression(Options),
    Classification(Options, usize),
}

pub struct Options {
    pub train_path: String,
    pub test_path: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub plot: Option<PathBuf>,
}

impl Options {
    pub fn demo(task: &str) -> Self {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/data/demo/");
        Self {
            train_path: format!("{root}{task}_train.csv"),
            test_path: format!("{root}{task}_test.csv"),
            epochs: 500,
            batch_size: 16,
            plot: None,
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
    let mut positional = Vec::new();
    let mut plot = None;
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--plot" {
            if plot.is_some() {
                return Err("--plot can only be specified once".into());
            }
            index += 1;
            let path = args.get(index).ok_or("--plot requires an SVG file path")?;
            let path = PathBuf::from(path);
            if !path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("svg"))
            {
                return Err("--plot requires a file with the .svg extension".into());
            }
            plot = Some(path);
        } else if args[index].starts_with('-') {
            return Err(format!("Unknown argument: {}\n{HELP}", args[index]));
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
    let options = Options {
        train_path: positional[1].into(),
        test_path: positional[2].into(),
        epochs: positive(
            positional.get(required).copied().unwrap_or("500"),
            "Epoch count",
        )?,
        batch_size: positive(
            positional.get(required + 1).copied().unwrap_or("16"),
            "Batch size",
        )?,
        plot,
    };
    if task == "regression" {
        Ok(Command::Regression(options))
    } else {
        let classes = positive(positional[3], "Class count")?;
        if classes < 2 {
            return Err("Softmax classification requires at least two classes".into());
        }
        Ok(Command::Classification(options, classes))
    }
}

fn positive(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| format!("{name} must be a positive integer"))
}
