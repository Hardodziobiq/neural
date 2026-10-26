mod cli;
mod experiment;
mod report;

use cli::{Command, Options};
use experiment::Task;
use std::{env, error::Error};

type AppResult<T> = Result<T, Box<dyn Error>>;

/// Parses the command and dispatches a demo, a single training run, or a TSV experiment plan.
fn main() -> AppResult<()> {
    match cli::parse(&env::args().skip(1).collect::<Vec<_>>())? {
        Command::Demo => {
            experiment::run(Options::demo("regression"), Task::Regression)?;
            experiment::run(Options::demo("classification"), Task::Classification(3))?;
        }
        Command::Help => println!("{}", cli::HELP),
        Command::Regression(options) => {
            experiment::run(options, Task::Regression)?;
        }
        Command::Classification(options, classes) => {
            experiment::run(options, Task::Classification(classes))?;
        }
        Command::Experiments(path, summary) => experiment::run_suite(&path, &summary)?,
    }
    Ok(())
}
