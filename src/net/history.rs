#[derive(Clone, Debug)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub test_loss: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ParameterSnapshot {
    pub iteration: usize,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Default)]
pub struct History {
    pub epochs: Vec<EpochMetrics>,
    pub parameter_snapshots: Vec<ParameterSnapshot>,
}

impl History {
    /// Saves a linear-scale SVG chart of training and available test losses.
    /// The parent directory must exist. Existing files are replaced only after validation.
    pub fn save_loss_plot(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let svg = crate::net::history_plot::render(self)?;
        std::fs::write(path, svg)
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_epoch(&mut self, epoch: usize, train_loss: f64, test_loss: Option<f64>) {
        self.epochs.push(EpochMetrics {
            epoch,
            train_loss,
            test_loss,
        });
    }

    pub fn record_parameters(
        &mut self,
        iteration: usize,
        weights: Vec<Vec<Vec<f64>>>,
        biases: Vec<Vec<f64>>,
    ) {
        self.parameter_snapshots.push(ParameterSnapshot {
            iteration,
            weights,
            biases,
        });
    }

    pub fn len(&self) -> usize {
        self.epochs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.epochs.is_empty()
    }
}
/// Parameter recording is disabled by default; epoch losses are always recorded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SnapshotPolicy {
    #[default]
    Disabled,
    EveryEpoch,
    /// Records every N parameter updates. N must be greater than zero.
    EveryIterations(usize),
}
