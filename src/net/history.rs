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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_epoch(
        &mut self,
        epoch: usize,
        train_loss: f64,
        test_loss: Option<f64>,
    ) {
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