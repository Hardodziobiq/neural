use std::io;

use crate::net::{
    charts::{Chart, invalid},
    history::History,
};

/// Validates epoch metrics and renders their loss curves with the shared chart engine.
/// Parameter snapshots are not required; missing evaluation values break the curve.
pub(super) fn render(history: &History) -> io::Result<String> {
    let epochs = &history.epochs;
    if epochs.is_empty() {
        return Err(invalid("Cannot plot an empty training history"));
    }
    if epochs.windows(2).any(|pair| pair[0].epoch >= pair[1].epoch) {
        return Err(invalid("Plot epochs must be strictly increasing"));
    }
    let mut minimum = 0.0_f64;
    let mut maximum = 0.0_f64;
    for epoch in epochs {
        for loss in std::iter::once(epoch.train_loss).chain(epoch.test_loss) {
            if !loss.is_finite() {
                return Err(invalid("Plot losses must be finite"));
            }
            minimum = minimum.min(loss);
            maximum = maximum.max(loss);
        }
    }
    if minimum == maximum {
        maximum = minimum + 1.0;
    }
    if !(maximum - minimum).is_finite() {
        return Err(invalid("Loss range is too large to plot"));
    }
    let mut first = epochs[0].epoch as f64;
    let mut last = epochs[epochs.len() - 1].epoch as f64;
    if first == last {
        let padding = first.abs().max(1.0) * 0.05;
        first -= padding;
        last += padding;
    }
    let mut chart = Chart::new(
        "Loss by epoch",
        "Epoch",
        "Loss",
        [first, last, minimum, maximum],
    );
    let training: Vec<_> = epochs
        .iter()
        .map(|epoch| (epoch.epoch as f64, epoch.train_loss))
        .collect();
    chart.legend("Training", "#2563eb");
    chart.line(&training, "#2563eb", "training-loss");
    if epochs.iter().any(|epoch| epoch.test_loss.is_some()) {
        chart.legend("Test", "#d97706");
    }
    let evaluation: Vec<_> = epochs
        .iter()
        .map(|epoch| (epoch.epoch as f64, epoch.test_loss))
        .collect();
    chart.gapped_line(&evaluation, "#d97706", "test-loss");
    Ok(chart.finish())
}
