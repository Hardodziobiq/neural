use crate::net::error::ValidationError;

const PROTECT: f64 = 1e-15;
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
    /// Lecture convention: half the sum of squared output errors, without division by output count.
    MeanSquaredError,
    MeanAbsoluteError,
    BinaryCrossEntropy,
    CategorialCrossEntropy,
}

impl Loss {
    /// Calculates loss for equally sized, nonempty vectors of finite values.
    pub fn calculate(&self, predictions: &[f64], real: &[f64]) -> Result<f64, ValidationError> {
        validate_inputs(predictions, real)?;
        Ok(match self {
            Loss::MeanSquaredError => mse(predictions, real),
            Loss::MeanAbsoluteError => mae(predictions, real),
            Loss::BinaryCrossEntropy => bce(predictions, real),
            Loss::CategorialCrossEntropy => cce(predictions, real),
        })
    }
    /// Returns the loss gradient with respect to predictions, before applying output-activation derivatives.
    pub fn calculate_init_gradient(
        &self,
        predictions: &[f64],
        real: &[f64],
    ) -> Result<Vec<f64>, ValidationError> {
        validate_inputs(predictions, real)?;
        Ok(match self {
            Loss::MeanSquaredError => init_gradient_mse(predictions, real),
            Loss::MeanAbsoluteError => init_gradient_mae(predictions, real),
            Loss::BinaryCrossEntropy => init_gradient_bce(predictions, real),
            Loss::CategorialCrossEntropy => init_gradient_cce(predictions, real),
        })
    }
}

/// Rejects empty or mismatched vectors and nonfinite values before loss or gradient calculation.
fn validate_inputs(predictions: &[f64], real: &[f64]) -> Result<(), ValidationError> {
    if predictions.len() != real.len() {
        return Err(ValidationError::new(format!(
            "Predictions and targets must have equal lengths: got {} and {}",
            predictions.len(),
            real.len(),
        )));
    }
    if predictions.is_empty() {
        return Err(ValidationError::new(
            "Predictions and targets must not be empty",
        ));
    }
    for (name, values) in [("Prediction", predictions), ("Target", real)] {
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(ValidationError::new(format!(
                    "{name} at index {index} must be finite",
                )));
            }
        }
    }
    Ok(())
}

/// Returns half the sum of squared errors for one sample, as defined in the lecture.
fn mse(predictions: &[f64], real: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| 0.5 * (prediction - real).powi(2))
        .sum::<f64>()
}
/// Returns `prediction - target` for each output, matching the lecture's half-squared loss.
fn init_gradient_mse(predictions: &[f64], real: &[f64]) -> Vec<f64> {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| prediction - real)
        .collect()
}

/// Returns the mean absolute error across the outputs of one sample.
fn mae(predictions: &[f64], real: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| (prediction - real).abs())
        .sum::<f64>()
        / predictions.len() as f64
}
/// Returns the mean-scaled sign of each error, choosing a zero subgradient when prediction equals target.
fn init_gradient_mae(predictions: &[f64], real: &[f64]) -> Vec<f64> {
    let n = predictions.len() as f64;
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| {
            if prediction > real {
                1.0 / n
            } else if prediction < real {
                -1.0 / n
            } else {
                0.0
            }
        })
        .collect()
}

/// Sums binary cross-entropy terms, clipping predicted probabilities to protect logarithms.
fn bce(predictions: &[f64], real: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| {
            let p = prediction.clamp(PROTECT, 1.0 - PROTECT);
            -(real * p.ln() + (1.0 - real) * (1.0 - p).ln())
        })
        .sum()
}
/// Returns the binary cross-entropy gradient with respect to clipped predicted probabilities.
fn init_gradient_bce(predictions: &[f64], real: &[f64]) -> Vec<f64> {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| {
            let p: f64 = prediction.clamp(PROTECT, 1.0 - PROTECT);
            (p - real) / (p * (1.0 - p))
        })
        .collect()
}

/// Sums categorical cross-entropy terms for class targets and clipped predicted probabilities.
fn cce(predictions: &[f64], real: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| {
            let p: f64 = prediction.clamp(PROTECT, 1.0 - PROTECT);
            -(real * p.ln())
        })
        .sum()
}
/// Returns `-target / probability` per class using clipped probabilities.
fn init_gradient_cce(predictions: &[f64], real: &[f64]) -> Vec<f64> {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| {
            let p = prediction.clamp(PROTECT, 1.0 - PROTECT);
            -real / p
        })
        .collect()
}
