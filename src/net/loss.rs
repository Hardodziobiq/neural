use crate::net::error::ValidationError;

const PROTECT: f64 = 1e-15;
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
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

fn mse(predictions: &[f64], real: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| (prediction - real).powi(2))
        .sum::<f64>()
        / predictions.len() as f64
}
fn init_gradient_mse(predictions: &[f64], real: &[f64]) -> Vec<f64> {
    let scale = 2.0 / predictions.len() as f64;
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| scale * (prediction - real))
        .collect()
}

fn mae(predictions: &[f64], real: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| (prediction - real).abs())
        .sum::<f64>()
        / predictions.len() as f64
}
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
