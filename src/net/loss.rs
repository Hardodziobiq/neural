pub enum Loss {
    MeanSquaredError,
    MeanAbsoluteError,
    BinaryCrossEntropy,
    CategorialCrossEntropy,
}

impl Loss {
    pub fn calculate(&self, predictions: &[f64], real: &[f64]) -> f64 {
        match self {
            Loss::MeanSquaredError => mse(predictions, real),
            Loss::MeanAbsoluteError => mae(predictions, real),
            Loss::BinaryCrossEntropy => bce(predictions, real),
            Loss::CategorialCrossEntropy => cce(predictions, real),
        }
    }
    pub fn calculate_init_gradient(&self, predictions: &[f64], real: &[f64]) -> Vec<f64> {
        match self {
            Loss::MeanSquaredError => init_gradient_mse(predictions, real),
            Loss::MeanAbsoluteError => init_gradient_mae(predictions, real),
            Loss::BinaryCrossEntropy => init_gradient_bce(predictions, real),
            Loss::CategorialCrossEntropy => init_gradient_cce(predictions, real),
        }
    }
}

fn mse(predictions: &[f64], real: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| (prediction - real).powi(2))
        .sum::<f64>()
        / 2.0
}
fn init_gradient_mse(predictions: &[f64], real: &[f64]) -> Vec<f64> {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| {
            prediction - real
        })
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
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| {
            (prediction - real).signum() / predictions.len() as f64
        })
        .collect()
}

fn bce(predictions: &[f64], real: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| {
            -(real * prediction.clamp(1e-15, 1.0 - 1e-15).ln()
                + (1.0 - real) * (1.0 - prediction.clamp(1e-15, 1.0 - 1e-15)).ln())
        })
        .sum()
}
fn init_gradient_bce(predictions: &[f64], real: &[f64]) -> Vec<f64> {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| {
            (prediction - real) / (prediction * (1.0 - prediction))
        })
        .collect()
}

fn cce(predictions: &[f64], real: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| -(real * prediction.clamp(1e-15, 1.0 - 1e-15).ln()))
        .sum()
}
fn init_gradient_cce(predictions: &[f64], real: &[f64]) -> Vec<f64> {
    predictions
        .iter()
        .zip(real.iter())
        .map(|(prediction, real)| {
            -real / prediction
        })
        .collect()
}