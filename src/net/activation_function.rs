const STANDARD_LINEAR_ALFA: f64 = 1.0;
const STANDARD_LINEAR_BETA: f64 = 0.0;
const STANDARD_SIGMOID_ALFA: f64 = 1.0;

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub enum ActivationFunction {
    Linear,
    Sigmoid,
    Tanh,
    Relu,
    SoftMax,
}

impl ActivationFunction {
    pub fn process(&self, values: &mut [f64]) {
        match self {
            ActivationFunction::Linear => linear(values),
            ActivationFunction::Sigmoid => sigmoid(values),
            ActivationFunction::Tanh => tanh(values),
            ActivationFunction::Relu => relu(values),
            ActivationFunction::SoftMax => soft_max(values),
        }
    }
}

fn linear(values: &mut [f64]) {
    values.iter_mut().for_each(|value| {
        *value = STANDARD_LINEAR_ALFA * *value + STANDARD_LINEAR_BETA;
    });
}
fn sigmoid(values: &mut [f64]) {
    values.iter_mut().for_each(|value| {
        *value = 1.0 / (1.0 + (-*value * STANDARD_SIGMOID_ALFA).exp());
    });
}
fn tanh(values: &mut [f64]) {
    values.iter_mut().for_each(|value| {
        *value = value.tanh();
    });
}
fn relu(values: &mut [f64]) {
    values.iter_mut().for_each(|value| {
        *value = value.max(0.0);
    });
}

fn soft_max(values: &mut [f64]) {
    let max: f64 = values
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .copied()
        .unwrap_or(0.0);
    let sum: f64 = values.iter().map(|value| (*value - max).exp()).sum();
    for value in values {
        *value = (*value - max).exp() / sum;
    }
}
// fn diff_linear(value: f64) -> f64 {
//     STANDARD_LINEAR_ALFA
// }

// fn diff_sigmoid(value: f64) -> f64 {
//     let sigmoid_value: f64 = sigmoid(value);
//     STANDARD_SIGMOID_ALFA * sigmoid_value * (1.0 - sigmoid_value)
// }

// fn diff_tanh(value: f64) -> f64 {
//     let tanh: f64 = tanh(value);
//     1.0 - tanh * tanh
// }

// fn diff_relu(value: f64) -> f64 {
//     // RELU IS NOT DIFFERENTIAL IN X = 0
//     if value < 0.0 {
//         return 0.0;
//     }
//     return 1.0;
// }
