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
    pub fn process(&self, output: &mut [f64]) {
        match self {
            ActivationFunction::Linear => linear(output),
            ActivationFunction::Sigmoid => sigmoid(output),
            ActivationFunction::Tanh => tanh(output),
            ActivationFunction::Relu => relu(output),
            ActivationFunction::SoftMax => soft_max(output),
        }
    }

    pub fn compute_delta(&self, given_gradient: &[f64], vec_z: &[f64], vec_a: &[f64]) -> Vec<f64> {
        match self {
            ActivationFunction::Linear => delta_linear(given_gradient),
            ActivationFunction::Sigmoid => delta_sigmoid(given_gradient, vec_a),
            ActivationFunction::Tanh => delta_tanh(given_gradient, vec_a),
            ActivationFunction::Relu => delta_relu(given_gradient, vec_z),
            ActivationFunction::SoftMax => delta_softmax(given_gradient, vec_a),
        }
    }
}

fn linear(output: &mut [f64]) {
    output.iter_mut().for_each(|value| {
        *value = STANDARD_LINEAR_ALFA * *value + STANDARD_LINEAR_BETA;
    });
}
fn delta_linear(given_gradient: &[f64]) -> Vec<f64> {
    given_gradient
        .iter()
        .map(|g| g * STANDARD_LINEAR_ALFA)
        .collect()
}

fn sigmoid(output: &mut [f64]) {
    output.iter_mut().for_each(|value| {
        *value = 1.0 / (1.0 + (-*value * STANDARD_SIGMOID_ALFA).exp());
    });
}
fn delta_sigmoid(given_gradient: &[f64], vec_a: &[f64]) -> Vec<f64> {
    given_gradient
        .iter()
        .zip(vec_a.iter())
        .map(|(g, a)| g * STANDARD_SIGMOID_ALFA * a * (1.0 - a))
        .collect()
}

fn tanh(output: &mut [f64]) {
    output.iter_mut().for_each(|value| {
        *value = value.tanh();
    });
}
fn delta_tanh(given_gradient: &[f64], vec_a: &[f64]) -> Vec<f64> {
    given_gradient
        .iter()
        .zip(vec_a.iter())
        .map(|(g, a)| g * (1.0 - a.powi(2)))
        .collect()
}

fn relu(output: &mut [f64]) {
    output.iter_mut().for_each(|value| {
        *value = value.max(0.0);
    });
}
fn delta_relu(given_gradient: &[f64], vec_z: &[f64]) -> Vec<f64> {
    given_gradient
        .iter()
        .zip(vec_z.iter())
        .map(|(g, z)| if *z > 0.0 { *g } else { 0.0 })
        .collect()
}

fn soft_max(output: &mut [f64]) {
    let max: f64 = output
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .copied()
        .unwrap_or(0.0);
    let sum: f64 = output.iter().map(|value| (*value - max).exp()).sum();
    for value in output {
        *value = (*value - max).exp() / sum;
    }
}
fn delta_softmax(given_gradient: &[f64], vec_a: &[f64]) -> Vec<f64> {
    let dot_product: f64 = vec_a
        .iter()
        .zip(given_gradient.iter())
        .map(|(a_k, g_k)| a_k * g_k)
        .sum();

    vec_a
        .iter()
        .zip(given_gradient.iter())
        .map(|(a_i, g_i)| a_i * (g_i - dot_product))
        .collect()
}
