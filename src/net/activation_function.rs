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
    pub fn backward_diff(&self, vec_z: &[f64], vec_a: &[f64]) -> Vec<f64> {
        match self {
            ActivationFunction::Linear => diff_linear(vec_z),
            ActivationFunction::Sigmoid => diff_sigmoid(vec_a),
            ActivationFunction::Tanh => diff_tanh(vec_a),
            ActivationFunction::Relu => diff_relu(vec_z),
            ActivationFunction::SoftMax => diff_softmax(0, vec_a), // PROBLEM
        }
    }
}

fn linear(output: &mut [f64]) {
    output.iter_mut().for_each(|value| {
        *value = STANDARD_LINEAR_ALFA * *value + STANDARD_LINEAR_BETA;
    });
}
fn diff_linear(vec_z: &[f64]) -> Vec<f64> {
    vec![STANDARD_LINEAR_ALFA; vec_z.len()]
}

fn sigmoid(output: &mut [f64]) {
    output.iter_mut().for_each(|value| {
        *value = 1.0 / (1.0 + (-*value * STANDARD_SIGMOID_ALFA).exp());
    });
}
fn diff_sigmoid(vec_a: &[f64]) -> Vec<f64> {
    vec_a
        .iter()
        .map(|a| {
           a * (1.0 - a) 
        })
        .collect()
}


fn tanh(output: &mut [f64]) {
    output.iter_mut().for_each(|value| {
        *value = value.tanh();
    });
}
fn diff_tanh(vec_a: &[f64]) -> Vec<f64> {
    vec_a
        .iter()
        .map(|a| {
           1.0 - a.powi(2)
        })
        .collect()
}

fn relu(output: &mut [f64]) {
    output.iter_mut().for_each(|value| {
        *value = value.max(0.0);
    });
}
fn diff_relu(vec_z: &[f64]) -> Vec<f64> {
    let mut new_vec: Vec<f64> = Vec::new();
    for z in vec_z {
        if 0.0 < *z {
            new_vec.push(1.0);
        }
        else {
            new_vec.push(0.0);
        }
    }
    new_vec
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
fn diff_softmax(k: usize, vec_a: &[f64]) -> Vec<f64> {
    vec_a
        .iter()
        .enumerate()
        .map(|(i, &a_i)| {
            if k == i {
                vec_a[k] * (1.0 - vec_a[k])
            } 
            else {
                -vec_a[k] * a_i
            }
        })
        .collect()
}
