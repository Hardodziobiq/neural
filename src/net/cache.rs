use crate::net::activation_function::ActivationFunction;

#[derive(Clone)]
pub struct Cache {
    /// FORWARD
    last_input: Vec<f64>,
    vec_z: Vec<f64>,
    vec_a: Vec<f64>,
    /// BACKWARD
    given_gradient: Vec<f64>,
    delta: Vec<f64>,
    weights_gradient: Vec<Vec<f64>>,
    bias_gradient: Vec<f64>,
    next_gradient: Vec<f64>,
}

impl Cache {
    pub fn new(neurons_count: usize) -> Self {
        Self {
            last_input: Vec::new(),
            vec_z: vec![0.0; neurons_count],
            vec_a: vec![0.0; neurons_count],
            given_gradient: Vec::new(),
            delta: Vec::new(),
            weights_gradient: Vec::new(),
            bias_gradient: Vec::new(),
            next_gradient: Vec::new(),
        }
    }
}

/// FORWARD PHASE
impl Cache {
    pub fn update_last_input(&mut self, new_last_input: Vec<f64>) {
        self.last_input = new_last_input;
    }
    pub fn update_vec_z(&mut self, new_output: &[f64]) {
        for (idx, value) in &mut self.vec_z.iter_mut().enumerate() {
            *value = new_output[idx];
        }
    }
    pub fn update_vec_a(&mut self, new_output: &[f64]) {
        for (idx, value) in &mut self.vec_a.iter_mut().enumerate() {
            *value = new_output[idx];
        }
    }
}

/// BACKWARD PHASE
impl Cache {
    pub fn update_backward(&mut self, given_gradient: Vec<f64>, activation_function: ActivationFunction, weights: Vec<Vec<f64>>) -> Vec<f64> {
        self.given_gradient = given_gradient;
        self.compute_delta(activation_function);
        self.compute_weights_gradient();
        self.compute_bias_gradient();
        self.compute_next_gradient(weights)
    }
    fn compute_delta(&mut self, activation_function: ActivationFunction) {
        let diff: Vec<f64> = activation_function.backward_diff(&self.vec_z, &self.vec_a);
        if self.delta.len() == 0 {
            self.delta = vec![0.0; diff.len()];
        }
        self.delta
            .iter_mut()
            .enumerate()
            .for_each(|(idx, val)| {
                *val = self.given_gradient[idx] * diff[idx];
        });
    }
    fn compute_weights_gradient(&mut self) {
        self.weights_gradient = self.delta
            .iter()
            .map(|delta_i| {
                self.last_input
                    .iter()
                    .map(|val| delta_i * val)
                    .collect()
            })
            .collect();
    }
    fn compute_bias_gradient(&mut self) {
        self.bias_gradient = self.delta.clone();
    }
    fn compute_next_gradient(&mut self, weights: Vec<Vec<f64>>) -> Vec<f64> {
        self.next_gradient = vec![0.0; self.given_gradient.len()];
        for (weight_idx, values) in weights.iter().enumerate() {
            for (idx, val) in values.iter().enumerate() {
                self.next_gradient[idx] += self.delta[weight_idx] * val;
            }
        }
        self.next_gradient.clone()
    }
}

/// LEARN PHASE
impl Cache {
    pub fn get_single_weights_gradient(&self, idx: usize) -> &Vec<f64> {
        &self.weights_gradient[idx]
    }
    pub fn get_single_bias_gradient(&self, idx: usize) -> &f64 {
        &self.bias_gradient[idx]
    }
}