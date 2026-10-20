use crate::net::{activation_function::ActivationFunction, neuron::Neuron};

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
    samples_count: usize,
}

impl Cache {
    pub fn new(neurons_count: usize, neuron_inputs_count: usize) -> Self {
        Self {
            last_input: vec![0.0; neuron_inputs_count],
            vec_z: vec![0.0; neurons_count],
            vec_a: vec![0.0; neurons_count],
            given_gradient: Vec::new(),
            delta: vec![0.0; neurons_count],
            weights_gradient: vec![vec![0.0; neuron_inputs_count]; neurons_count],
            bias_gradient: vec![0.0; neurons_count],
            next_gradient: vec![0.0; neuron_inputs_count],
            samples_count: 0,
        }
    }
}

/// FORWARD PHASE
impl Cache {
    pub fn update_last_input(&mut self, new_last_input: Vec<f64>) {
        self.last_input = new_last_input;
    }
    pub fn update_vec_z(&mut self, new_output: &[f64]) {
        for (idx, value) in self.vec_z.iter_mut().enumerate() {
            *value = new_output[idx];
        }
    }
    pub fn update_vec_a(&mut self, new_output: &[f64]) {
        for (idx, value) in self.vec_a.iter_mut().enumerate() {
            *value = new_output[idx];
        }
    }
}

/// BACKWARD PHASE
impl Cache {
    pub fn update_backward(
        &mut self,
        given_gradient: Vec<f64>,
        activation_function: ActivationFunction,
        neurons: &[Neuron],
        need_next_gradient: bool,
    ) -> Vec<f64> {
        self.given_gradient = given_gradient;
        self.compute_delta(activation_function);
        self.accumulate_weights_gradient();
        self.accumulate_bias_gradient();
        self.samples_count += 1;

        if need_next_gradient {
            self.compute_next_gradient(neurons)
        } else {
            Vec::new()
        }
    }

    fn compute_delta(&mut self, activation_function: ActivationFunction) {
        self.delta =
            activation_function.compute_delta(&self.given_gradient, &self.vec_z, &self.vec_a);
    }

    fn accumulate_weights_gradient(&mut self) {
        for (neuron_idx, delta_i) in self.delta.iter().enumerate() {
            for (input_idx, input_val) in self.last_input.iter().enumerate() {
                self.weights_gradient[neuron_idx][input_idx] += delta_i * input_val;
            }
        }
    }

    fn accumulate_bias_gradient(&mut self) {
        for (neuron_idx, delta_i) in self.delta.iter().enumerate() {
            self.bias_gradient[neuron_idx] += delta_i;
        }
    }

    fn compute_next_gradient(&mut self, neurons: &[Neuron]) -> Vec<f64> {
        self.next_gradient.fill(0.0);
        if self.next_gradient.len() != self.last_input.len() {
            self.next_gradient.resize(self.last_input.len(), 0.0);
        }

        for (neuron_idx, neuron) in neurons.iter().enumerate() {
            let delta_i = self.delta[neuron_idx];
            for (input_idx, weight_val) in neuron.weights().iter().enumerate() {
                self.next_gradient[input_idx] += delta_i * weight_val;
            }
        }
        self.next_gradient.clone()
    }
}

/// LEARN PHASE
impl Cache {
    pub fn get_single_weights_gradient(&self, idx: usize) -> &[f64] {
        &self.weights_gradient[idx]
    }
    pub fn get_single_bias_gradient(&self, idx: usize) -> f64 {
        self.bias_gradient[idx]
    }
    pub fn samples_count(&self) -> usize {
        self.samples_count
    }
    pub fn clear_gradients(&mut self) {
        for row in &mut self.weights_gradient {
            row.fill(0.0);
        }
        self.bias_gradient.fill(0.0);
        self.samples_count = 0;
    }
}
