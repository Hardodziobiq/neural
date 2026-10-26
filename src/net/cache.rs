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
    /// Allocates forward-pass buffers and zeroed batch-gradient accumulators for a layer.
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
    /// Stores the latest sample input for computing weight gradients during backpropagation.
    pub fn update_last_input(&mut self, new_last_input: Vec<f64>) {
        self.last_input = new_last_input;
    }
    /// Copies the latest sums before activation into the preallocated cache.
    pub fn update_vec_z(&mut self, new_output: &[f64]) {
        for (idx, value) in self.vec_z.iter_mut().enumerate() {
            *value = new_output[idx];
        }
    }
    /// Copies the latest outputs after activation for use by activation derivatives.
    pub fn update_vec_a(&mut self, new_output: &[f64]) {
        for (idx, value) in self.vec_a.iter_mut().enumerate() {
            *value = new_output[idx];
        }
    }
}

/// BACKWARD PHASE
impl Cache {
    /// Accumulates one sample's parameter gradients and optionally returns the gradient for the preceding layer.
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

    /// Computes the loss gradient before activation from the incoming gradient and cached forward values.
    fn compute_delta(&mut self, activation_function: ActivationFunction) {
        self.delta =
            activation_function.compute_delta(&self.given_gradient, &self.vec_z, &self.vec_a);
    }

    /// Adds `delta * input` to each weight gradient; parameters remain unchanged.
    fn accumulate_weights_gradient(&mut self) {
        for (neuron_idx, delta_i) in self.delta.iter().enumerate() {
            for (input_idx, input_val) in self.last_input.iter().enumerate() {
                self.weights_gradient[neuron_idx][input_idx] += delta_i * input_val;
            }
        }
    }

    /// Adds each neuron's delta to its accumulated bias gradient.
    fn accumulate_bias_gradient(&mut self) {
        for (neuron_idx, delta_i) in self.delta.iter().enumerate() {
            self.bias_gradient[neuron_idx] += delta_i;
        }
    }

    /// Returns `W^T * delta`, the loss gradient with respect to this layer's inputs.
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
    /// Borrows the accumulated weight-gradient row for one neuron.
    pub fn get_single_weights_gradient(&self, idx: usize) -> &[f64] {
        &self.weights_gradient[idx]
    }
    /// Returns the accumulated bias gradient for one neuron.
    pub fn get_single_bias_gradient(&self, idx: usize) -> f64 {
        self.bias_gradient[idx]
    }
    /// Returns how many backward passes contributed to the current batch gradients.
    pub fn samples_count(&self) -> usize {
        self.samples_count
    }
    /// Resets parameter-gradient sums and the sample counter for the next batch.
    pub fn clear_gradients(&mut self) {
        for row in &mut self.weights_gradient {
            row.fill(0.0);
        }
        self.bias_gradient.fill(0.0);
        self.samples_count = 0;
    }
}
