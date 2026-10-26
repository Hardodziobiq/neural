use rand::{RngExt, rngs::StdRng};

pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    bias_on: bool,
}

impl Neuron {
    /// Creates random input weights and a zero bias, remembering whether the bias is enabled.
    pub fn new(rng: &mut StdRng, inputs_count: usize, bias_on: bool) -> Self {
        let bias: f64 = 0.0;
        let weights: Vec<f64> = Self::random_weights(rng, inputs_count);
        Self {
            weights,
            bias,
            bias_on,
        }
    }
    /// Draws one independent weight per input uniformly from `[-1, 1)` using the supplied RNG.
    fn random_weights(rng: &mut StdRng, inputs_count: usize) -> Vec<f64> {
        (0..inputs_count)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect()
    }
}

impl Neuron {
    /// Returns the weighted input sum plus the enabled bias; the input must match the weight count.
    pub fn process(&self, input: &[f64]) -> f64 {
        let mut sum: f64 = 0.0;
        for (idx, x) in input.iter().enumerate() {
            sum += self.weights[idx] * x;
        }
        if self.bias_on {
            sum += self.bias;
        }
        sum
    }

    /// Borrows the neuron's weight vector without copying it.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Returns the stored bias value, including when bias use is disabled.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Subtracts learning-rate-scaled average batch gradients; an empty batch leaves parameters unchanged.
    pub fn learn(
        &mut self,
        learning_rate: f64,
        weights_gradient_sum: &[f64],
        bias_gradient_sum: f64,
        samples_count: usize,
    ) {
        if samples_count == 0 {
            return;
        }
        let step: f64 = learning_rate / samples_count as f64;

        for (idx, value) in self.weights.iter_mut().enumerate() {
            *value -= step * weights_gradient_sum[idx];
        }
        if self.bias_on {
            self.bias -= step * bias_gradient_sum;
        }
    }
}
