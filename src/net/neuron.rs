use rand::{RngExt, rngs::StdRng};

pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    bias_on: bool,
}

impl Neuron {
    pub fn new(rng: &mut StdRng, inputs_count: usize, bias_on: bool) -> Self {
        let bias: f64 = 0.0;
        let weights: Vec<f64> = Self::random_weights(rng, inputs_count);
        Self {
            weights,
            bias,
            bias_on,
        }
    }
    fn random_weights(rng: &mut StdRng, inputs_count: usize) -> Vec<f64> {
        (0..inputs_count)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect()
    }
}

impl Neuron {
    pub fn process(&mut self, input: &[f64]) -> f64 {
        let mut sum: f64 = 0.0;
        for (idx, x) in input.iter().enumerate() {
            sum += self.weights[idx] * x;
        }
        if self.bias_on {
            sum += self.bias;
        }
        sum
    }
    pub fn weights(&self) -> Vec<f64> {
        self.weights.clone()
    }
    pub fn learn(&mut self, learning_rate: f64, single_weights_gradient :&Vec<f64>, single_bias_gradient: &f64) {
        for (idx, value) in self.weights.iter_mut().enumerate() {
            *value -= learning_rate * single_weights_gradient[idx];
        }
        if self.bias_on {
            self.bias -= learning_rate * single_bias_gradient;
        }
    }
}
