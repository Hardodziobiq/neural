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
        sum += self.bias;
        sum
    }
}
