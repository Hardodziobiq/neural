use crate::net::{activation_function::ActivationFunction, cache::Cache, neuron::Neuron};
use rand::rngs::StdRng;

pub struct Layer {
    neurons: Vec<Neuron>,
    last_input: Vec<f64>,
    caches: Vec<Cache>,
    activation_function: ActivationFunction,
}

impl Layer {
    pub fn new(
        rng: &mut StdRng,
        bias_on: bool,
        neurons_count: usize,
        neuron_inputs_count: usize,
        activation_function: ActivationFunction,
    ) -> Self {
        let neurons: Vec<Neuron> = Self::neurons(rng, bias_on, neurons_count, neuron_inputs_count);
        let last_input: Vec<f64> = Self::last_input();
        let caches: Vec<Cache> = Self::caches(neurons_count);
        Self {
            neurons,
            last_input,
            caches,
            activation_function,
        }
    }
    fn neurons(
        rng: &mut StdRng,
        bias_on: bool,
        neurons_count: usize,
        neuron_inputs_count: usize,
    ) -> Vec<Neuron> {
        (0..neurons_count)
            .map(|_| Neuron::new(rng, neuron_inputs_count, bias_on))
            .collect()
    }
    fn last_input() -> Vec<f64> {
        Vec::new()
    }
    fn caches(neurons_count: usize) -> Vec<Cache> {
        vec![Cache::default(); neurons_count]
    }
}

impl Layer {
    pub fn process(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.last_input = input.clone();
        let mut output: Vec<f64> = Vec::with_capacity(self.neurons.len());

        for neuron in &mut self.neurons {
            output.push(neuron.process(&input));
        }
        for (idx, cache) in self.caches.iter_mut().enumerate() {
            cache.change_sum(output[idx]);
        }

        self.activation_function.process(&mut output);
        for (idx, cache) in self.caches.iter_mut().enumerate() {
            cache.change_output(output[idx]);
        }

        output
    }
    pub fn output_count(&self) -> usize {
        self.neurons.len()
    }
    pub fn change_activation_function(&mut self, activation_function: ActivationFunction) {
        self.activation_function = activation_function;
    }
}
