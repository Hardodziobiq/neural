use crate::net::{activation_function::ActivationFunction, cache::Cache, neuron::Neuron};
use rand::rngs::StdRng;

pub struct Layer {
    neurons: Vec<Neuron>,
    activation_function: ActivationFunction,
    cache: Cache,
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
        let cache: Cache = Self::caches(neurons_count, neuron_inputs_count);
        Self {
            neurons,
            activation_function,
            cache,
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
    fn caches(neurons_count: usize, neuron_inputs_count: usize) -> Cache {
        Cache::new(neurons_count, neuron_inputs_count)
    }
}

impl Layer {
    pub fn process(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.cache.update_last_input(input.clone());

        let mut output: Vec<f64> = Vec::with_capacity(self.neurons.len());
        for neuron in &self.neurons {
            output.push(neuron.process(&input));
        }

        self.cache.update_vec_z(&output);
        self.activation_function.process(&mut output);
        self.cache.update_vec_a(&output);

        output
    }

    pub fn process_backward(
        &mut self,
        given_gradient: Vec<f64>,
        need_next_gradient: bool,
    ) -> Vec<f64> {
        self.cache.update_backward(
            given_gradient,
            self.activation_function,
            &self.neurons,
            need_next_gradient,
        )
    }

    pub fn weights(&self) -> Vec<Vec<f64>> {
        self.neurons.iter().map(|n| n.weights().to_vec()).collect()
    }

    pub fn biases(&self) -> Vec<f64> {
        self.neurons.iter().map(|n| n.bias()).collect()
    }

    pub fn output_count(&self) -> usize {
        self.neurons.len()
    }

    pub fn change_activation_function(&mut self, activation_function: ActivationFunction) {
        self.activation_function = activation_function;
    }

    pub fn learn(&mut self, learning_rate: f64) {
        let samples_count = self.cache.samples_count();
        for (idx, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.learn(
                learning_rate,
                self.cache.get_single_weights_gradient(idx),
                self.cache.get_single_bias_gradient(idx),
                samples_count,
            );
        }
        self.cache.clear_gradients();
    }
}
