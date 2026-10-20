use crate::net::{activation_function::ActivationFunction, layer::Layer};
use rand::{SeedableRng, rngs::StdRng};

pub struct NeuralNet {
    layers: Vec<Layer>,
}

impl NeuralNet {
    pub fn new(
        seed: u64,
        inputs_count: usize,
        bias_on: bool,
        layers_neuron_count: Vec<usize>,
        hidden_activation_function: ActivationFunction,
        output_activation_function: ActivationFunction,
    ) -> Self {
        let mut layers: Vec<Layer> = Vec::new();
        let mut rng: StdRng = Self::rng(seed);
        let mut neuron_inputs_count: usize = inputs_count;

        for layer_neurons_count in layers_neuron_count {
            let layer: Layer = Layer::new(
                &mut rng,
                bias_on,
                layer_neurons_count,
                neuron_inputs_count,
                hidden_activation_function,
            );
            neuron_inputs_count = layer.output_count();
            layers.push(layer);
        }

        if let Some(output_layer) = layers.last_mut() {
            output_layer.change_activation_function(output_activation_function);
        }

        Self { layers }
    }

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }
}

impl NeuralNet {
    pub fn process(&mut self, mut input: Vec<f64>) -> Vec<f64> {
        for layer in &mut self.layers {
            input = layer.process(input);
        }
        input
    }
}
