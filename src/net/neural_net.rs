use crate::net::{activation_function::ActivationFunction, data_set::Dataset, history::History, layer::Layer, loss::Loss};
use rand::{SeedableRng, rngs::StdRng};

pub struct NeuralNet {
    layers: Vec<Layer>,
    loss: Loss,
    learning_rate: f64,
    rng: StdRng,
}

impl NeuralNet {
    pub fn new(
        seed: u64,
        inputs_count: usize,
        bias_on: bool,
        layers_neuron_count: Vec<usize>,
        hidden_activation_function: ActivationFunction,
        output_activation_function: ActivationFunction,
        loss: Loss,
        learning_rate: f64,
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

        Self {
            layers,
            loss,
            learning_rate,
            rng
        }
    }

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }
}

impl NeuralNet {
    fn process(&mut self, mut input: Vec<f64>) -> Vec<f64> {
        for layer in &mut self.layers {
            input = layer.process(input);
        }
        input
    }

    fn process_backward(&mut self, predictions: &[f64], real: &[f64]) {
        let mut given_gradient: Vec<f64> = self.loss.calculate_init_gradient(predictions, real);
        for idx in (0..self.layers.len()).rev() {
            let need_next_gradient = idx > 0;
            given_gradient = self.layers[idx].process_backward(given_gradient, need_next_gradient);
        }
    }

    fn learn(&mut self) {
        for layer in &mut self.layers {
            layer.learn(self.learning_rate);
        }
    }

    pub fn train(
        &mut self,
        train_dataset: &Dataset,
        test_dataset: Option<&Dataset>,
        epoch_count: usize,
        batch_size: usize,
    ) -> History {
        let mut history: History = History::new();
        let mut iteration: usize = 0;

        for epoch in 1..=epoch_count {
            let shuffled_data: Dataset = train_dataset.shuffled(&mut self.rng);
            for (inputs, targets) in shuffled_data
                .features
                .chunks(batch_size)
                .zip(shuffled_data.targets.chunks(batch_size))
            {
                for (input, real) in inputs.iter().zip(targets) {
                    let predictions: Vec<f64> = self.process(input.clone());
                    self.process_backward(&predictions, real);
                }

                self.learn();
                iteration += 1;

                history.record_parameters(
                    iteration,
                    self.weights_snapshot(),
                    self.biases_snapshot(),
                );
            }

            let train_loss: f64 = self
                .test(train_dataset)
                .expect("Training dataset must not be empty");

            let test_loss: Option<f64> =
                test_dataset.and_then(|dataset| self.test(dataset));

            history.record_epoch(epoch, train_loss, test_loss);
        }

        history
    }

    pub fn test(&mut self, dataset: &Dataset) -> Option<f64> {
        if dataset.features.is_empty() || dataset.features.len() != dataset.targets.len() {
            return None;
        }

        let mut total_loss: f64 = 0.0;
        for (input, real) in dataset.features.iter().zip(&dataset.targets) {
            let predictions: Vec<f64> = self.process(input.clone());
            total_loss += self.loss.calculate(&predictions, real);
        }

        Some(total_loss / dataset.features.len() as f64)
    }

    pub fn weights_snapshot(&self) -> Vec<Vec<Vec<f64>>> {
        self.layers.iter().map(|layer| layer.weights()).collect()
    }

    pub fn biases_snapshot(&self) -> Vec<Vec<f64>> {
        self.layers.iter().map(|layer| layer.biases()).collect()
    }
}
