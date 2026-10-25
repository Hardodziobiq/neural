use neural::net::{
    activation_function::ActivationFunction as Activation, data_set::Dataset,
    history::SnapshotPolicy, loss::Loss, neural_net::NeuralNet,
};

fn network() -> NeuralNet {
    NeuralNet::new(
        42,
        2,
        true,
        vec![3, 2],
        Activation::Tanh,
        Activation::Linear,
        Loss::MeanSquaredError,
        0.05,
    )
    .unwrap()
}

fn dataset() -> Dataset {
    Dataset {
        features: vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ],
        targets: vec![
            vec![0.0, 0.0],
            vec![1.0, -1.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![1.0, 0.0],
        ],
    }
}

#[test]
fn prediction_matches_manual_forward_calculation_for_multiple_outputs() {
    let net = network();
    let input = [0.2, -0.4];
    let weights = net.weights_snapshot();
    let biases = net.biases_snapshot();
    let hidden: Vec<f64> = weights[0]
        .iter()
        .zip(&biases[0])
        .map(|(weights, bias)| {
            (weights.iter().zip(input).map(|(w, x)| w * x).sum::<f64>() + bias).tanh()
        })
        .collect();
    let expected: Vec<f64> = weights[1]
        .iter()
        .zip(&biases[1])
        .map(|(weights, bias)| weights.iter().zip(&hidden).map(|(w, x)| w * x).sum::<f64>() + bias)
        .collect();
    assert_eq!(net.predict(&input).unwrap(), expected);
    assert_eq!(net.seed(), 42);
}

#[test]
fn prediction_rejects_wrong_dimensions_and_nonfinite_inputs() {
    let net = network();
    for input in [
        vec![],
        vec![1.0],
        vec![1.0, 2.0, 3.0],
        vec![0.0, f64::NAN],
        vec![f64::INFINITY, 0.0],
    ] {
        assert!(net.predict(&input).is_err());
    }
}

#[test]
fn prediction_and_evaluation_do_not_change_future_training() {
    let mut net = network();
    let mut control = network();
    let data = dataset();
    for _ in 0..3 {
        net.predict(&[0.3, 0.7]).unwrap();
        net.test(&data).unwrap();
    }
    net.train(&data, None, 10, 3).unwrap();
    control.train(&data, None, 10, 3).unwrap();
    assert_eq!(net.weights_snapshot(), control.weights_snapshot());
    assert_eq!(net.biases_snapshot(), control.biases_snapshot());
}

#[test]
fn default_training_keeps_epoch_metrics_without_parameter_copies() {
    let history = network().train(&dataset(), None, 3, 3).unwrap();
    assert_eq!(history.epochs.len(), 3);
    assert!(history.parameter_snapshots.is_empty());
}

#[test]
fn epoch_policy_records_initial_and_end_of_epoch_parameters() {
    let mut net = network();
    let initial = net.weights_snapshot();
    let history = net
        .train_with_history(&dataset(), None, 3, 3, SnapshotPolicy::EveryEpoch)
        .unwrap();
    let iterations: Vec<_> = history
        .parameter_snapshots
        .iter()
        .map(|snapshot| snapshot.iteration)
        .collect();
    assert_eq!(iterations, vec![0, 2, 4, 6]);
    assert_eq!(history.parameter_snapshots[0].weights, initial);
    assert_eq!(
        history.parameter_snapshots.last().unwrap().weights,
        net.weights_snapshot()
    );
}

#[test]
fn interval_policy_records_updates_across_epoch_boundaries() {
    let history = network()
        .train_with_history(&dataset(), None, 3, 3, SnapshotPolicy::EveryIterations(3))
        .unwrap();
    let iterations: Vec<_> = history
        .parameter_snapshots
        .iter()
        .map(|snapshot| snapshot.iteration)
        .collect();
    assert_eq!(iterations, vec![0, 3, 6]);
}

#[test]
fn invalid_interval_is_rejected_before_changing_model_or_random_state() {
    let mut net = network();
    let mut control = network();
    let before = net.weights_snapshot();
    assert!(
        net.train_with_history(&dataset(), None, 3, 3, SnapshotPolicy::EveryIterations(0))
            .is_err()
    );
    assert_eq!(net.weights_snapshot(), before);
    net.train(&dataset(), None, 3, 3).unwrap();
    control.train(&dataset(), None, 3, 3).unwrap();
    assert_eq!(net.weights_snapshot(), control.weights_snapshot());
    assert_eq!(net.biases_snapshot(), control.biases_snapshot());
}

#[test]
fn recording_policy_does_not_change_training_results() {
    let mut baseline = network();
    baseline.train(&dataset(), None, 3, 3).unwrap();
    for policy in [
        SnapshotPolicy::EveryEpoch,
        SnapshotPolicy::EveryIterations(1),
    ] {
        let mut net = network();
        net.train_with_history(&dataset(), None, 3, 3, policy)
            .unwrap();
        assert_eq!(net.weights_snapshot(), baseline.weights_snapshot());
        assert_eq!(net.biases_snapshot(), baseline.biases_snapshot());
    }
}

#[test]
fn zero_epochs_does_not_record_snapshots_even_when_enabled() {
    let history = network()
        .train_with_history(&dataset(), None, 0, 3, SnapshotPolicy::EveryEpoch)
        .unwrap();
    assert!(history.epochs.is_empty());
    assert!(history.parameter_snapshots.is_empty());
}

#[test]
fn mse_value_and_gradient_average_over_output_count() {
    let loss = Loss::MeanSquaredError;
    assert_eq!(loss.calculate(&[3.0], &[1.0]).unwrap(), 4.0);
    assert_eq!(
        loss.calculate_init_gradient(&[3.0], &[1.0]).unwrap(),
        vec![4.0]
    );
    assert_eq!(
        loss.calculate(&[3.0, 3.0, 3.0], &[1.0, 1.0, 1.0]).unwrap(),
        4.0
    );
    assert_eq!(
        loss.calculate_init_gradient(&[3.0, 3.0, 3.0], &[1.0, 1.0, 1.0])
            .unwrap(),
        vec![4.0 / 3.0; 3]
    );
}

#[test]
fn backward_weight_and_bias_gradients_match_numerical_mse_derivatives() {
    let mut net = network();
    let before_weights = net.weights_snapshot();
    let before_biases = net.biases_snapshot();
    let input = [0.2, -0.4];
    let target = [0.7, -0.1];
    let output_count = target.len() as f64;
    let manual_loss = |weights: &[Vec<Vec<f64>>], biases: &[Vec<f64>]| {
        let hidden: Vec<f64> = weights[0]
            .iter()
            .zip(&biases[0])
            .map(|(weights, bias)| {
                (weights.iter().zip(input).map(|(w, x)| w * x).sum::<f64>() + bias).tanh()
            })
            .collect();
        weights[1]
            .iter()
            .zip(&biases[1])
            .zip(target)
            .map(|((weights, bias), target)| {
                let prediction =
                    weights.iter().zip(&hidden).map(|(w, x)| w * x).sum::<f64>() + bias;
                (prediction - target).powi(2) / output_count
            })
            .sum::<f64>()
    };
    let data = Dataset {
        features: vec![input.to_vec()],
        targets: vec![target.to_vec()],
    };
    net.train(&data, None, 1, 1).unwrap();
    let after_weights = net.weights_snapshot();
    let after_biases = net.biases_snapshot();
    let h = 1e-6;
    for layer in 0..before_weights.len() {
        for neuron in 0..before_weights[layer].len() {
            for index in 0..before_weights[layer][neuron].len() {
                let mut plus = before_weights.clone();
                let mut minus = before_weights.clone();
                plus[layer][neuron][index] += h;
                minus[layer][neuron][index] -= h;
                let numerical = (manual_loss(&plus, &before_biases)
                    - manual_loss(&minus, &before_biases))
                    / (2.0 * h);
                let analytic = (before_weights[layer][neuron][index]
                    - after_weights[layer][neuron][index])
                    / 0.05;
                assert!((numerical - analytic).abs() < 1e-7);
            }
            let mut plus = before_biases.clone();
            let mut minus = before_biases.clone();
            plus[layer][neuron] += h;
            minus[layer][neuron] -= h;
            let numerical = (manual_loss(&before_weights, &plus)
                - manual_loss(&before_weights, &minus))
                / (2.0 * h);
            let analytic = (before_biases[layer][neuron] - after_biases[layer][neuron]) / 0.05;
            assert!((numerical - analytic).abs() < 1e-7);
        }
    }
}

#[test]
fn standardized_multioutput_regression_predicts_in_original_units() {
    let mut train = Dataset {
        features: vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]],
        targets: vec![
            vec![-3.0, 10.0],
            vec![-1.0, 7.0],
            vec![1.0, 4.0],
            vec![3.0, 1.0],
            vec![5.0, -2.0],
        ],
    };
    let feature_scaler = train.standardize_features().unwrap();
    let target_scaler = train.standardize_targets().unwrap();
    let mut net = NeuralNet::new(
        42,
        1,
        true,
        vec![2],
        Activation::Linear,
        Activation::Linear,
        Loss::MeanSquaredError,
        0.1,
    )
    .unwrap();
    net.train(&train, None, 300, 5).unwrap();
    let input = feature_scaler.transform_sample(&[1.5]).unwrap();
    let predicted = target_scaler
        .inverse_transform_sample(&net.predict(&input).unwrap())
        .unwrap();
    assert!((predicted[0] - 4.0).abs() < 1e-6);
    assert!((predicted[1] + 0.5).abs() < 1e-6);
}
