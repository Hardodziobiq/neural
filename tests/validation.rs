use neural::net::{
    activation_function::ActivationFunction, data_set::Dataset, error::ValidationError,
    history::SnapshotPolicy, loss::Loss, neural_net::NeuralNet,
};

/// Builds a small linear-regression fixture with known inputs and targets.
fn dataset() -> Dataset {
    Dataset {
        features: vec![vec![-1.0], vec![0.0], vec![1.0], vec![2.0]],
        targets: vec![vec![-1.0], vec![1.0], vec![3.0], vec![5.0]],
    }
}

/// Constructs a seeded linear network while allowing invalid constructor arguments to be tested.
fn configured_network(
    inputs_count: usize,
    layers: Vec<usize>,
    learning_rate: f64,
) -> Result<NeuralNet, ValidationError> {
    NeuralNet::new(
        5,
        inputs_count,
        true,
        layers,
        ActivationFunction::Linear,
        ActivationFunction::Linear,
        Loss::MeanSquaredError,
        learning_rate,
    )
}

/// Returns the valid default network used by validation tests.
fn network() -> NeuralNet {
    configured_network(1, vec![1], 0.1).unwrap()
}

/// Checks rejection of empty datasets and unequal feature/target row counts.
#[test]
fn dataset_requires_nonempty_matching_sample_counts() {
    let empty = Dataset {
        features: vec![],
        targets: vec![],
    };
    assert!(empty.validate(1, 1).is_err());

    for targets in [vec![], vec![vec![1.0], vec![2.0]]] {
        let invalid = Dataset {
            features: vec![vec![0.0]],
            targets,
        };
        assert!(invalid.validate(1, 1).is_err());
    }
}

/// Checks every row, including late short or long feature and target rows.
#[test]
fn dataset_checks_dimensions_of_every_row() {
    for invalid_row in [vec![], vec![1.0, 2.0]] {
        let mut invalid_features = dataset();
        invalid_features.features[3] = invalid_row.clone();
        let error = invalid_features.validate(1, 1).unwrap_err();
        assert!(error.to_string().contains("Features row 3"));

        let mut invalid_targets = dataset();
        invalid_targets.targets[3] = invalid_row;
        let error = invalid_targets.validate(1, 1).unwrap_err();
        assert!(error.to_string().contains("Targets row 3"));
    }
}

/// Checks rejection of NaN and both infinities in either dataset matrix.
#[test]
fn dataset_rejects_nonfinite_features_and_targets() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut invalid = dataset();
        invalid.features[3][0] = value;
        assert!(invalid.validate(1, 1).is_err());

        let mut invalid = dataset();
        invalid.targets[3][0] = value;
        assert!(invalid.validate(1, 1).is_err());
    }
}

/// Checks valid matrix dimensions and rejection of zero expected input or output sizes.
#[test]
fn dataset_accepts_multiple_inputs_and_outputs() {
    let data = Dataset {
        features: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        targets: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
    };
    assert!(data.validate(2, 3).is_ok());
    assert!(data.validate(0, 3).is_err());
    assert!(data.validate(2, 0).is_err());
}

/// Checks rejection of zero inputs, no layers, and zero-width layers.
#[test]
fn constructor_rejects_empty_network_dimensions() {
    assert!(configured_network(0, vec![1], 0.1).is_err());
    for layers in [vec![], vec![0], vec![2, 0, 1]] {
        assert!(configured_network(1, layers, 0.1).is_err());
    }
}

/// Checks invalid learning rates while allowing zero for deliberately frozen parameters.
#[test]
fn constructor_rejects_negative_or_nonfinite_learning_rates() {
    for rate in [-0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(configured_network(1, vec![1], rate).is_err());
    }
    assert!(configured_network(1, vec![1], 0.0).is_ok());
}

/// Checks that failed dataset validation preserves model parameters and subsequent training order.
#[test]
fn invalid_training_data_does_not_change_parameters_or_random_state() {
    let mut net = network();
    let mut control = network();
    let before_weights = net.weights_snapshot();
    let before_biases = net.biases_snapshot();
    let mut invalid = dataset();
    invalid.targets[3].push(9.0);

    assert!(net.train(&invalid, None, 1, 1).is_err());
    assert_eq!(net.weights_snapshot(), before_weights);
    assert_eq!(net.biases_snapshot(), before_biases);

    let valid = dataset();
    let history = net
        .train_with_history(&valid, None, 2, 1, SnapshotPolicy::EveryIterations(1))
        .unwrap();
    let expected = control
        .train_with_history(&valid, None, 2, 1, SnapshotPolicy::EveryIterations(1))
        .unwrap();
    assert_eq!(history.parameter_snapshots.len(), 9);
    assert_eq!(expected.parameter_snapshots.len(), 9);
    for (actual, expected) in history
        .parameter_snapshots
        .iter()
        .zip(&expected.parameter_snapshots)
    {
        assert_eq!(actual.weights, expected.weights);
        assert_eq!(actual.biases, expected.biases);
    }
}

/// Checks that an invalid optional evaluation set fails before the first training update.
#[test]
fn invalid_test_data_is_rejected_before_training() {
    let mut net = network();
    let weights = net.weights_snapshot();
    let biases = net.biases_snapshot();
    let mut invalid = dataset();
    invalid.features[3][0] = f64::NAN;

    assert!(net.train(&dataset(), Some(&invalid), 2, 1).is_err());
    assert_eq!(net.weights_snapshot(), weights);
    assert_eq!(net.biases_snapshot(), biases);
}

/// Checks that invalid batch size and absent training samples return errors without panicking.
#[test]
fn zero_batch_size_and_empty_training_data_return_errors() {
    let mut net = network();
    assert!(net.train(&dataset(), None, 1, 0).is_err());
    let empty = Dataset {
        features: vec![],
        targets: vec![],
    };
    assert!(net.train(&empty, None, 1, 1).is_err());
}

/// Checks validation at the evaluation boundary and preservation of model parameters.
#[test]
fn evaluation_reports_invalid_data_without_changing_parameters() {
    let net = network();
    let weights = net.weights_snapshot();
    let biases = net.biases_snapshot();
    let mut invalid = dataset();
    invalid.features[3].clear();
    assert!(net.test(&invalid).is_err());
    assert_eq!(net.weights_snapshot(), weights);
    assert_eq!(net.biases_snapshot(), biases);
}

/// Checks the supported no-update cases for zero epochs and a zero learning rate.
#[test]
fn zero_epochs_and_zero_learning_rate_leave_parameters_unchanged() {
    let data = dataset();
    let mut net = network();
    let weights = net.weights_snapshot();
    let biases = net.biases_snapshot();
    let history = net.train(&data, None, 0, 2).unwrap();
    assert!(history.is_empty());
    assert!(history.parameter_snapshots.is_empty());
    assert_eq!(net.weights_snapshot(), weights);
    assert_eq!(net.biases_snapshot(), biases);

    let mut net = configured_network(1, vec![1], 0.0).unwrap();
    let weights = net.weights_snapshot();
    let biases = net.biases_snapshot();
    net.train(&data, None, 2, 3).unwrap();
    assert_eq!(net.weights_snapshot(), weights);
    assert_eq!(net.biases_snapshot(), biases);
}

/// Returns each implemented loss so shared validation and gradient checks cover all variants.
fn losses() -> [Loss; 4] {
    [
        Loss::MeanSquaredError,
        Loss::MeanAbsoluteError,
        Loss::BinaryCrossEntropy,
        Loss::CategorialCrossEntropy,
    ]
}

/// Checks shape validation for loss values and their initial prediction gradients.
#[test]
fn losses_reject_empty_and_mismatched_vectors_in_both_operations() {
    for loss in losses() {
        for (predictions, real) in [
            (vec![], vec![]),
            (vec![1.0], vec![1.0, 100.0]),
            (vec![1.0, 100.0], vec![1.0]),
        ] {
            assert!(loss.calculate(&predictions, &real).is_err());
            assert!(loss.calculate_init_gradient(&predictions, &real).is_err());
        }
    }
}

/// Checks finite-value validation for both loss APIs.
#[test]
fn losses_reject_nonfinite_values_in_both_operations() {
    for loss in losses() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for (predictions, real) in [([0.2, value], [0.0, 1.0]), ([0.2, 0.8], [0.0, value])] {
                assert!(loss.calculate(&predictions, &real).is_err());
                assert!(loss.calculate_init_gradient(&predictions, &real).is_err());
            }
        }
    }
}

/// Checks analytic loss gradients against numerical derivatives away from clipping boundaries.
#[test]
fn valid_loss_gradients_match_finite_differences() {
    let predictions = [0.2, 0.7];
    let real = [0.0, 1.0];
    let h = 1e-6;
    for loss in losses() {
        let gradient = loss.calculate_init_gradient(&predictions, &real).unwrap();
        for index in 0..predictions.len() {
            let mut plus = predictions;
            let mut minus = predictions;
            plus[index] += h;
            minus[index] -= h;
            let numerical = (loss.calculate(&plus, &real).unwrap()
                - loss.calculate(&minus, &real).unwrap())
                / (2.0 * h);
            assert!((numerical - gradient[index]).abs() < 1e-7);
        }
    }
    // Lecture MSE is half the sum of output errors squared.
    assert_eq!(
        Loss::MeanSquaredError
            .calculate(&[1.0, 3.0, 4.0], &[0.0, 1.0, 1.0])
            .unwrap(),
        7.0
    );
}

/// Checks convergence, partial-batch averaging, and recorded evaluation metrics in a complete training run.
#[test]
fn training_still_learns_with_a_partial_last_batch_and_test_metrics() {
    let data = dataset();
    let mut net = network();
    let before = net.test(&data).unwrap();
    let history = net.train(&data, Some(&data), 200, 3).unwrap();
    let after = net.test(&data).unwrap();

    assert!(after < before && after < 1e-8);
    assert_eq!(history.len(), 200);
    assert!(history.parameter_snapshots.is_empty());
    let last = history.epochs.last().unwrap();
    assert_eq!(last.train_loss, after);
    assert_eq!(last.test_loss, Some(after));
}
