use neural::net::{
    activation_function::ActivationFunction as Activation, data_set::Dataset, loss::Loss,
    neural_net::NeuralNet,
};

#[test]
fn batch_history_has_update_epoch_and_actual_batch_size() {
    let data = Dataset {
        features: vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
        targets: vec![vec![1.0], vec![3.0], vec![5.0], vec![7.0]],
    };
    let mut net = NeuralNet::new(
        9,
        1,
        true,
        vec![1],
        Activation::Linear,
        Activation::Linear,
        Loss::MeanSquaredError,
        0.0,
    )
    .unwrap();
    let reference = net.test(&data).unwrap();
    let history = net.train(&data, None, 3, 3).unwrap();
    assert_eq!(history.iterations.len(), 6);
    for epoch in 0..3 {
        let rows = &history.iterations[epoch * 2..epoch * 2 + 2];
        assert_eq!(rows[0].samples_count, 3);
        assert_eq!(rows[1].samples_count, 1);
        assert_eq!(rows[0].iteration, epoch * 2 + 1);
        assert_eq!(rows[1].iteration, epoch * 2 + 2);
        assert_eq!(rows[0].epoch, epoch + 1);
        assert_eq!(rows[1].epoch, epoch + 1);
        let weighted = rows
            .iter()
            .map(|row| row.batch_loss * row.samples_count as f64 / 4.0)
            .sum::<f64>();
        assert!((weighted - reference).abs() < 1e-12);
    }
    assert!(history.parameter_snapshots.is_empty());
}

#[test]
fn iteration_loss_is_measured_before_the_update_and_epoch_loss_afterwards() {
    let data = Dataset {
        features: vec![vec![1.0]],
        targets: vec![vec![3.0]],
    };
    let mut net = NeuralNet::new(
        9,
        1,
        true,
        vec![1],
        Activation::Linear,
        Activation::Linear,
        Loss::MeanSquaredError,
        0.1,
    )
    .unwrap();
    let before = net.test(&data).unwrap();
    let history = net.train(&data, None, 1, 1).unwrap();
    assert_eq!(history.iterations[0].batch_loss, before);
    assert_eq!(history.epochs[0].train_loss, net.test(&data).unwrap());
    assert!(history.epochs[0].train_loss < before);
}
