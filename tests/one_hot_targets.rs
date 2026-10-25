use neural::net::data_set::Dataset;

fn dataset(targets: Vec<Vec<f64>>) -> Dataset {
    Dataset {
        features: (0..targets.len()).map(|index| vec![index as f64]).collect(),
        targets,
    }
}

// Compare bit patterns so that unchanged NaN labels can also be checked.
fn target_bits(data: &Dataset) -> Vec<Vec<u64>> {
    data.targets
        .iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

#[test]
fn encodes_all_classes_without_changing_features() {
    let mut data = dataset(vec![vec![0.0], vec![1.0], vec![2.0], vec![1.0]]);
    let features = data.features.clone();

    data.one_hot_encode_targets(3).unwrap();

    assert_eq!(
        data.targets,
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ],
    );
    assert_eq!(data.features, features);
    data.validate(1, 3).unwrap();
}

#[test]
fn zero_class_count_leaves_targets_unchanged() {
    let mut data = dataset(vec![vec![0.0], vec![1.0]]);
    let before = target_bits(&data);

    assert!(data.one_hot_encode_targets(0).is_err());
    assert_eq!(target_bits(&data), before);
}

#[test]
fn empty_targets_are_rejected() {
    let mut data = dataset(vec![]);

    assert!(data.one_hot_encode_targets(3).is_err());
    assert!(data.targets.is_empty());
}

#[test]
fn invalid_scalar_labels_do_not_partially_encode_the_dataset() {
    for invalid in [
        -1.0,
        0.5,
        3.0,
        9.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let mut data = dataset(vec![vec![0.0], vec![invalid], vec![2.0]]);
        let before = target_bits(&data);
        let features = data.features.clone();

        let error = data.one_hot_encode_targets(3).unwrap_err();

        assert!(error.to_string().contains("Targets row 1"));
        assert!(error.to_string().contains("0..3"));
        assert_eq!(target_bits(&data), before);
        assert_eq!(data.features, features);
    }
}

#[test]
fn empty_or_multivalue_labels_do_not_partially_encode_the_dataset() {
    for invalid in [vec![], vec![1.0, 2.0], vec![0.0, 0.0, 1.0]] {
        let mut data = dataset(vec![vec![0.0], invalid, vec![2.0]]);
        let before = target_bits(&data);

        let error = data.one_hot_encode_targets(3).unwrap_err();

        assert!(error.to_string().contains("Targets row 1"));
        assert_eq!(target_bits(&data), before);
    }
}

#[test]
fn repeated_encoding_returns_an_error_without_changing_classes() {
    let mut data = dataset(vec![vec![2.0], vec![0.0], vec![1.0]]);
    data.one_hot_encode_targets(3).unwrap();
    let before = target_bits(&data);

    assert!(data.one_hot_encode_targets(3).is_err());

    assert_eq!(target_bits(&data), before);
    assert_eq!(data.targets[0], vec![0.0, 0.0, 1.0]);
}

#[test]
fn single_class_can_be_encoded_and_repeated_encoding_is_rejected() {
    let mut data = dataset(vec![vec![0.0], vec![0.0]]);

    data.one_hot_encode_targets(1).unwrap();
    assert_eq!(data.targets, vec![vec![1.0], vec![1.0]]);
    let before = target_bits(&data);

    assert!(data.one_hot_encode_targets(1).is_err());
    assert_eq!(target_bits(&data), before);
}
