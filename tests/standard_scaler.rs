use neural::net::{data_set::Dataset, standard_scaler::StandardScaler};

/// Compares matrix shapes and values within a small floating-point tolerance.
fn assert_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }
    }
}

/// Checks that evaluation and new inputs use the original fitted training statistics.
#[test]
fn test_and_new_inputs_reuse_training_statistics() {
    let training = vec![vec![0.0], vec![2.0]];
    let scaler = StandardScaler::fit(&training).unwrap();
    assert_eq!(scaler.means(), &[1.0]);
    assert_eq!(scaler.standard_deviations(), &[1.0]);
    let mut test = vec![vec![10.0], vec![12.0]];
    scaler.transform(&mut test).unwrap();
    assert_eq!(test, vec![vec![9.0], vec![11.0]]);
    assert_eq!(scaler.transform_sample(&[10.0]).unwrap(), test[0]);
    scaler.inverse_transform(&mut test).unwrap();
    assert_eq!(test, vec![vec![10.0], vec![12.0]]);
    assert_eq!(scaler.means(), &[1.0]);
}

/// Checks independent column statistics and forward/inverse transforms.
#[test]
fn fits_each_column_using_population_standard_deviation() {
    let original = vec![vec![1.0, 10.0], vec![3.0, 14.0], vec![5.0, 18.0]];
    let scaler = StandardScaler::fit(&original).unwrap();
    assert_eq!(scaler.means(), &[3.0, 14.0]);
    let std = (8.0_f64 / 3.0).sqrt();
    assert!((scaler.standard_deviations()[0] - std).abs() < 1e-12);
    assert!((scaler.standard_deviations()[1] - 2.0 * std).abs() < 1e-12);
    let mut transformed = original.clone();
    scaler.transform(&mut transformed).unwrap();
    assert_close(
        &transformed,
        &[
            vec![-2.0 / std, -2.0 / std],
            vec![0.0, 0.0],
            vec![2.0 / std, 2.0 / std],
        ],
    );
    scaler.inverse_transform(&mut transformed).unwrap();
    assert_close(&transformed, &original);
}

/// Checks scale-one handling of zero-variance columns and single-sample datasets.
#[test]
fn constant_columns_and_single_sample_remain_reversible() {
    let scaler = StandardScaler::fit(&[vec![5.0, 2.0], vec![5.0, 4.0]]).unwrap();
    assert_eq!(scaler.standard_deviations(), &[0.0, 1.0]);
    let transformed = scaler.transform_sample(&[7.0, 5.0]).unwrap();
    assert_eq!(transformed, vec![2.0, 2.0]);
    assert_eq!(
        scaler.inverse_transform_sample(&transformed).unwrap(),
        vec![7.0, 5.0]
    );

    let scaler = StandardScaler::fit(&[vec![42.0]]).unwrap();
    assert_eq!(scaler.transform_sample(&[42.0]).unwrap(), vec![0.0]);
    assert_eq!(scaler.inverse_transform_sample(&[0.0]).unwrap(), vec![42.0]);
}

/// Checks that dataset helpers return the scalers needed for later inputs and original target units.
#[test]
fn dataset_helpers_return_reusable_feature_and_target_scalers() {
    let mut training = Dataset {
        features: vec![vec![0.0], vec![2.0]],
        targets: vec![vec![10.0], vec![14.0]],
    };
    let features = training.standardize_features().unwrap();
    assert_eq!(training.targets, vec![vec![10.0], vec![14.0]]);
    let targets = training.standardize_targets().unwrap();
    assert_eq!(training.features, vec![vec![-1.0], vec![1.0]]);
    assert_eq!(training.targets, vec![vec![-1.0], vec![1.0]]);
    assert_eq!(features.transform_sample(&[4.0]).unwrap(), vec![3.0]);
    assert_eq!(
        targets.inverse_transform_sample(&[2.0]).unwrap(),
        vec![16.0]
    );
}

/// Checks validation before fitting statistics on malformed matrices.
#[test]
fn fitting_rejects_empty_ragged_and_nonfinite_data() {
    for rows in [vec![], vec![vec![]], vec![vec![1.0], vec![2.0, 3.0]]] {
        assert!(StandardScaler::fit(&rows).is_err());
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(StandardScaler::fit(&[vec![1.0], vec![value]]).is_err());
    }
}

/// Checks that a late invalid row leaves the entire matrix unchanged in either transform direction.
#[test]
fn transform_and_inverse_do_not_partially_modify_invalid_rows() {
    let scaler = StandardScaler::fit(&[vec![0.0], vec![2.0]]).unwrap();
    for invalid in [vec![], vec![2.0, 3.0], vec![f64::NAN], vec![f64::INFINITY]] {
        for inverse in [false, true] {
            let mut rows = vec![vec![5.0], invalid.clone()];
            let before: Vec<Vec<u64>> = rows
                .iter()
                .map(|row| row.iter().map(|x| x.to_bits()).collect())
                .collect();
            let result = if inverse {
                scaler.inverse_transform(&mut rows)
            } else {
                scaler.transform(&mut rows)
            };
            assert!(result.is_err());
            let after: Vec<Vec<u64>> = rows
                .iter()
                .map(|row| row.iter().map(|x| x.to_bits()).collect())
                .collect();
            assert_eq!(after, before);
        }
    }
}

/// Checks that fitting and transforming extreme values returns errors rather than invalid numbers.
#[test]
fn numeric_overflow_is_reported_instead_of_producing_nan() {
    assert!(StandardScaler::fit(&[vec![-f64::MAX], vec![f64::MAX]]).is_err());
    let scaler = StandardScaler::fit(&[vec![-f64::MAX]]).unwrap();
    assert!(scaler.transform_sample(&[f64::MAX]).is_err());
    let scaler = StandardScaler::fit(&[vec![f64::MAX]]).unwrap();
    assert!(scaler.inverse_transform_sample(&[f64::MAX]).is_err());
}

/// Checks that dataset-level preprocessing also preserves data when validation fails.
#[test]
fn invalid_dataset_standardization_does_not_change_earlier_rows() {
    let mut data = Dataset {
        features: vec![vec![1.0], vec![2.0, 3.0]],
        targets: vec![vec![4.0], vec![]],
    };
    let features = data.features.clone();
    let targets = data.targets.clone();
    assert!(data.standardize_features().is_err());
    assert!(data.standardize_targets().is_err());
    assert_eq!(data.features, features);
    assert_eq!(data.targets, targets);
}
