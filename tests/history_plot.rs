mod common;

use neural::net::history::History;

/// Checks that an epoch chart includes both available loss series and their expected coordinates.
#[test]
fn plots_training_and_optional_test_series() {
    let mut history = History::new();
    history.record_epoch(1, 2.0, Some(2.5));
    history.record_epoch(2, 1.0, Some(1.5));
    let file = common::TempPlot::new();
    history.save_loss_plot(&file.path).unwrap();
    let svg = std::fs::read_to_string(&file.path).unwrap();
    assert!(svg.starts_with("<svg xmlns="));
    assert!(svg.ends_with("</svg>\n"));
    assert!(svg.contains("id=\"training-loss\""));
    assert!(svg.contains("id=\"test-loss\""));
    assert!(svg.contains(">Epoch</text>"));
    assert!(svg.contains(">Loss</text>"));
    assert!(svg.contains("points=\"100.00,150.00 790.00,310.00\""));
    assert!(svg.contains("points=\"100.00,70.00 790.00,230.00\""));
}

/// Checks visible single-point charts and finite SVG coordinates at extreme valid loss scales.
#[test]
fn single_epoch_zero_loss_and_extreme_finite_loss_have_finite_coordinates() {
    for loss in [0.0, 1e-300, f64::MAX] {
        let mut history = History::new();
        history.record_epoch(1, loss, None);
        let file = common::TempPlot::new();
        history.save_loss_plot(&file.path).unwrap();
        let svg = std::fs::read_to_string(&file.path).unwrap();
        assert!(svg.contains("<circle"));
        assert!(!svg.contains("NaN"));
        assert!(!svg.contains("inf"));
        assert!(!svg.contains(">Test</text>"));
    }
}

/// Checks that absent evaluation metrics break a curve instead of drawing an invented connection.
#[test]
fn missing_test_metrics_break_the_test_line() {
    let mut history = History::new();
    history.record_epoch(1, 3.0, Some(4.0));
    history.record_epoch(2, 2.0, None);
    history.record_epoch(3, 1.0, Some(2.0));
    let file = common::TempPlot::new();
    history.save_loss_plot(&file.path).unwrap();
    let svg = std::fs::read_to_string(&file.path).unwrap();
    let test_series = svg
        .split("<g id=\"test-loss\"")
        .nth(1)
        .unwrap()
        .split("</g>")
        .next()
        .unwrap();
    assert_eq!(test_series.matches("<circle").count(), 2);
    assert!(!test_series.contains("<path"));
    assert_eq!(test_series.matches("<polyline").count(), 2);
    for coordinates in test_series.split("points=\"").skip(1) {
        let points = coordinates.split('"').next().unwrap();
        assert_eq!(points.split_whitespace().count(), 1);
    }
}

/// Checks that invalid histories fail before an existing output file can be replaced.
#[test]
fn invalid_histories_do_not_overwrite_existing_files() {
    let mut nonfinite = History::new();
    nonfinite.record_epoch(1, f64::NAN, None);
    let mut nonfinite_test = History::new();
    nonfinite_test.record_epoch(1, 1.0, Some(f64::INFINITY));
    let mut unordered = History::new();
    unordered.record_epoch(2, 1.0, None);
    unordered.record_epoch(1, 1.0, None);
    let mut duplicate = History::new();
    duplicate.record_epoch(1, 1.0, None);
    duplicate.record_epoch(1, 0.5, None);
    for history in [
        History::new(),
        nonfinite,
        nonfinite_test,
        unordered,
        duplicate,
    ] {
        let file = common::TempPlot::new();
        std::fs::write(&file.path, "original").unwrap();
        let error = history.save_loss_plot(&file.path).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(std::fs::read_to_string(&file.path).unwrap(), "original");
    }
}
