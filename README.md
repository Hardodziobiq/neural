# neural

A fully connected neural network in Rust with regression, classification,
backpropagation, and mini-batch gradient descent.

## Run

```console
cargo run --release
cargo test --locked
```

The program runs two examples using separate training and test CSV files from
`data/demo`: regression of `y = 2x + 1` and classification into three classes.
It prints regression MSE, one prediction in original units, classification
cross-entropy, and test accuracy. These small datasets demonstrate the workflow.

## Configure

Use command-line arguments for your own CSV files and training settings:

```console
cargo run -- regression train.csv test.csv 500 16
cargo run -- classification train.csv test.csv 3 500 16
cargo run -- --help
```

The arguments are the task, training file, test file, class count (classification
only), then optional epoch count and batch size. Defaults are 500 epochs and batch
size 16. Counts must be positive, and classification requires at least two classes.
Input dimensions are read from the training data. Edit `src/main.rs` to change
network architecture, learning rate, or random seed.

| Example | Inputs | Layers | Hidden / output activation | Loss | Learning rate |
| --- | --- | --- | --- | --- | --- |
| Regression | 1 | 16, 16, 1 | Tanh / Linear | MSE | 0.03 |
| Classification | 2 | 8, 3 | Tanh / Softmax | Cross-entropy | 0.05 |

Both examples use biases and random seed 42. CSV files need a header, numeric
values separated by commas, and the target or class label in the last column.
Class labels start at zero. Quoted fields containing commas are not supported.

## Loss plots

Add `--plot FILE.svg` to save training and test loss by epoch after training:

```console
cargo run -- regression data/demo/regression_train.csv data/demo/regression_test.csv 500 16 --plot regression.svg
cargo run -- classification data/demo/classification_train.csv data/demo/classification_test.csv 3 --plot classification.svg
```

Open the SVG in a browser or image viewer. The parent directory must already
exist; an existing output file is overwritten. Without `--plot`, no plot is saved.
Plots use a linear loss axis and the epoch metrics in `History`, without weight
snapshots or additional dependencies. Regression loss is in standardized target
units. The library equivalent is `history.save_loss_plot("loss.svg")?`.

## Library essentials

- Validation, training, prediction, and preprocessing return `Result`; handle
  errors using `?` or `match`.
- Fit standardization on training data only. Reuse the returned scaler for test
  data and new inputs. Standardize targets only for regression.
- `predict(&self, input)` expects prepared inputs and does not change model state.
  Use the target scaler to convert regression predictions back to original units.
- MSE averages squared errors over outputs; its gradient is `2 * (prediction -
  target) / output_count`. Dataset loss is also averaged over samples.
- `train()` records epoch losses without copying weights. Use `train_with_history()`
  with `SnapshotPolicy::EveryEpoch` or `EveryIterations(n)` for parameter snapshots.
  Enabled recording includes the initial parameters at iteration zero; zero epochs
  produce an empty history. The interval must be positive.

Example preprocessing and prediction, with `train`, `test`, and `net` already created:

```rust,ignore
let features = train.standardize_features()?;
features.transform(&mut test.features)?;
let targets = train.standardize_targets()?;
targets.transform(&mut test.targets)?;

net.train(&train, Some(&test), 500, 16)?;
let input = features.transform_sample(&[1.5])?;
let prediction = targets.inverse_transform_sample(&net.predict(&input)?)?;
```

`StandardScaler` supports single samples and whole matrices, including inverse
transforms. Constant columns use scale 1. Invalid transformations leave data
unchanged. Training validates both datasets before updating weights; it does not
roll back earlier updates if a numerical error occurs later.
