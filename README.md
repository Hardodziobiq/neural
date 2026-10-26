# neural

A fully connected neural network in Rust with backpropagation and mini-batch SGD.
No neural-network library is used. Code, program output, and reports are in English.

## Run an experiment

```console
cargo run --release
cargo run --release -- regression train.csv test.csv 500 16 --out-dir results/regression
cargo run --release -- classification train.csv test.csv 3 500 16 --out-dir results/classification
cargo run -- --help
```

No arguments run both bundled examples and save reports in `results/demo-regression`
and `results/demo-classification`. Open `index.html` in the output directory to see
all charts. Explicit commands create a report only when `--out-dir` is supplied.
The optional positional counts are epochs (default 500) and batch size (default 16).
Classification also requires a class count of at least two.

CSV input needs a header, comma-separated numbers, and a single target or integer
class label in the last column. Classes start at zero. Quoted fields with embedded
commas are not supported. Input dimensions are inferred and both datasets validated.

## Granular configuration

Each run can change one factor while holding the others fixed:

| Flag | Values / default |
| --- | --- |
| `--hidden` | Hidden widths, e.g. `8,8`; `none` means zero hidden layers. Defaults: `16,16` for regression, `8` for classification. The output layer is added automatically. |
| `--activation` | `sigmoid`, `tanh` (default), `relu`, `linear`, `softmax` |
| `--output` | Regression: `linear`; multiclass: `softmax`; binary BCE: `sigmoid` |
| `--loss` | Regression: `mse` (default) or `mae`; multiclass: `cce` (default) or `mse`; binary: `bce` |
| `--learning-rate` | Finite nonnegative number; default `0.03` for regression, `0.05` for classification |
| `--seed` | Unsigned 64-bit seed, including zero; default `42` |
| `--bias` | `true` (default) or `false` |
| `--out-dir` | Output directory for the complete experiment report |
| `--snapshot-every` | Record weights every N updates; requires a report. Default interval targets about 200 snapshots. Initial and final states are retained. |
| `--weight-limit` | Maximum weight/bias curves shown, default 12, selected evenly across parameters. CSV contains all recorded parameters. |
| `--plot` | Optional separate epoch-loss SVG, e.g. `--plot loss.svg` |

For example, change the hidden activation and architecture without editing code:

```console
cargo run --release -- regression data/demo/regression_train.csv data/demo/regression_test.csv 200 4 --hidden 8,8 --activation relu --loss mse --seed 42 --bias false --out-dir results/relu
```

For two-class BCE, `--loss bce` selects one sigmoid output and scalar 0/1 targets.
MSE classification retains one-hot targets and softmax outputs. Incompatible
output/loss combinations are rejected. Use at least one output layer even with no
hidden layers; `--hidden none` handles this automatically.

## Experiment plans

```console
cargo run --release -- experiments experiments/demo.tsv --summary results/summary.csv
```

A UTF-8 tab-separated plan specifies independent runs. The included 16-row plan
compares sigmoid/Tanh/ReLU, zero through four hidden layers, MSE/MAE for regression,
and CCE/MSE for classification on the small demonstration data. It is a template
for coursework experiments, not a substitute for running the provided course data.

Required columns are `task`, `train`, `test`, `out_dir`; classification also needs
`classes`. Optional columns match CLI flags with underscores: `hidden`, `activation`,
`output`, `loss`, `learning_rate`, `seed`, `bias`, `snapshot_every`, `weight_limit`,
plus `epochs` and `batch`. Blank optional cells use defaults. Unknown or duplicate
columns and repeated output directories are rejected before running. Paths in a
plan are relative to that plan, so spaces need no quoting. Tabs/newlines inside
values are not supported. Lines starting with `#` are comments.

Every run gets a fresh model and RNG. A failed run is marked in the summary and
later runs continue; the command returns failure if any run failed. The summary
(default `results/summary.csv`) lists losses, original-unit regression MSE,
classification accuracy (0 to 1), timing, report paths, and errors.

Replay a completed report with its exact resolved configuration:

```console
cargo run --release -- experiments results/relu/configuration.tsv --summary results/replay.csv
```

The saved plan contains absolute data/output paths, all effective model settings,
and the seed. Keep the data and Cargo.lock unchanged for repeatability. Timing is
not deterministic. Replaying replaces files in the saved report directory.

## Report contents

| File | Meaning |
| --- | --- |
| `index.html` | Overview, configuration, metrics, links and all visualizations |
| `training-data.svg` | Training/test observations in original coordinates |
| `results.svg` | Regression curve on a dense grid, or 2D decision regions with predictions; classification mistakes have a black border |
| `epoch-loss.svg` | Whole training-set loss after each epoch |
| `iteration-loss.svg` | Mean loss of each batch **before** its parameter update |
| `weights.svg` | Selected weight and bias values by update, including initialization |
| `configuration.tsv` | Replayable experiment configuration |
| `metrics.csv` | Final training/test losses, test MSE/accuracy, update count and time |
| `epochs.csv` | Epoch history; validation column is empty for CLI runs |
| `iterations.csv` | Update number, epoch, actual batch size and pre-update batch loss |
| `weights.csv` | All recorded weights and biases, with layer/neuron/input indices |

Epoch and batch loss are different measurements: a noisy batch curve does not
necessarily mean the whole-dataset loss increased. Regression training losses use
standardized targets; the result curve and final test MSE use original units.
Only 1D regression and 2D classification get result plots; other dimensions still
get numeric metrics and training-history charts. Generated files replace files
with the same names in the chosen directory and `results/` is Git-ignored.

The official test set is evaluated only after training. Neither its labels nor
its feature statistics influence the trained parameters. For hyperparameter
selection use a validation split from the training data as the evaluation file;
keep the final test set for the final assessment. The CLI does not implement
checkpoint selection or early stopping based on evaluation data.

## Library and checks

### MNIST export (Python)

`scripts/prepare_mnist.py` prepares **MNIST only** from local IDX files; it does not
download data or train a model. Python 3.10+ is sufficient; no packages are needed.
Place these four files in `data/raw/mnist` (plain or `.gz`):

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

```console
python scripts/prepare_mnist.py --input-dir data/raw/mnist --output-dir data/mnist/processed
python scripts/prepare_mnist.py --input-dir data/raw/mnist --output-dir data/mnist/subset --train-limit 5000 --seed 42
```

The output is `train.csv`, `test.csv`, and `metadata.json`. Each CSV has a header,
784 pixels in row-major order normalized by `pixel / 255.0`, then one integer label
from 0 to 9 in the last column. Labels are not one-hot encoded; Rust handles that.
The official train/test split stays separate. Optional subsets are reproducible
for the selected Python version and seed, and do not change normalization.
The complete test set is exported unless `--test-limit` is supplied.
Existing exports require `--overwrite`. Invalid source data does not replace final
exports. Files are streamed, so all images are not loaded into memory at once.

The format is directly supported by `load_classification`:

```console
cargo run --release -- classification data/mnist/subset/train.csv data/mnist/subset/test.csv 10 10 64 --hidden 128 --activation relu --seed 42
python -m unittest discover -s scripts -p "test_prepare_mnist.py"
```

The current Rust CLI **additionally standardizes features** using training-only
statistics after reading the normalized CSV. Values in the exported file are in
`[0, 1]`; values inside that CLI's network are therefore standardized, not confined
to `[0, 1]`. For unit-range inputs through the library, pass the exported features
directly to the network without calling `standardize_features`.
CSV is compatible with the existing loader but full MNIST exports are large; use
a subset for initial experiments. These commands are examples, not measured
MNIST accuracy results.

### Rust library

`new`, `train`, `test`, `predict`, preprocessing, and label encoding return `Result`.
Fit scalers only on training data; reuse them on evaluation/new inputs and invert
the target transform for regression predictions. Constant columns use scale 1.
`predict(&self, input)` does not change model state. MSE is the mean squared error
over outputs and its gradient is `2 * (prediction - target) / output_count`.

`History` contains epoch metrics, per-update batch metrics, and optional parameter
snapshots. `train()` stores no weight copies; `train_with_history()` selects
`SnapshotPolicy::EveryEpoch` or `EveryIterations(n)`. The CLI adds a final snapshot
when necessary for a complete report. No rollback of earlier updates is performed
if a numerical failure occurs later during training.

```console
cargo test --locked
cargo test --test iteration_history
cargo test --test experiments
```
