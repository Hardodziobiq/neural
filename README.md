# neural

A fully connected neural network in Rust with backpropagation and mini-batch SGD.
No neural-network library is used. Code, program output, and reports are in English.

Keep your datasets in `data/`, which is excluded from Git. See
[DATA_FORMAT.md](DATA_FORMAT.md) for the required directory layout and CSV examples.
Only four tiny demonstration files are included in `examples/` so the demo and
automated tests work without downloading external datasets.

## CLI help

Run from the project directory. Replace uppercase placeholders; brackets mark optional arguments.

```text
RUST — prefix commands with: cargo run --release --
  --help | -h
  regression TRAIN.csv TEST.csv [EPOCHS] [BATCH] [OPTIONS]
  classification TRAIN.csv TEST.csv CLASSES [EPOCHS] [BATCH] [OPTIONS]
  experiments PLAN.tsv [--summary FILE.csv]

  No arguments: cargo run --release (both demos with reports).
  EPOCHS: positive integer, default 500. BATCH: positive integer, default 16.
  CLASSES: integer >= 2; CSV labels must be 0..CLASSES-1.

SINGLE-RUN OPTIONS — use separate values, e.g. --seed 42
  --hidden WIDTHS|none      Hidden widths, e.g. 16,8; output added automatically.
                           Defaults: regression 16,16; classification 8.
  --activation NAME        sigmoid|tanh|relu|linear|softmax; default tanh.
  --output NAME            linear|softmax|sigmoid; selected automatically.
  --loss NAME              Regression: mse (default)|mae, output linear.
                           Classification: cce (default)|mse, output softmax.
                           Two-class BCE: bce, one sigmoid output.
  --learning-rate NUMBER   Finite and >= 0; regression 0.03, classification 0.05.
  --seed NUMBER            Unsigned 64-bit integer; default 42.
  --bias true|false        Default true.
  --out-dir DIRECTORY      Save HTML report, CSV histories and SVG plots.
  --snapshot-every N       Positive update interval; requires --out-dir.
                           Default targets ~200 snapshots; initial/final retained.
  --weight-limit N         Positive plotted-curve limit, default 12; CSV keeps all.
  --plot FILE.svg          Separate epoch-loss plot; parent directory must exist.

PLANS
  --summary FILE.csv       Plan results; default results/summary.csv.
  Network settings belong in PLAN.tsv, not after the experiments command.
  Replay: use a report's configuration.tsv as PLAN.tsv (fresh training).

PYTHON — python scripts/prepare_mnist.py [OPTIONS]
  --help | -h              Show exporter help.
  --input-dir DIRECTORY    Required: four local MNIST IDX/IDX.gz source files.
  --output-dir DIRECTORY   Default data/mnist/processed.
  --train-limit N          Positive subset size; default all training samples.
  --test-limit N           Positive subset size; default all test samples.
  --seed NUMBER            Unsigned 64-bit selection seed, default 42.
  --overwrite              Replace existing exports; takes no value.

PATHS
  Quote paths containing spaces. CLI paths are relative to the working directory;
  data/report paths inside a plan are relative to that plan. Reports overwrite
  files with matching names. Use validation data while choosing settings.

EXAMPLES
  cargo run --release -- regression examples/regression_train.csv examples/regression_test.csv 200 4 --hidden 8,8 --out-dir results/regression
  cargo run --release -- classification examples/classification_train.csv examples/classification_test.csv 3 200 4 --activation relu --out-dir results/classification
  cargo run --release -- experiments experiments/demo.tsv --summary results/summary.csv
  python scripts/prepare_mnist.py --input-dir data/raw/mnist --output-dir data/mnist/subset --train-limit 5000
  cargo run --release -- classification data/mnist/subset/train.csv data/mnist/subset/test.csv 10 10 64 --hidden 128 --activation relu --out-dir results/mnist
```

## Experiment plans

```console
cargo run --release -- experiments experiments/demo.tsv --summary results/summary.csv
```

A UTF-8 tab-separated plan specifies independent runs. It is a template
for coursework experiment.

One plan must use exactly one training/test file pair, task and class count.
Course plans are in `experiments/activation/`, `experiments/architecture/` and
`experiments/loss/`, named `DATASET-N.tsv`. Short checks are in `experiments/smoke/`.

```console
cargo run --release -- experiments experiments/activation/simple-100.tsv --summary results/course/simple-100-activation.csv
```

Required columns are `task`, `train`, `test`, `out_dir`; classification also needs
`classes`. Optional columns match CLI flags with underscores: `hidden`, `activation`,
`output`, `loss`, `learning_rate`, `seed`, `bias`, `snapshot_every`, `weight_limit`,
plus `epochs` and `batch`. Blank optional cells use defaults.

Every run gets a fresh model and RNG. A failed run is marked in the summary and
later runs continue; the command returns failure if any run failed. The summary
(default `results/summary.csv`) lists losses, original-unit regression MSE,
classification accuracy (0 to 1), timing, report paths, and errors.

Each plan also saves an HTML batch report next to its summary CSV:
`--summary results/run.csv` creates `results/run.html`. It compares configurations
and metrics, shows the dataset once, and plots configuration means across distinct
seeds on shared epoch-loss and batch-loss charts. Shading and tables show sample
standard deviations; repeated configuration/seed pairs count once. Failed runs are
listed but excluded from averages. These plots are embedded in the HTML. Prediction
plots and weight/bias histories remain in the linked individual reports.

Replay a completed report with its exact resolved configuration:

```console
cargo run --release -- experiments results/configuration.tsv --summary results/replay.csv
```


## Report contents

`--loss mse` follows the lecture: `L = 0.5 * sum((prediction - target)^2)`
per sample, with gradient `prediction - target`. Batch and dataset losses are
averaged over samples, not outputs. Regression loss histories use standardized
targets; `test_mse` is a separate mean squared error metric in original units,
without the factor `0.5`. For scalar regression, standardized MSE is twice the
reported standardized MSE loss. Earlier runs used a different loss scaling;
rerun experiments rather than mixing their loss histories with new results.

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

Only 1D regression and 2D classification get result plots; other dimensions still
get numeric metrics and training-history charts.


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

The output is `train.csv`, `test.csv`, and `metadata.json`. Each CSV has a header,
784 pixels in row-major order normalized by `pixel / 255.0`, then one integer label
from 0 to 9 in the last column. Labels are not one-hot encoded; Rust handles that.
The complete test set is exported unless `--test-limit` is supplied.
Existing exports require `--overwrite`.

The format is directly supported by `load_classification`:

```console
cargo run --release -- classification data/mnist/subset/train.csv data/mnist/subset/test.csv 10 10 64 --hidden 128 --activation relu --seed 42
python -m unittest discover -s scripts -p "test_prepare_mnist.py"
```
