# Dataset format

The `data/` directory is local and excluded from Git. Download or prepare your
datasets there; they are not included in the repository. The four tiny CSV files
in `examples/` are only demonstration and test fixtures.

## Suggested local layout

```text
data/
  regression/
    train.csv
    test.csv
  classification/
    train.csv
    test.csv
  raw/
    mnist/
      train-images-idx3-ubyte
      train-labels-idx1-ubyte
      t10k-images-idx3-ubyte
      t10k-labels-idx1-ubyte
  mnist/
    processed/
      train.csv
      test.csv
      metadata.json
```

Paths and CSV filenames are arbitrary: pass the chosen files as command-line
arguments. The directory layout above is a convention, not a loader requirement.

## Rules shared by regression and classification

- CSV must have a header. The first line is always treated as a header.
- Separate columns with commas and use a dot for decimal values, e.g. `0.25`.
- Every column before the last is an input feature. The last column is the target.
- Every data row must have exactly the same number of columns as the header.
- Training and test files must have identical feature dimensions and feature order.
- Values must be finite numbers; no missing cells, text labels, NaN or infinity.
- Both datasets must contain at least one sample. Blank lines are ignored.
- Header names are descriptive only; columns are interpreted by position.
- The simple parser does not support quoted fields containing commas.

Keep training and evaluation samples separate. Preprocessing statistics are fitted
only on training data. The command-line runner evaluates the test set after training.

## Regression

For one input and one numeric target, `data/regression/train.csv` can contain:

```csv
x,y
-2,-3
-1,-1
0,1
1,3
2,5
```

A separate `data/regression/test.csv`:

```csv
x,y
-1.5,-2
0.5,2
1.5,4
```

The samples follow `y = 2*x + 1`. `x` is the input and `y` is the expected output.
For multiple inputs use, for example, `x1,x2,y`; the last column remains the single
regression target. Result plots are available only for one-dimensional inputs.

```console
cargo run --release -- regression data/regression/train.csv data/regression/test.csv 200 4 --out-dir results/regression
```

## Classification

For two inputs and three classes, `data/classification/train.csv` can contain:

```csv
x1,x2,class
-2,-2,0
-1.8,-2.2,0
2,-2,1
2.2,-1.8,1
0,2,2
0.2,2.2,2
```

A separate `data/classification/test.csv`:

```csv
x1,x2,class
-2.1,-1.9,0
1.9,-2.1,1
-0.1,2.1,2
```

Class labels must be integers from `0` to `CLASSES - 1`. For three classes use
`0, 1, 2`, not `1, 2, 3`. If an external dataset uses different labels, map them
consistently in both files before training. Store one scalar label per sample;
the Rust program performs one-hot encoding when required.

```console
cargo run --release -- classification data/classification/train.csv data/classification/test.csv 3 200 4 --out-dir results/classification
```

The number after the test path is the number of classes. Input dimensions are
inferred from the CSV columns. Decision-region plots require exactly two inputs.

## MNIST

Place the four original MNIST files in `data/raw/mnist`. Plain IDX files and
their `.gz` versions are supported. Then export them with Python 3.10+:

```console
python scripts/prepare_mnist.py --input-dir data/raw/mnist --output-dir data/mnist/processed
```

The script writes a header, 784 row-major pixel columns in `[0, 1]` (`pixel / 255`),
and an integer label `0` through `9` in the last column. Each CSV row has **785
columns**. Let the script generate these wide files rather than writing them by hand.
No one-hot label columns are exported. The Rust CLI additionally standardizes
features using the training statistics.

```console
cargo run --release -- classification data/mnist/processed/train.csv data/mnist/processed/test.csv 10 1 64 --hidden 64 --activation relu --seed 42
```

MNIST has 784 inputs, so it produces training-history charts and numeric metrics
when requested, but no two-dimensional decision-region plot.
