# Dataset format

Keep local datasets in `data/` (ignored by Git), in separate `train.csv` and
`test.csv` files with the same feature order.

- CSV with a header, comma separators and decimal dots.
- Input features first; one target or class label in the last column.
- All rows must match the header width. Use finite numbers; no empty cells.
- Class labels: integers from `0` to `CLASSES - 1`.

## Regression

```csv
x,y
-1,-1
0,1
1,3
```

## Classification

```csv
x1,x2,class
-2,-2,0
2,-2,1
0,2,2
```

## MNIST

Use `scripts/prepare_mnist.py` to export CSV files with a header, 784 pixels
scaled to `[0, 1]`, and a final label `0`–`9` (785 columns total).
See [README.md](README.md#cli-help) for commands.
