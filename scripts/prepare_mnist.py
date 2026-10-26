#!/usr/bin/env python3
"""Export MNIST IDX/IDX.gz files to the CSV format used by this Rust project.

Python 3.10+, standard library only. No download or model training is performed.
Each row contains 784 row-major pixels in [0, 1], followed by a digit label.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import platform
import random
import struct
import sys
import tempfile
from typing import BinaryIO, Iterator
import zlib


PIXELS = tuple(format(value / 255.0, ".9g") for value in range(256))
OUTPUT_NAMES = ("train.csv", "test.csv", "metadata.json")
FILE_NAMES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}


def open_idx(path: Path) -> BinaryIO:
    """Opens a plain or gzip-compressed IDX file as a binary stream."""
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def read_exact(stream: BinaryIO, count: int, path: Path) -> bytes:
    """Reads the requested byte count or raises an error identifying a truncated input file."""
    data = stream.read(count)
    if len(data) != count:
        raise ValueError(f"Truncated IDX file: {path}")
    return data


def find_file(directory: Path, name: str) -> Path:
    """Finds an existing MNIST filename variant and returns its absolute path."""
    # Both common MNIST naming conventions are accepted, compressed or plain.
    for stem in (name, name.replace("-idx", ".idx")):
        for suffix in ("", ".gz"):
            path = directory / (stem + suffix)
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(f"Missing {name} or {name}.gz in {directory}")


class MnistSplit:
    def __init__(self, images: Path, labels: Path):
        """Validates image and label headers, their counts, and the required 28-by-28 shape."""
        self.images = images
        self.labels = labels
        with open_idx(images) as stream:
            magic, count, rows, columns = struct.unpack(">IIII", read_exact(stream, 16, images))
        if magic != 2051 or (rows, columns) != (28, 28) or count == 0:
            raise ValueError(f"Expected nonempty 28x28 MNIST images (magic 2051): {images}")
        with open_idx(labels) as stream:
            magic, label_count = struct.unpack(">II", read_exact(stream, 8, labels))
        if magic != 2049 or label_count != count:
            raise ValueError(f"Invalid MNIST label header or image/label count mismatch: {labels}")
        self.count = count

    def samples(self, selected: set[int] | None) -> Iterator[tuple[bytes, int]]:
        """Stream and validate all IDX records, yielding only selected samples."""
        with open_idx(self.images) as images, open_idx(self.labels) as labels:
            read_exact(images, 16, self.images)
            read_exact(labels, 8, self.labels)
            for index in range(self.count):
                pixels = read_exact(images, 784, self.images)
                label = read_exact(labels, 1, self.labels)[0]
                if label > 9:
                    raise ValueError(f"Invalid MNIST digit {label} at sample {index}: {self.labels}")
                if selected is None or index in selected:
                    yield pixels, label
            if images.read(1) or labels.read(1):
                raise ValueError("IDX payload has extra bytes beyond its declared sample count")


def select_indices(count: int, limit: int | None, seed: int) -> set[int] | None:
    """Returns seeded sample indices without replacement, or None to export the complete split."""
    if limit is None or limit == count:
        return None
    if limit > count:
        raise ValueError(f"Requested {limit} samples, but the split contains only {count}")
    return set(random.Random(seed).sample(range(count), limit))


def export_split(source: MnistSplit, path: Path, limit: int | None, seed: int) -> dict:
    """Writes normalized pixels and scalar labels to CSV and returns source, count, and selection metadata."""
    selected = select_indices(source.count, limit, seed)
    counts: Counter[int] = Counter()
    with path.open("w", encoding="ascii", newline="") as output:
        output.write(",".join(f"pixel_{index}" for index in range(784)) + ",label\n")
        for pixels, label in source.samples(selected):
            output.write(",".join(PIXELS[value] for value in pixels) + f",{label}\n")
            counts[label] += 1
    return {
        "source_images": str(source.images),
        "source_labels": str(source.labels),
        "source_samples": source.count,
        "exported_samples": sum(counts.values()),
        "label_counts": {str(label): counts[label] for label in range(10)},
        "selection_seed": seed,
    }


def prepare(input_dir: Path, output_dir: Path, train_limit: int | None = None,
            test_limit: int | None = None, seed: int = 42, overwrite: bool = False) -> dict:
    """Exports the official training and test splits separately and returns their metadata.
    Invalid source data leaves final exports unchanged; replacing existing files requires overwrite.
    """
    if not 0 <= seed < 2**64:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    if any(limit is not None and limit <= 0 for limit in (train_limit, test_limit)):
        raise ValueError("Sample limits must be positive")
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    files = {key: find_file(input_dir, name) for key, name in FILE_NAMES.items()}
    train = MnistSplit(files["train_images"], files["train_labels"])
    test = MnistSplit(files["test_images"], files["test_labels"])
    # Reject invalid limits before starting a potentially large export.
    select_indices(train.count, train_limit, seed)
    select_indices(test.count, test_limit, seed + 1)
    for name in OUTPUT_NAMES:
        destination = output_dir / name
        if destination.resolve() in files.values():
            raise ValueError(f"Output would overwrite an input file: {destination}")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {destination}. Use --overwrite to replace it.")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep final exports untouched if either source contains invalid/truncated data.
    with tempfile.TemporaryDirectory(prefix=".mnist-export-", dir=output_dir) as temporary:
        staging = Path(temporary)
        train_info = export_split(train, staging / "train.csv", train_limit, seed)
        test_info = export_split(test, staging / "test.csv", test_limit, seed + 1)
        metadata = {
            "dataset": "MNIST",
            "image_shape": [28, 28],
            "feature_count": 784,
            "class_count": 10,
            "classes": list(range(10)),
            "normalization": "pixel / 255.0",
            "pixel_range": [0, 1],
            "flatten_order": "row-major, top-left to bottom-right",
            "csv_format": "header; 784 numeric pixel columns; integer label in last column",
            "decimal_precision": "9 significant digits",
            "selection": "all samples by default; limits use seeded uniform sampling without replacement, in source order",
            "seed": seed,
            "python_version": platform.python_version(),
            "train": train_info,
            "test": test_info,
            "rust_cli_preprocessing": "The current Rust CLI additionally standardizes features using training statistics.",
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for name in OUTPUT_NAMES:
            (staging / name).replace(output_dir / name)
    return metadata


def positive(value: str) -> int:
    """Parses a positive sample limit for argparse, rejecting zero and negative counts."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main() -> int:
    """Parses export arguments, reports expected input errors, and returns zero after a successful export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing the four local MNIST IDX or IDX.gz files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/mnist/processed"))
    parser.add_argument("--train-limit", type=positive, help="Optional seeded subset for initial experiments")
    parser.add_argument("--test-limit", type=positive, help="Optional test subset; omit for the complete test split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        metadata = prepare(args.input_dir, args.output_dir, args.train_limit,
                           args.test_limit, args.seed, args.overwrite)
    except (OSError, ValueError, EOFError, zlib.error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Train: {metadata['train']['exported_samples']} samples")
    print(f"Test: {metadata['test']['exported_samples']} samples")
    print(f"Saved train.csv, test.csv and metadata.json to {args.output_dir.resolve()}")
    print("Format: 784 pixels in [0, 1], then label 0-9. No one-hot encoding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
