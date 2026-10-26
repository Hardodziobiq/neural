"""Tests use tiny synthetic IDX files, not downloaded MNIST data."""

import csv
import gzip
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import prepare_mnist as mnist


def write_split(root, prefix, samples, compressed=False):
    """Creates a small synthetic image/label IDX pair, optionally compressed, for exporter tests."""
    images_name = f"{prefix}-images-idx3-ubyte"
    labels_name = f"{prefix}-labels-idx1-ubyte"
    images = struct.pack(">IIII", 2051, len(samples), 28, 28) + b"".join(pixels for pixels, _ in samples)
    labels = struct.pack(">II", 2049, len(samples)) + bytes(label for _, label in samples)
    for name, content in ((images_name, images), (labels_name, labels)):
        path = root / (name + (".gz" if compressed else ""))
        if compressed:
            with gzip.open(path, "wb") as output:
                output.write(content)
        else:
            path.write_bytes(content)


class MnistExportTests(unittest.TestCase):
    def setUp(self):
        """Allocates isolated input and output paths and registers cleanup for each test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "raw"
        self.source.mkdir()
        self.output = self.root / "processed"

    def inputs(self, compressed=False):
        """Writes tiny training and test fixtures containing known pixel intensities and digit labels."""
        pixels = bytes([0, 255, 128, 64] + [0] * 780)
        write_split(self.source, "train", [(pixels, 7), (bytes([255]) * 784, 2)], compressed)
        write_split(self.source, "t10k", [(bytes([64]) * 784, 9)], compressed)

    def test_raw_export_matches_rust_column_order_and_normalization(self):
        """Checks row-major pixels, unit-range scaling, scalar labels, headers, and exported sample metadata."""
        self.inputs()
        metadata = mnist.prepare(self.source, self.output)
        with (self.output / "train.csv").open(newline="", encoding="ascii") as source:
            rows = list(csv.reader(source))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(rows[0]), 785)
        self.assertEqual(rows[0][0], "pixel_0")
        self.assertEqual(rows[0][-1], "label")
        self.assertEqual(rows[1][-1], "7")
        self.assertEqual(rows[2][-1], "2")
        self.assertEqual(float(rows[1][0]), 0.0)
        self.assertEqual(float(rows[1][1]), 1.0)
        self.assertAlmostEqual(float(rows[1][2]), 128 / 255, places=8)
        self.assertAlmostEqual(float(rows[1][3]), 64 / 255, places=8)
        self.assertTrue(all(0 <= float(value) <= 1 for row in rows[1:] for value in row[:-1]))
        self.assertEqual(metadata["train"]["exported_samples"], 2)
        self.assertEqual(metadata["test"]["exported_samples"], 1)
        self.assertEqual(metadata["train"]["label_counts"]["7"], 1)
        self.assertEqual(json.loads((self.output / "metadata.json").read_text())["class_count"], 10)

    def test_gzip_export_matches_plain_idx_export(self):
        """Checks that compressed and uncompressed inputs produce identical CSV values."""
        self.inputs()
        mnist.prepare(self.source, self.output)
        other = self.root / "compressed"
        self.inputs(compressed=True)
        for name in mnist.FILE_NAMES.values():
            (self.source / name).unlink()
        mnist.prepare(self.source, other)
        for name in ("train.csv", "test.csv"):
            self.assertEqual((self.output / name).read_bytes(), (other / name).read_bytes())

    def test_common_dot_idx_file_names_are_supported(self):
        """Checks discovery of the alternative dot-based IDX filenames."""
        self.inputs()
        for name in mnist.FILE_NAMES.values():
            (self.source / name).rename(self.source / name.replace("-idx", ".idx"))
        self.assertEqual(mnist.prepare(self.source, self.output)["train"]["exported_samples"], 2)

    def test_subsets_are_reproducible_and_keep_scalar_digit_labels(self):
        """Checks seeded subset sizes and identical exports when the selection seed is repeated."""
        samples = [(bytes([index]) * 784, index % 10) for index in range(30)]
        write_split(self.source, "train", samples)
        write_split(self.source, "t10k", samples)
        first = mnist.prepare(self.source, self.output, train_limit=7, test_limit=4, seed=0)
        second_dir = self.root / "same"
        mnist.prepare(self.source, second_dir, train_limit=7, test_limit=4, seed=0)
        different_dir = self.root / "different"
        mnist.prepare(self.source, different_dir, train_limit=7, test_limit=4, seed=1)
        self.assertEqual(first["train"]["exported_samples"], 7)
        self.assertEqual(first["test"]["exported_samples"], 4)
        self.assertEqual((self.output / "train.csv").read_bytes(), (second_dir / "train.csv").read_bytes())
        self.assertNotEqual((self.output / "train.csv").read_bytes(), (different_dir / "train.csv").read_bytes())

    def test_test_data_does_not_change_training_export(self):
        """Checks that changes in the official test split cannot affect exported training values."""
        self.inputs()
        mnist.prepare(self.source, self.output)
        write_split(self.source, "t10k", [(bytes([255]) * 784, 0)])
        other = self.root / "other"
        mnist.prepare(self.source, other)
        self.assertEqual((self.output / "train.csv").read_bytes(), (other / "train.csv").read_bytes())

    def test_invalid_test_payload_does_not_replace_existing_exports(self):
        """Checks that a truncated test file leaves all final exports intact and removes temporary output."""
        self.inputs()
        mnist.prepare(self.source, self.output)
        before = {name: (self.output / name).read_bytes() for name in mnist.OUTPUT_NAMES}
        path = self.source / mnist.FILE_NAMES["test_images"]
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            mnist.prepare(self.source, self.output, overwrite=True)
        self.assertEqual(before, {name: (self.output / name).read_bytes() for name in mnist.OUTPUT_NAMES})
        self.assertFalse(list(self.output.glob(".mnist-export-*")))

    def test_invalid_mnist_headers_labels_and_extra_payload_are_rejected(self):
        """Checks rejection of invalid magic values, image shapes, digit labels, counts, and trailing bytes."""
        self.inputs()
        path = self.source / mnist.FILE_NAMES["train_images"]
        original = path.read_bytes()
        for header in [struct.pack(">IIII", 0, 2, 28, 28), struct.pack(">IIII", 2051, 2, 32, 32)]:
            path.write_bytes(header + original[16:])
            with self.assertRaises(ValueError):
                mnist.prepare(self.source, self.output)
        path.write_bytes(original + b"x")
        with self.assertRaisesRegex(ValueError, "extra bytes"):
            mnist.prepare(self.source, self.output)
        path.write_bytes(original)
        labels = self.source / mnist.FILE_NAMES["train_labels"]
        labels.write_bytes(struct.pack(">II", 2049, 2) + bytes([10, 2]))
        with self.assertRaisesRegex(ValueError, "Invalid MNIST digit"):
            mnist.prepare(self.source, self.output)
        labels.write_bytes(struct.pack(">II", 2049, 3) + bytes([1, 2, 3]))
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            mnist.prepare(self.source, self.output)

    def test_invalid_limits_and_existing_files_require_explicit_overwrite(self):
        """Checks limit and seed ranges plus explicit permission to replace existing exports."""
        self.inputs()
        for limit in (0, -1, 3):
            with self.assertRaises(ValueError):
                mnist.prepare(self.source, self.output, train_limit=limit)
        for seed in (-1, 2**64):
            with self.assertRaises(ValueError):
                mnist.prepare(self.source, self.output, seed=seed)
        mnist.prepare(self.source, self.output)
        with self.assertRaises(FileExistsError):
            mnist.prepare(self.source, self.output)
        mnist.prepare(self.source, self.output, overwrite=True)

    def test_command_line_exports_files_and_reports_missing_inputs_without_traceback(self):
        """Checks the exporter CLI success path and readable missing-file errors."""
        self.inputs()
        script = Path(mnist.__file__)
        result = subprocess.run([sys.executable, str(script), "--input-dir", str(self.source),
                                 "--output-dir", str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Train: 2 samples", result.stdout)
        result = subprocess.run([sys.executable, str(script), "--input-dir", str(self.root / "missing")],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_rust_binary_accepts_exported_csv_when_already_built(self):
        """Checks exported CSV with an existing Rust executable; skips when no binary has been built."""
        executable = "neural.exe" if sys.platform == "win32" else "neural"
        binary = Path(__file__).resolve().parents[1] / "target" / "debug" / executable
        if not binary.is_file():
            self.skipTest("Build the Rust executable with cargo build to check CSV interoperability")
        self.inputs()
        mnist.prepare(self.source, self.output)
        result = subprocess.run([
            str(binary), "classification", str(self.output / "train.csv"), str(self.output / "test.csv"),
            "10", "1", "2", "--hidden", "none", "--learning-rate", "0",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Classification", result.stdout)
        self.assertNotIn("NaN", result.stdout)


if __name__ == "__main__":
    unittest.main()
