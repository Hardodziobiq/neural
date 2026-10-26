use std::fs;
use std::io;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::net::error::ValidationError;
use crate::net::standard_scaler::StandardScaler;

#[derive(Clone)]
pub struct Dataset {
    pub features: Vec<Vec<f64>>,
    pub targets: Vec<Vec<f64>>,
}

impl Dataset {
    /// Checks every sample against the network dimensions before training or evaluation.
    pub fn validate(
        &self,
        inputs_count: usize,
        outputs_count: usize,
    ) -> Result<(), ValidationError> {
        if inputs_count == 0 || outputs_count == 0 {
            return Err(ValidationError::new(
                "Expected input and output dimensions must be greater than zero",
            ));
        }
        if self.features.len() != self.targets.len() {
            return Err(ValidationError::new(format!(
                "Sample count mismatch: {} feature rows, {} target rows",
                self.features.len(),
                self.targets.len(),
            )));
        }
        if self.features.is_empty() {
            return Err(ValidationError::new("Dataset must not be empty"));
        }

        for (name, rows, expected) in [
            ("Features", &self.features, inputs_count),
            ("Targets", &self.targets, outputs_count),
        ] {
            for (sample, row) in rows.iter().enumerate() {
                if row.len() != expected {
                    return Err(ValidationError::new(format!(
                        "{name} row {sample}: expected {expected} values, got {}",
                        row.len(),
                    )));
                }
                for (index, value) in row.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(ValidationError::new(format!(
                            "{name} row {sample}, column {index}: value must be finite",
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Fits and applies training statistics. Reuse the returned scaler for test/new inputs.
    pub fn standardize_features(&mut self) -> Result<StandardScaler, ValidationError> {
        let scaler = StandardScaler::fit(&self.features)?;
        scaler.transform(&mut self.features)?;
        Ok(scaler)
    }
    /// For regression only. Use inverse_transform_sample to restore predicted target units.
    pub fn standardize_targets(&mut self) -> Result<StandardScaler, ValidationError> {
        let scaler = StandardScaler::fit(&self.targets)?;
        scaler.transform(&mut self.targets)?;
        Ok(scaler)
    }
    /// Converts scalar class labels to one-hot vectors after validating every label.
    /// On validation failure, all targets remain unchanged. Row indices are zero-based.
    pub fn one_hot_encode_targets(&mut self, num_classes: usize) -> Result<(), ValidationError> {
        if num_classes == 0 {
            return Err(ValidationError::new(
                "Class count must be greater than zero",
            ));
        }
        if self.targets.is_empty() {
            return Err(ValidationError::new("Targets must not be empty"));
        }

        for (sample, target) in self.targets.iter().enumerate() {
            if target.len() != 1 {
                return Err(ValidationError::new(format!(
                    "Targets row {sample}: expected exactly one class label before one-hot encoding, got {} values",
                    target.len(),
                )));
            }
            let class_value = target[0];
            if !class_value.is_finite()
                || class_value < 0.0
                || class_value.fract() != 0.0
                || class_value >= num_classes as f64
            {
                return Err(ValidationError::new(format!(
                    "Targets row {sample}: class label must be a finite integer in 0..{num_classes}, got {class_value}",
                )));
            }
        }

        for target in &mut self.targets {
            *target = Self::one_hot(target[0] as usize, num_classes);
        }
        Ok(())
    }
    /// Returns a shuffled copy using one shared permutation for feature rows and their matching targets.
    pub fn shuffled(&self, rng: &mut StdRng) -> Self {
        let mut dataset: Dataset = self.clone();
        let mut indices: Vec<usize> = (0..dataset.features.len()).collect();
        indices.shuffle(rng);
        dataset.features = indices.iter().map(|&i| self.features[i].clone()).collect();
        dataset.targets = indices.iter().map(|&i| self.targets[i].clone()).collect();
        dataset
    }
}

impl Dataset {
    /// Creates a class indicator vector; the class index must already have been validated.
    fn one_hot(class: usize, num_classes: usize) -> Vec<f64> {
        let mut result: Vec<f64> = vec![0.0; num_classes];

        result[class] = 1.0;

        result
    }
}

/// Loads headered numeric CSV; all columns except the last are inputs and the last is one regression target.
pub fn load_regression(path: &str) -> io::Result<Dataset> {
    let content: String = fs::read_to_string(path)?;
    let mut lines: std::str::Lines<'_> = content.lines();

    let header: &str = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Empty file!"))?;

    let columns: usize = header.split(',').count();

    if columns < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Regression requires at least 2 columns!",
        ));
    }

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<Vec<f64>> = Vec::new();

    for line in lines.filter(|line| !line.trim().is_empty()) {
        let values: Vec<f64> = line
            .split(',')
            .map(|s| {
                s.trim().parse::<f64>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Incorrect number: {s}"))
                })
            })
            .collect::<io::Result<_>>()?;

        if values.len() != columns {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Count of columns don't match!",
            ));
        }

        features.push(values[..columns - 1].to_vec());
        targets.push(vec![values[columns - 1]]);
    }

    Ok(Dataset { features, targets })
}

/// Loads headered numeric CSV into feature rows and nonnegative integer class labels from the last column.
pub fn load_classification(path: &str) -> io::Result<(Vec<Vec<f64>>, Vec<usize>)> {
    let content: String = fs::read_to_string(path)?;
    let mut lines: std::str::Lines<'_> = content.lines();

    let header: &str = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Empty file!"))?;

    let columns: usize = header.split(',').count();

    if columns < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Classification requires features and labels!",
        ));
    }

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    for line in lines.filter(|line| !line.trim().is_empty()) {
        let values: Vec<f64> = line
            .split(',')
            .map(|s| {
                s.trim().parse::<f64>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid number: {s}"))
                })
            })
            .collect::<io::Result<_>>()?;

        if values.len() != columns {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Count of columns don't match!",
            ));
        }

        let class: f64 = values[columns - 1];

        if !class.is_finite() || class < 0.0 || class.fract() != 0.0 || class > usize::MAX as f64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Incorrect label number!",
            ));
        }

        features.push(values[..columns - 1].to_vec());
        labels.push(class as usize);
    }

    Ok((features, labels))
}
