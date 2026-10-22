use std::fs;
use std::io;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;

#[derive(Clone)]
pub struct Dataset {
    pub features: Vec<Vec<f64>>,
    pub targets: Vec<Vec<f64>>,
}

impl Dataset {
    pub fn standardize_features(&mut self) {
        if self.features.is_empty() {
            return;
        }

        let num_features = self.features[0].len();

        for col in 0..num_features {
            let mut values: Vec<f64> = self.features
                .iter()
                .map(|row| row[col])
                .collect();

            Self::standardize(&mut values);

            for (row, value) in self.features.iter_mut().zip(values) {
                row[col] = value;
            }
        }
    }
    pub fn standardize_targets(&mut self) {
        if self.targets.is_empty() {
            return;
        }

        let num_targets = self.targets[0].len();

        for col in 0..num_targets {
            let mut values: Vec<f64> = self.targets
                .iter()
                .map(|row| row[col])
                .collect();

            Self::standardize(&mut values);

            for (row, value) in self.targets.iter_mut().zip(values) {
                row[col] = value;
            }
        }
    }
    pub fn one_hot_encode_targets(&mut self, num_classes: usize) {
        for target in &mut self.targets {
            if let Some(&class_value) = target.first() {
                if !class_value.is_finite()
                    || class_value < 0.0
                    || class_value.fract() != 0.0
                    || class_value >= num_classes as f64
                {
                    continue;
                }

                *target = Self::one_hot(class_value as usize, num_classes);
            }
        }
    }
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
    fn standardize(values: &mut [f64]) -> (f64, f64) {
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;

        let variance: f64 = values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;

        let std: f64 = variance.sqrt();

        for x in values.iter_mut() {
            *x = if std > 0.0 {
                (*x - mean) / std
            } else {
                0.0
            };
        }

        (mean, std)
    }
    fn one_hot(class: usize, num_classes: usize) -> Vec<f64> {
        let mut result: Vec<f64> = vec![0.0; num_classes];

        if class < num_classes {
            result[class] = 1.0;
        }

        result
    }
}

pub fn load_regression(path: &str) -> io::Result<Dataset> {
    let content: String = fs::read_to_string(path)?;
    let mut lines: std::str::Lines<'_> = content.lines();

    let header: &str = lines.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "Empty file!")
    })?;

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
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Incorrect number: {s}"),
                    )
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

pub fn load_classification(
    path: &str,
) -> io::Result<(Vec<Vec<f64>>, Vec<usize>)> {
    let content: String = fs::read_to_string(path)?;
    let mut lines: std::str::Lines<'_> = content.lines();

    let header: &str = lines.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "Empty file!")
    })?;

    let columns: usize = header.split(',').count();

    if columns < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Classifictaion requires traits and labels!",
        ));
    }

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    for line in lines.filter(|line| !line.trim().is_empty()) {
        let values: Vec<f64> = line
            .split(',')
            .map(|s| {
                s.trim().parse::<f64>().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Niepoprawna liczba: {s}"),
                    )
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

        if !class.is_finite()
            || class < 0.0
            || class.fract() != 0.0
            || class > usize::MAX as f64
        {
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