use crate::net::error::ValidationError;

/// Column-wise standardization fitted exclusively on training data.
/// Constant training columns use a scale of one, so the transform remains reversible.
#[derive(Clone, Debug)]
pub struct StandardScaler {
    means: Vec<f64>,
    standard_deviations: Vec<f64>,
}

impl StandardScaler {
    pub fn fit(rows: &[Vec<f64>]) -> Result<Self, ValidationError> {
        let columns = rows.first().map_or(0, Vec::len);
        if rows.is_empty() || columns == 0 {
            return Err(ValidationError::new(
                "Scaler requires nonempty rows and columns",
            ));
        }
        let mut means = vec![0.0; columns];
        let mut squared_deviations = vec![0.0; columns];
        for (sample, row) in rows.iter().enumerate() {
            if row.len() != columns {
                return Err(ValidationError::new(format!(
                    "Scaler row {sample}: expected {columns} values, got {}",
                    row.len(),
                )));
            }
            for (column, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(ValidationError::new(format!(
                        "Scaler row {sample}, column {column}: value must be finite",
                    )));
                }
                // Welford's algorithm avoids subtracting two large, similar sums.
                let delta = value - means[column];
                means[column] += delta / (sample + 1) as f64;
                squared_deviations[column] += delta * (value - means[column]);
                if !means[column].is_finite() || !squared_deviations[column].is_finite() {
                    return Err(ValidationError::new(format!(
                        "Scaler column {column}: numeric overflow while fitting",
                    )));
                }
            }
        }
        let standard_deviations = squared_deviations
            .iter()
            .map(|sum| (sum.max(0.0) / rows.len() as f64).sqrt())
            .collect();
        Ok(Self {
            means,
            standard_deviations,
        })
    }

    pub fn means(&self) -> &[f64] {
        &self.means
    }

    pub fn standard_deviations(&self) -> &[f64] {
        &self.standard_deviations
    }

    /// Applies the saved training statistics; no statistics are recomputed.
    /// On error, every row remains unchanged.
    pub fn transform(&self, rows: &mut [Vec<f64>]) -> Result<(), ValidationError> {
        self.transform_rows(rows, false)
    }

    pub fn inverse_transform(&self, rows: &mut [Vec<f64>]) -> Result<(), ValidationError> {
        self.transform_rows(rows, true)
    }

    pub fn transform_sample(&self, input: &[f64]) -> Result<Vec<f64>, ValidationError> {
        self.map_sample(input, false)
    }

    pub fn inverse_transform_sample(&self, input: &[f64]) -> Result<Vec<f64>, ValidationError> {
        self.map_sample(input, true)
    }

    fn transform_rows(&self, rows: &mut [Vec<f64>], inverse: bool) -> Result<(), ValidationError> {
        let transformed = rows
            .iter()
            .enumerate()
            .map(|(sample, row)| {
                self.map_sample(row, inverse)
                    .map_err(|error| ValidationError::new(format!("Scaler row {sample}: {error}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (row, transformed) in rows.iter_mut().zip(transformed) {
            *row = transformed;
        }
        Ok(())
    }

    fn map_sample(&self, input: &[f64], inverse: bool) -> Result<Vec<f64>, ValidationError> {
        if input.len() != self.means.len() {
            return Err(ValidationError::new(format!(
                "Expected {} values, got {}",
                self.means.len(),
                input.len(),
            )));
        }
        input
            .iter()
            .enumerate()
            .map(|(column, &value)| {
                if !value.is_finite() {
                    return Err(ValidationError::new(format!(
                        "Column {column} must be finite"
                    )));
                }
                let std = self.standard_deviations[column];
                let scale = if std > 0.0 { std } else { 1.0 };
                let result = if inverse {
                    value * scale + self.means[column]
                } else {
                    (value - self.means[column]) / scale
                };
                if !result.is_finite() {
                    return Err(ValidationError::new(format!(
                        "Numeric overflow while transforming column {column}",
                    )));
                }
                Ok(result)
            })
            .collect()
    }
}
