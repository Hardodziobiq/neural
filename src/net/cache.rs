#[derive(Clone)]
pub struct Cache {
    unchanged_sum: f64,
    unchanged_output: f64,
}

impl Cache {
    pub fn sum(&self) -> f64 {
        self.unchanged_sum
    }
    pub fn change_sum(&mut self, sum: f64) {
        self.unchanged_sum = sum;
    }
    pub fn output(&self) -> f64 {
        self.unchanged_output
    }
    pub fn change_output(&mut self, output: f64) {
        self.unchanged_output = output;
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            unchanged_sum: 0.0,
            unchanged_output: 0.0,
        }
    }
}
