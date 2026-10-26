use std::{fmt::Write, io};

pub const COLORS: [&str; 12] = [
    "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d", "#65a30d",
    "#7c3aed", "#0f766e", "#b45309", "#475569",
];

/// Escapes text for HTML and SVG so labels and paths are displayed literally.
pub fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Returns `[min_x, max_x, min_y, max_y]` with padding; rejects empty or nonfinite data.
pub fn bounds(points: &[(f64, f64)]) -> io::Result<[f64; 4]> {
    if points.is_empty() || points.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
        return Err(invalid("Chart requires finite, nonempty points"));
    }
    let mut bounds = [points[0].0, points[0].0, points[0].1, points[0].1];
    for &(x, y) in points {
        bounds[0] = bounds[0].min(x);
        bounds[1] = bounds[1].max(x);
        bounds[2] = bounds[2].min(y);
        bounds[3] = bounds[3].max(y);
    }
    for offset in [0, 2] {
        let span = bounds[offset + 1] - bounds[offset];
        let padding = if span == 0.0 {
            bounds[offset].abs().max(1.0) * 0.05
        } else {
            span * 0.05
        };
        bounds[offset] -= padding;
        bounds[offset + 1] += padding;
        if !bounds[offset].is_finite()
            || !bounds[offset + 1].is_finite()
            || !(bounds[offset + 1] - bounds[offset]).is_finite()
            || bounds[offset] >= bounds[offset + 1]
        {
            return Err(invalid("Chart coordinate range is too large or too small"));
        }
    }
    Ok(bounds)
}

pub struct Chart {
    title: String,
    x_label: String,
    y_label: String,
    bounds: [f64; 4],
    body: String,
    legend: Vec<(String, String)>,
}

impl Chart {
    /// Starts an empty chart using titles and a prevalidated coordinate range.
    pub fn new(title: &str, x_label: &str, y_label: &str, bounds: [f64; 4]) -> Self {
        Self {
            title: title.into(),
            x_label: x_label.into(),
            y_label: y_label.into(),
            bounds,
            body: String::new(),
            legend: Vec::new(),
        }
    }

    /// Maps a data value to its horizontal position inside the SVG plotting area.
    fn x(&self, value: f64) -> f64 {
        100.0 + (value - self.bounds[0]) / (self.bounds[1] - self.bounds[0]) * 690.0
    }
    /// Maps a data value to its SVG vertical position, reversing the downward screen axis.
    fn y(&self, value: f64) -> f64 {
        470.0 - (value - self.bounds[2]) / (self.bounds[3] - self.bounds[2]) * 400.0
    }

    /// Adds a series label and its color to the legend drawn by `finish`.
    pub fn legend(&mut self, label: &str, color: &str) {
        self.legend.push((label.into(), color.into()));
    }

    /// Draws a translucent interval around a series, using lower and upper values at each x coordinate.
    pub fn band(&mut self, intervals: &[(f64, f64, f64)], color: &str) {
        let points = intervals
            .iter()
            .map(|&(x, _, upper)| (x, upper))
            .chain(intervals.iter().rev().map(|&(x, lower, _)| (x, lower)))
            .map(|(x, y)| format!("{:.2},{:.2}", self.x(x), self.y(y)))
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(
            self.body,
            "<polygon points=\"{points}\" fill=\"{color}\" opacity=\"0.15\"/>"
        )
        .unwrap();
    }

    /// Appends a connected line through data points; a single observation gets a visible marker.
    pub fn line(&mut self, points: &[(f64, f64)], color: &str, id: &str) {
        let coordinates = points
            .iter()
            .map(|&(x, y)| format!("{:.2},{:.2}", self.x(x), self.y(y)))
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(self.body, "<polyline id=\"{}\" points=\"{coordinates}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"/>", escape(id)).unwrap();
        if points.len() == 1 {
            self.point(points[0], color, false, "Single observation", false);
        }
    }

    /// Draws a series in separate segments, breaking the line wherever a measurement is missing.
    pub fn gapped_line(&mut self, points: &[(f64, Option<f64>)], color: &str, id: &str) {
        writeln!(self.body, "<g id=\"{}\">", escape(id)).unwrap();
        for (index, segment) in points.split(|(_, value)| value.is_none()).enumerate() {
            if segment.is_empty() {
                continue;
            }
            let values: Vec<_> = segment.iter().map(|&(x, y)| (x, y.unwrap())).collect();
            self.line(&values, color, &format!("{id}-{index}"));
        }
        self.body.push_str("</g>\n");
    }

    /// Draws a circle or square with a tooltip, using a dark border to mark a prediction error.
    pub fn point(
        &mut self,
        point: (f64, f64),
        color: &str,
        square: bool,
        label: &str,
        error: bool,
    ) {
        let (x, y) = (self.x(point.0), self.y(point.1));
        let stroke = if error { "#111827" } else { "white" };
        let stroke_width = if error { 3 } else { 1 };
        let title = escape(label);
        if square {
            writeln!(self.body, "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"8\" height=\"8\" fill=\"{color}\" stroke=\"{stroke}\" stroke-width=\"{stroke_width}\"><title>{title}</title></rect>", x-4.0, y-4.0).unwrap();
        } else {
            writeln!(self.body, "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"4\" fill=\"{color}\" stroke=\"{stroke}\" stroke-width=\"{stroke_width}\"><title>{title}</title></circle>").unwrap();
        }
    }

    /// Draws a translucent data-space rectangle used to show classification decision regions.
    pub fn cell(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, color: &str) {
        let (x, y) = (self.x(x0), self.y(y1));
        let (width, height) = (self.x(x1) - x, self.y(y0) - y);
        writeln!(self.body, "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{width:.2}\" height=\"{height:.2}\" fill=\"{color}\" opacity=\"0.18\"/>").unwrap();
    }

    /// Consumes the chart and returns complete SVG text, including axes and legend; does not write a file.
    pub fn finish(self) -> String {
        let height = 560.max(100 + self.legend.len() * 20);
        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1000\" height=\"{height}\" viewBox=\"0 0 1000 {height}\">\n<title>{}</title><rect width=\"1000\" height=\"{height}\" fill=\"white\"/><g font-family=\"sans-serif\" font-size=\"12\" fill=\"#253047\"><text x=\"100\" y=\"32\" font-size=\"20\">{}</text>\n",
            escape(&self.title),
            escape(&self.title)
        );
        svg.push_str(&self.body);
        for tick in 0..=5 {
            let fraction = tick as f64 / 5.0;
            let x_value = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * fraction;
            let y_value = self.bounds[2] + (self.bounds[3] - self.bounds[2]) * fraction;
            let (x, y) = (self.x(x_value), self.y(y_value));
            writeln!(svg, "<text x=\"{x:.2}\" y=\"492\" text-anchor=\"middle\">{x_value:.2e}</text><text x=\"90\" y=\"{:.2}\" text-anchor=\"end\">{y_value:.2e}</text>", y+4.0).unwrap();
        }
        svg.push_str("<path d=\"M 100 70 V 470 H 790\" fill=\"none\" stroke=\"#64748b\"/>\n");
        writeln!(svg, "<text x=\"445\" y=\"530\" text-anchor=\"middle\">{}</text><text transform=\"translate(22 270) rotate(-90)\" text-anchor=\"middle\">{}</text>", escape(&self.x_label), escape(&self.y_label)).unwrap();
        for (index, (label, color)) in self.legend.iter().enumerate() {
            writeln!(
                svg,
                "<text x=\"815\" y=\"{}\" fill=\"{color}\">{}</text>",
                80 + index * 20,
                escape(label)
            )
            .unwrap();
        }
        svg.push_str("</g></svg>\n");
        svg
    }
}

/// Creates an invalid-input I/O error for data that cannot be plotted.
pub fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
