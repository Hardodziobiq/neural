use std::{fmt::Write, io};

pub const COLORS: [&str; 12] = [
    "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d", "#65a30d",
    "#7c3aed", "#0f766e", "#b45309", "#475569",
];

pub fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

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

    fn x(&self, value: f64) -> f64 {
        100.0 + (value - self.bounds[0]) / (self.bounds[1] - self.bounds[0]) * 690.0
    }
    fn y(&self, value: f64) -> f64 {
        470.0 - (value - self.bounds[2]) / (self.bounds[3] - self.bounds[2]) * 400.0
    }

    pub fn legend(&mut self, label: &str, color: &str) {
        self.legend.push((label.into(), color.into()));
    }

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

    pub fn cell(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, color: &str) {
        let (x, y) = (self.x(x0), self.y(y1));
        let (width, height) = (self.x(x1) - x, self.y(y0) - y);
        writeln!(self.body, "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{width:.2}\" height=\"{height:.2}\" fill=\"{color}\" opacity=\"0.18\"/>").unwrap();
    }

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

pub fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
