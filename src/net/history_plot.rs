use std::{fmt::Write, io};

use crate::net::history::History;

/// Renders only epoch metrics; parameter snapshots are not required.
pub(super) fn render(history: &History) -> io::Result<String> {
    let epochs = &history.epochs;
    if epochs.is_empty() {
        return Err(invalid("Cannot plot an empty training history"));
    }
    if epochs.windows(2).any(|pair| pair[0].epoch >= pair[1].epoch) {
        return Err(invalid("Plot epochs must be strictly increasing"));
    }
    let mut minimum = 0.0_f64;
    let mut maximum = 0.0_f64;
    for epoch in epochs {
        for loss in std::iter::once(epoch.train_loss).chain(epoch.test_loss) {
            if !loss.is_finite() {
                return Err(invalid("Plot losses must be finite"));
            }
            minimum = minimum.min(loss);
            maximum = maximum.max(loss);
        }
    }
    if minimum == maximum {
        maximum = minimum + 1.0;
    }
    let range = maximum - minimum;
    if !range.is_finite() {
        return Err(invalid("Loss range is too large to plot"));
    }

    let (left, top, width, height) = (100.0, 60.0, 760.0, 360.0);
    let first = epochs[0].epoch;
    let last = epochs[epochs.len() - 1].epoch;
    let x = |epoch: usize| {
        if first == last {
            left + width / 2.0
        } else {
            left + (epoch - first) as f64 / (last - first) as f64 * width
        }
    };
    let y = |loss: f64| top + height - (loss - minimum) / range * height;
    let mut svg = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"500\" viewBox=\"0 0 900 500\">\n\
         <title>Training and test loss by epoch</title>\n\
         <rect width=\"900\" height=\"500\" fill=\"white\"/>\n\
         <g font-family=\"sans-serif\" font-size=\"13\" fill=\"#253047\">\n\
         <text x=\"100\" y=\"30\" font-size=\"20\">Loss by epoch</text>\n\
         <text x=\"480\" y=\"480\" text-anchor=\"middle\">Epoch</text>\n\
         <text transform=\"translate(22 240) rotate(-90)\" text-anchor=\"middle\">Loss</text>\n\
         <text x=\"620\" y=\"30\" fill=\"#2563eb\">Training</text>\n",
    );
    if epochs.iter().any(|epoch| epoch.test_loss.is_some()) {
        svg.push_str("<text x=\"750\" y=\"30\" fill=\"#d97706\">Test</text>\n");
    }
    for tick in 0..=5 {
        let loss = minimum + range * (tick as f64 / 5.0);
        let position = y(loss);
        writeln!(
            svg,
            "<path d=\"M {left} {position:.2} h {width}\" stroke=\"#e2e8f0\"/>"
        )
        .unwrap();
        writeln!(
            svg,
            "<text x=\"90\" y=\"{:.2}\" text-anchor=\"end\">{loss:.2e}</text>",
            position + 4.0
        )
        .unwrap();
    }
    let ticks = epochs.len().min(6);
    for tick in 0..ticks {
        let index = if ticks == 1 {
            0
        } else {
            tick * (epochs.len() - 1) / (ticks - 1)
        };
        let epoch = epochs[index].epoch;
        writeln!(
            svg,
            "<text x=\"{:.2}\" y=\"445\" text-anchor=\"middle\">{epoch}</text>",
            x(epoch)
        )
        .unwrap();
    }
    writeln!(
        svg,
        "<path d=\"M {left} {top} v {height} h {width}\" fill=\"none\" stroke=\"#64748b\"/>"
    )
    .unwrap();
    let points = epochs
        .iter()
        .map(|epoch| format!("{:.2},{:.2}", x(epoch.epoch), y(epoch.train_loss)))
        .collect::<Vec<_>>()
        .join(" ");
    writeln!(svg, "<polyline id=\"training-loss\" points=\"{points}\" fill=\"none\" stroke=\"#2563eb\" stroke-width=\"2\"/>").unwrap();
    if epochs.len() == 1 {
        writeln!(
            svg,
            "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"3\" fill=\"#2563eb\"/>",
            x(first),
            y(epochs[0].train_loss)
        )
        .unwrap();
    }
    svg.push_str("<g id=\"test-loss\" stroke=\"#d97706\" fill=\"#d97706\">\n");
    for (index, epoch) in epochs.iter().enumerate() {
        if let Some(loss) = epoch.test_loss {
            // Missing test metrics break the line rather than implying a measured value.
            if index > 0 {
                if let Some(previous) = epochs[index - 1].test_loss {
                    writeln!(
                        svg,
                        "<path d=\"M {:.2} {:.2} L {:.2} {:.2}\" stroke-width=\"2\"/>",
                        x(epochs[index - 1].epoch),
                        y(previous),
                        x(epoch.epoch),
                        y(loss)
                    )
                    .unwrap();
                }
            }
            writeln!(
                svg,
                "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"2\"/>",
                x(epoch.epoch),
                y(loss)
            )
            .unwrap();
        }
    }
    svg.push_str("</g>\n</g>\n</svg>\n");
    Ok(svg)
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
