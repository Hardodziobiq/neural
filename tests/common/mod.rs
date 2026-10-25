use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

pub struct TempPlot {
    pub path: PathBuf,
}

impl TempPlot {
    pub fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self {
            path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join(format!(
                    "loss plot test {} {}.svg",
                    std::process::id(),
                    COUNTER.fetch_add(1, Ordering::Relaxed),
                )),
        }
    }
}

impl Drop for TempPlot {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
