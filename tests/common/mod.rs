use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

#[allow(dead_code)]
pub struct TempPlot {
    pub path: PathBuf,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
pub struct TempDir {
    pub path: PathBuf,
}

#[allow(dead_code)]
impl TempDir {
    pub fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!(
                "experiment test {} {} {}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                COUNTER.fetch_add(1, Ordering::Relaxed),
            ));
        std::fs::create_dir(&path).unwrap();
        Self { path }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
