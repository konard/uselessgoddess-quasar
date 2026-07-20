//! Saving and resuming a run.
//!
//! A checkpoint is a directory holding `model.bpk`, `optim.bpk` and a small
//! `state.json`. Burnpack rather than safetensors because it keeps `ParamId`s,
//! which is what lets the optimizer state be matched back to the parameters it
//! belongs to; safetensors would lose the mapping.

use std::io;
use std::path::{Path, PathBuf};

use burn::optim::ModuleOptimizer;
use burn::store::{BurnpackStore, ModuleSnapshot};
use serde::{Deserialize, Serialize};

use crate::model::Quasar;

/// Where a run stands, beside its weights.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct State {
    /// Optimizer steps already taken; the run resumes at this index.
    pub step: usize,
    pub tokens: u64,
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    /// Burnpack refused the file — a truncated write, or weights whose shapes
    /// no longer match the config.
    Store(String),
}

/// Write `dir/{model,optim}.bpk` and `dir/state.json`.
pub fn save(
    dir: &Path,
    state: State,
    model: &Quasar,
    optim: &ModuleOptimizer,
) -> Result<(), Error> {
    std::fs::create_dir_all(dir).map_err(Error::Io)?;
    // Overwriting is the normal case, not an accident: a run resumed from the
    // last checkpoint writes that same step again when it finishes.
    let mut store = BurnpackStore::from_file(dir.join("model.bpk")).overwrite(true);
    model.save_into(&mut store).map_err(|e| Error::Store(e.to_string()))?;
    optim.save(dir.join("optim.bpk")).map_err(|e| Error::Store(e.to_string()))?;

    let json = serde_json::to_string_pretty(&state).map_err(io::Error::other).map_err(Error::Io)?;
    std::fs::write(dir.join("state.json"), json).map_err(Error::Io)
}

/// Fill `model` from `dir`, ignoring the optimizer — what inference needs.
pub fn weights(dir: &Path, model: &mut Quasar) -> Result<(), Error> {
    let mut store = BurnpackStore::from_file(dir.join("model.bpk"));
    model.load_from(&mut store).map_err(|e| Error::Store(e.to_string()))?;
    Ok(())
}

/// Fill `model` and `optim` from `dir`, returning where the run had got to.
pub fn load(
    dir: &Path,
    model: &mut Quasar,
    optim: ModuleOptimizer,
) -> Result<(ModuleOptimizer, State), Error> {
    weights(dir, model)?;
    let optim = optim.load(dir.join("optim.bpk")).map_err(|e| Error::Store(e.to_string()))?;

    let file = std::fs::File::open(dir.join("state.json")).map_err(Error::Io)?;
    let state = serde_json::from_reader(file).map_err(io::Error::other).map_err(Error::Io)?;
    Ok((optim, state))
}

/// The most advanced checkpoint under `root`, if any.
pub fn latest(root: &Path) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join("state.json").is_file())
        .collect();
    // Zero-padded names sort as their step numbers do, which is the only
    // reason `dir` formats them that way.
    dirs.sort();
    dirs.pop()
}

/// The directory a checkpoint at `step` belongs in.
pub fn dir(root: &Path, step: usize) -> PathBuf {
    root.join(format!("step_{step:07}"))
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Store(message) => write!(f, "burnpack: {message}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::optim::AdamWConfig;
    use burn::prelude::*;

    use crate::config;

    #[test]
    fn a_reloaded_model_answers_the_same() {
        let (cfg, device) = (config::Model::toy(), Device::default());
        let tokens = Tensor::<2, Int>::zeros([1, 8], &device);
        let saved = Quasar::new(&cfg, &device);
        let root = tempfile::tempdir().unwrap();

        save(
            &dir(root.path(), 3),
            State { step: 3, tokens: 0 },
            &saved,
            &AdamWConfig::new().init(),
        )
        .unwrap();

        let mut loaded = Quasar::new(&cfg, &device);
        load(&dir(root.path(), 3), &mut loaded, AdamWConfig::new().init()).unwrap();
        let (a, b) = (saved.forward(tokens.clone()), loaded.forward(tokens));
        assert!((a - b).abs().max().into_scalar::<f32>() < 1e-6);
    }

    #[test]
    fn latest_picks_the_highest_step() {
        let (cfg, device) = (config::Model::toy(), Device::default());
        let model = Quasar::new(&cfg, &device);
        let root = tempfile::tempdir().unwrap();

        for step in [2, 11] {
            let state = State { step, tokens: 0 };
            save(&dir(root.path(), step), state, &model, &AdamWConfig::new().init()).unwrap();
        }

        assert_eq!(latest(root.path()).unwrap(), dir(root.path(), 11));
    }
}
