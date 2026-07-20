//! The training loop.
//!
//! Everything the loop needs is a pure function of the step index — the batch,
//! the learning rate, the schedule — so a crashed run resumes into exactly the
//! stream it left. The loop itself is deliberately a plain `for`: burn's
//! `Learner` owns its own metrics, renderer and checkpointing, none of which
//! survive contact with a run that has to report bits-per-byte and be stopped
//! whenever the card is wanted for something else.

use std::io;
use std::path::Path;
use std::time::Instant;

use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams};
use burn::prelude::*;

use crate::data::{Batcher, Shards};
use crate::model::Quasar;
use crate::train::checkpoint::{self, State};
use crate::train::schedule::Wsd;
use crate::{config, eval};

/// Everything about a run that is not the model itself.
///
/// The defaults are the `tiny` recipe: 24 hours on one RX 9070 XT.
#[derive(Config, Debug)]
pub struct Run {
    #[config(default = 60_000)]
    pub steps: usize,
    /// Sequences per forward pass. This is the VRAM knob — raise it until the
    /// card refuses, then compensate with `accum`.
    #[config(default = 8)]
    pub micro_batch: usize,
    /// Forward/backward passes per optimizer step. `micro_batch * accum *
    /// seq_len` is the token batch, which is what the learning rate is tuned
    /// against, so the two must move together.
    #[config(default = 16)]
    pub accum: usize,
    #[config(default = 3e-3)]
    pub lr: f64,
    /// Final rate as a fraction of `lr`.
    #[config(default = 0.1)]
    pub lr_floor: f64,
    #[config(default = 2_000)]
    pub warmup: usize,
    #[config(default = 12_000)]
    pub decay: usize,
    #[config(default = 0.1)]
    pub weight_decay: f32,
    #[config(default = 1.0)]
    pub clip: f32,
    #[config(default = 1337)]
    pub seed: u64,
    #[config(default = 20)]
    pub log_every: usize,
    #[config(default = 1_000)]
    pub eval_every: usize,
    #[config(default = 20)]
    pub eval_batches: usize,
    #[config(default = 2_000)]
    pub save_every: usize,
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Checkpoint(checkpoint::Error),
}

/// Train `cfg` on the shards under `data`, checkpointing into `out`.
///
/// Resumes by itself from the newest checkpoint in `out`, because that is what
/// a run interrupted at 3am needs to do when it is restarted at 9.
pub fn run(
    cfg: &config::Model,
    run: &Run,
    data: &Path,
    out: &Path,
    device: &Device,
) -> Result<(), Error> {
    let inference = device.clone();
    let device = device.clone().autodiff();
    device.seed(run.seed);

    let train = batcher(&data.join("train"), cfg.seq_len, run)?;
    let valid = batcher(&data.join("valid"), cfg.seq_len, run)?;

    let mut model = Quasar::new(cfg, &device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(run.weight_decay)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(run.clip)))
        .init();

    std::fs::create_dir_all(out).map_err(Error::Io)?;
    cfg.save(out.join("model.json")).map_err(Error::Io)?;
    run.save(out.join("run.json")).map_err(Error::Io)?;

    let mut state = State { step: 0, tokens: 0 };
    if let Some(dir) = checkpoint::latest(out) {
        let (loaded, resumed) = checkpoint::load(&dir, &mut model, optim)?;
        (optim, state) = (loaded, resumed);
        println!("resuming {} at step {}", dir.display(), state.step);
    }

    let schedule = Wsd::new(run.lr, run.lr_floor, run.warmup, run.decay, run.steps);
    let per_step = (run.micro_batch * run.accum * cfg.seq_len) as u64;
    let mut grads = GradientsAccumulator::new();
    let mut window = Window::new();

    for step in state.step..run.steps {
        let lr = schedule.lr(step);
        // Reading a loss scalar blocks until the device catches up, so it is
        // read on logging steps only; the rest never leave the queue.
        let logging = due(step, run.log_every);

        for micro in 0..run.accum {
            let batch = train.train((step * run.accum + micro) as u64, &device);
            let loss = model.loss(batch.input, batch.target);
            if logging {
                window.loss += loss.nll.clone().into_scalar::<f32>() as f64 / run.accum as f64;
            }
            // Scaling here rather than after accumulating keeps the gradient of
            // an accumulated step identical to that of one big batch.
            let step = loss.total.div_scalar(run.accum as f64).backward();
            grads.accumulate(&model, GradientsParams::from_grads(step, &model));
        }
        model = optim.step(lr, model, grads.grads());
        state = State { step: step + 1, tokens: state.tokens + per_step };

        if logging {
            println!("{}", window.report(state, run, lr, per_step));
            window = Window::new();
        }
        if due(step, run.eval_every) {
            let report = eval::evaluate(&model.valid(), &valid, run.eval_batches, &inference);
            println!("  valid: {report}");
        }
        if due(step, run.save_every) {
            checkpoint::save(&checkpoint::dir(out, state.step), state, &model, &optim)?;
        }
    }

    checkpoint::save(&checkpoint::dir(out, state.step), state, &model, &optim)?;
    println!("final: {}", eval::evaluate(&model.valid(), &valid, run.eval_batches, &inference));
    Ok(())
}

fn batcher(dir: &Path, seq_len: usize, run: &Run) -> Result<Batcher, Error> {
    let shards = Shards::open(dir).map_err(Error::Io)?;
    Ok(Batcher::new(shards, seq_len, run.micro_batch, run.seed))
}

/// Whether `step` is the last of an `every`-sized period; `every == 0` disables.
fn due(step: usize, every: usize) -> bool {
    every > 0 && (step + 1).is_multiple_of(every)
}

/// The accumulator behind one log line.
struct Window {
    loss: f64,
    started: Instant,
}

impl Window {
    fn new() -> Self {
        Self { loss: 0.0, started: Instant::now() }
    }

    fn report(&self, state: State, run: &Run, lr: f64, per_step: u64) -> String {
        let steps = run.log_every.max(1);
        let seconds = self.started.elapsed().as_secs_f64();
        let throughput = (steps as u64 * per_step) as f64 / seconds;
        let loss = self.loss / steps as f64;
        format!(
            "step {}/{} | loss {loss:.4} | lr {lr:.2e} | {:.0} tok/s | {:.2}B tokens",
            state.step,
            run.steps,
            throughput,
            state.tokens as f64 / 1e9,
        )
    }
}

impl From<checkpoint::Error> for Error {
    fn from(error: checkpoint::Error) -> Self {
        Self::Checkpoint(error)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Checkpoint(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::shard;

    fn shards(dir: &Path) {
        let mut writer = shard::Writer::create(dir, 64, 0).unwrap();
        let doc: Vec<u16> = (0..1024).map(|i| i % 64).collect();
        writer.push(&doc, doc.len() * 4).unwrap();
        writer.finish().unwrap();
    }

    fn tiny_run() -> Run {
        Run::new()
            .with_steps(2)
            .with_micro_batch(2)
            .with_accum(2)
            .with_warmup(1)
            .with_decay(1)
            .with_log_every(0)
            .with_eval_every(0)
            .with_save_every(0)
            .with_eval_batches(1)
    }

    #[test]
    fn a_short_run_leaves_a_resumable_checkpoint() {
        let data = tempfile::tempdir().unwrap();
        shards(&data.path().join("train"));
        shards(&data.path().join("valid"));
        let out = tempfile::tempdir().unwrap();

        run(&config::Model::toy(), &tiny_run(), data.path(), out.path(), &Device::default())
            .unwrap();

        assert_eq!(checkpoint::latest(out.path()).unwrap(), checkpoint::dir(out.path(), 2));
    }

    #[test]
    fn a_disabled_period_never_comes_due() {
        assert!(!due(41, 0));
    }
}
