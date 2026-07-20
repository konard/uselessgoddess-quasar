//! Turning a shard directory into `[batch, seq]` tensors.
//!
//! Every batch is a pure function of its step number, so resuming a run at step
//! 40 000 replays exactly the batches the first attempt would have seen. That
//! costs a `ChaCha8` seeding per step and buys the ability to trust a loss curve
//! that spans a crash.

use burn::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::data::Shards;

/// One training example pair: `target` is `input` shifted left by one token.
#[derive(Debug, Clone)]
pub struct Batch {
    pub input: Tensor<2, Int>,
    pub target: Tensor<2, Int>,
}

/// Draws windows out of a corpus.
#[derive(Debug)]
pub struct Batcher {
    shards: Shards,
    seq_len: usize,
    batch: usize,
    seed: u64,
}

impl Batcher {
    pub fn new(shards: Shards, seq_len: usize, batch: usize, seed: u64) -> Self {
        assert!(shards.len() > seq_len + 1, "corpus shorter than one window");
        Self { shards, seq_len, batch, seed }
    }

    pub fn shards(&self) -> &Shards {
        &self.shards
    }

    /// Tokens per byte of source text — the factor turning a per-token loss into
    /// bits-per-byte, which is the only figure comparable across tokenizers.
    pub fn tokens_per_byte(&self) -> f64 {
        let meta = self.shards.meta();
        meta.tokens as f64 / meta.bytes as f64
    }

    /// The batch for training `step`, drawn uniformly at random.
    ///
    /// Uniform offsets rather than a shuffled epoch: at 2–20 tokens per
    /// parameter the run never revisits the corpus, so an epoch structure would
    /// only add bookkeeping and a resume hazard.
    pub fn train(&self, step: u64, device: &Device) -> Batch {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed ^ step);
        let last = self.shards.len() - self.seq_len - 1;
        let starts: Vec<usize> = (0..self.batch).map(|_| rng.random_range(0..=last)).collect();
        self.gather(&starts, device)
    }

    /// How many non-overlapping evaluation batches the corpus holds.
    pub fn evals(&self) -> usize {
        (self.shards.len() - 1) / self.seq_len / self.batch
    }

    /// The `index`-th evaluation batch: contiguous, non-overlapping windows from
    /// the start of the corpus, so every token is predicted exactly once.
    pub fn eval(&self, index: usize, device: &Device) -> Batch {
        let starts: Vec<usize> =
            (0..self.batch).map(|i| (index * self.batch + i) * self.seq_len).collect();
        self.gather(&starts, device)
    }

    fn gather(&self, starts: &[usize], device: &Device) -> Batch {
        let mut window = Vec::with_capacity(self.seq_len + 1);
        let mut input = Vec::with_capacity(starts.len() * self.seq_len);
        let mut target = Vec::with_capacity(starts.len() * self.seq_len);
        for &start in starts {
            self.shards.read(start, self.seq_len + 1, &mut window);
            input.extend(window[..self.seq_len].iter().map(|&t| t as i32));
            target.extend(window[1..].iter().map(|&t| t as i32));
        }
        let shape = [starts.len(), self.seq_len];
        Batch {
            input: Tensor::from_data(TensorData::new(input, shape), device),
            target: Tensor::from_data(TensorData::new(target, shape), device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::shard;

    fn batcher(tokens: usize, seq_len: usize, batch: usize) -> (tempfile::TempDir, Batcher) {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = shard::Writer::create(dir.path(), 1024, 0).unwrap();
        let doc: Vec<u16> = (0..tokens as u16).collect();
        writer.push(&doc, tokens).unwrap();
        writer.finish().unwrap();
        let shards = Shards::open(dir.path()).unwrap();
        (dir, Batcher::new(shards, seq_len, batch, 7))
    }

    #[test]
    fn the_target_is_the_input_shifted_by_one() {
        let (_dir, batcher) = batcher(256, 8, 2);

        let batch = batcher.eval(0, &Device::default());

        let input = batch.input.slice([0..1, 1..8]);
        assert_eq!(input.into_data(), batch.target.slice([0..1, 0..7]).into_data());
    }

    #[test]
    fn evaluation_windows_do_not_overlap() {
        let (_dir, batcher) = batcher(256, 8, 2);

        let batch = batcher.eval(0, &Device::default());

        let starts = batch.input.slice([0..2, 0..1]).into_data();
        assert_eq!(starts.to_vec::<i32>().unwrap(), [0, 8]);
    }

    #[test]
    fn a_step_always_draws_the_same_batch() {
        let (_dir, batcher) = batcher(256, 8, 2);
        let device = Device::default();

        let (first, again) = (batcher.train(41, &device), batcher.train(41, &device));

        assert_eq!(first.input.into_data(), again.input.into_data());
    }

    #[test]
    fn different_steps_draw_different_batches() {
        let (_dir, batcher) = batcher(4096, 8, 2);
        let device = Device::default();

        let (first, second) = (batcher.train(1, &device), batcher.train(2, &device));

        assert_ne!(first.input.into_data(), second.input.into_data());
    }
}
