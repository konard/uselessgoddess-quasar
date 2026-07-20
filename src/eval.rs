//! Validation: the numbers a run is judged by.
//!
//! Bits-per-byte leads, and perplexity follows only for comparison with papers
//! that report it. Token-level perplexity is not comparable across tokenizers —
//! a vocabulary that packs more bytes into a token gets a better number for
//! free — so a run that changed its tokenizer can only be believed in bpb.

use std::fmt;

use burn::prelude::*;

use crate::data::Batcher;
use crate::model::Quasar;

/// What one validation pass measured.
#[derive(Debug, Clone, Copy)]
pub struct Report {
    /// Mean negative log-likelihood per token, in nats.
    pub nll: f64,
    pub perplexity: f64,
    pub bits_per_byte: f64,
    pub tokens: u64,
}

/// Average the loss over `batches` non-overlapping windows.
///
/// `model` must be the inference copy (`model.valid()`): the evaluation pass
/// builds no graph, and on a 16 GB card the activation memory of one is the
/// difference between an eval that fits and one that does not.
pub fn evaluate(model: &Quasar, data: &Batcher, batches: usize, device: &Device) -> Report {
    let batches = batches.min(data.evals()).max(1);
    let mut total = 0.0;
    let mut tokens = 0u64;
    for index in 0..batches {
        let batch = data.eval(index, device);
        let counted = batch.input.dims().iter().product::<usize>() as u64;
        let loss = model.loss(batch.input, batch.target);
        total += loss.nll.into_scalar::<f32>() as f64 * counted as f64;
        tokens += counted;
    }
    Report::new(total / tokens as f64, tokens, data.tokens_per_byte())
}

impl Report {
    fn new(nll: f64, tokens: u64, tokens_per_byte: f64) -> Self {
        Self {
            nll,
            perplexity: nll.exp(),
            bits_per_byte: nll / std::f64::consts::LN_2 * tokens_per_byte,
            tokens,
        }
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { nll, perplexity, bits_per_byte, tokens } = self;
        write!(f, "loss {nll:.4} | ppl {perplexity:.2} | bpb {bits_per_byte:.4} | {tokens} tokens")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config;
    use crate::data::{Shards, shard};

    fn batcher() -> (tempfile::TempDir, Batcher) {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = shard::Writer::create(dir.path(), 64, 0).unwrap();
        let doc: Vec<u16> = (0..256).map(|i| i % 64).collect();
        writer.push(&doc, doc.len() * 4).unwrap();
        writer.finish().unwrap();
        let batcher = Batcher::new(Shards::open(dir.path()).unwrap(), 16, 2, 0);
        (dir, batcher)
    }

    #[test]
    fn an_untrained_model_scores_about_log_vocab() {
        let (cfg, device) = (config::Model::toy(), Device::default());
        let (_dir, data) = batcher();

        let report = evaluate(&Quasar::new(&cfg, &device), &data, 4, &device);

        assert!((report.nll - (cfg.vocab_size as f64).ln()).abs() < 0.5, "{report}");
    }

    #[test]
    fn bits_per_byte_follows_the_compression_ratio() {
        let report = Report::new(std::f64::consts::LN_2, 1, 0.25);

        assert!((report.bits_per_byte - 0.25).abs() < 1e-12, "{report}");
    }
}
