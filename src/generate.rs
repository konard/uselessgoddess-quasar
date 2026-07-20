//! Sampling from a trained model.
//!
//! There is no recurrent cache: each new token re-runs the whole context. That
//! is `O(n²)` work for an `O(n)` architecture and it is deliberate — a cache
//! duplicates the SSD recurrence in a second code path, and the first thing a
//! subtly wrong cache does is make the model look worse than it is. This is for
//! eyeballing samples during a run, not for serving.

use burn::prelude::*;
use burn::tensor::activation::softmax;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::data::Tokenizer;
use crate::data::tokenizer;
use crate::model::Quasar;

/// How to sample.
#[derive(Debug, Clone, Copy)]
pub struct Sampler {
    /// `0.0` is greedy.
    pub temperature: f64,
    /// Candidates kept per step; `0` keeps the whole vocabulary.
    pub top_k: usize,
    pub max_tokens: usize,
    pub seed: u64,
}

impl Default for Sampler {
    fn default() -> Self {
        Self { temperature: 0.8, top_k: 40, max_tokens: 128, seed: 1337 }
    }
}

/// Continue `prompt`, stopping at `max_tokens` or the end-of-text token.
///
/// The context is cropped to `seq_len` from the right, so a long generation
/// slides rather than failing.
pub fn generate(
    model: &Quasar,
    tokenizer: &Tokenizer,
    prompt: &str,
    seq_len: usize,
    sampler: &Sampler,
    device: &Device,
) -> Result<String, tokenizer::Error> {
    let mut ids = tokenizer.encode_raw(prompt)?;
    let mut rng = ChaCha8Rng::seed_from_u64(sampler.seed);
    let start = ids.len();

    for _ in 0..sampler.max_tokens {
        let context = &ids[ids.len().saturating_sub(seq_len)..];
        let next = step(model, context, sampler, &mut rng, device);
        if next == tokenizer.eos() {
            break;
        }
        ids.push(next);
    }
    tokenizer.decode(&ids[start..])
}

/// One token, sampled from the distribution at the last position.
fn step(
    model: &Quasar,
    context: &[u16],
    sampler: &Sampler,
    rng: &mut ChaCha8Rng,
    device: &Device,
) -> u16 {
    let ids: Vec<i32> = context.iter().map(|&id| id as i32).collect();
    let tokens = Tensor::<1, Int>::from_ints(&ids[..], device).reshape([1, ids.len()]);
    let logits = model.forward(tokens);

    let [_, seq, vocab] = logits.dims();
    let last = logits.slice([0..1, seq - 1..seq, 0..vocab]).reshape([vocab]);
    let scaled = match sampler.temperature {
        t if t <= 0.0 => return last.argmax(0).into_scalar::<i32>() as u16,
        t => last.div_scalar(t),
    };
    let probs = softmax(scaled, 0).into_data().to_vec::<f32>().expect("probabilities are f32");
    pick(&probs, sampler.top_k, rng)
}

/// Sample an index from `probs`, restricted to its `top_k` largest entries.
fn pick(probs: &[f32], top_k: usize, rng: &mut ChaCha8Rng) -> u16 {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    let keep = if top_k == 0 { probs.len() } else { top_k.min(probs.len()) };
    order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    order.truncate(keep);

    let mass: f32 = order.iter().map(|&i| probs[i]).sum();
    let mut target = rng.random_range(0.0..mass.max(f32::MIN_POSITIVE));
    for &index in &order {
        target -= probs[index];
        if target <= 0.0 {
            return index as u16;
        }
    }
    // Only reachable through float rounding, where the last kept token is as
    // good an answer as any.
    *order.last().expect("the vocabulary is not empty") as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config;

    /// A toy model wide enough at the vocabulary to accept this tokenizer.
    fn pair() -> (config::Model, Tokenizer) {
        let docs = ["the quick brown fox", "the lazy dog", "quick quick brown"];
        let tokenizer = Tokenizer::train(docs.into_iter(), 300).unwrap();
        let toy = config::Model::toy();
        (config::Model { vocab_size: tokenizer.vocab_size(), ..toy }, tokenizer)
    }

    #[test]
    fn a_greedy_sample_is_reproducible() {
        let (cfg, tokenizer) = pair();
        let device = Device::default();
        let model = Quasar::new(&cfg, &device);
        let sampler = Sampler { temperature: 0.0, max_tokens: 4, ..Sampler::default() };

        let once = generate(&model, &tokenizer, "the", cfg.seq_len, &sampler, &device).unwrap();
        let twice = generate(&model, &tokenizer, "the", cfg.seq_len, &sampler, &device).unwrap();

        assert_eq!(once, twice);
    }

    #[test]
    fn top_k_never_leaves_the_top_k() {
        let probs = [0.1, 0.7, 0.05, 0.15];
        let mut rng = ChaCha8Rng::seed_from_u64(7);

        let picks: Vec<u16> = (0..64).map(|_| pick(&probs, 2, &mut rng)).collect();

        assert!(picks.iter().all(|&i| i == 1 || i == 3), "{picks:?}");
    }

    #[test]
    fn a_certain_distribution_always_picks_its_token() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);

        let pick = pick(&[0.0, 0.0, 1.0], 0, &mut rng);

        assert_eq!(pick, 2);
    }
}
