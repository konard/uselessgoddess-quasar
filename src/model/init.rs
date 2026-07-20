//! Weight initialisation for the non-SSM parameters.
//!
//! GPT-2's scheme, and for the reason GPT-2 gave: a residual stream that sums
//! `2 · n_layers` sublayer outputs grows like `√(2 · n_layers)` unless the
//! projections that write into it are scaled down by exactly that. burn's
//! defaults are Kaiming (tuned for a plain MLP) and, for embeddings, unit
//! normal — the latter alone puts an untrained model's loss far off `ln V`.
//!
//! Mamba-3's own parameters (`A`, `Δ`, the MIMO channels) keep burn-mamba's
//! initialisation, which is derived from the discretisation and is not ours to
//! second-guess.

use burn::nn::Initializer;

/// Standard deviation for every non-residual weight.
const STD: f64 = 0.02;

pub fn normal() -> Initializer {
    Initializer::Normal { mean: 0.0, std: STD }
}

/// For projections writing into the residual stream (`Ffn::down`, `Attention::o`).
pub fn residual(n_layers: usize) -> Initializer {
    Initializer::Normal { mean: 0.0, std: STD / (2.0 * n_layers as f64).sqrt() }
}
