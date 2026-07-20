//! The SwiGLU feed-forward sublayer.

use burn::nn::{Linear, LinearConfig, SwiGlu, SwiGluConfig};
use burn::prelude::*;

use crate::config;
use crate::model::init;

/// `down(silu(gate(x)) * up(x))` — three bias-free matrices, no dropout.
///
/// Dropout is absent on purpose: at 2–20 tokens per parameter the run is
/// nowhere near memorising the corpus, so the regulariser would only slow
/// convergence.
#[derive(Module, Debug)]
pub struct Ffn {
    gate: SwiGlu,
    down: Linear,
}

impl Ffn {
    pub fn new(cfg: &config::Model, device: &Device) -> Self {
        let d_ff = cfg.d_ff();
        Self {
            gate: SwiGluConfig::new(cfg.d_model, d_ff).with_initializer(init::normal()).init(device),
            down: LinearConfig::new(d_ff, cfg.d_model)
                .with_bias(false)
                .with_initializer(init::residual(cfg.n_layers))
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.down.forward(self.gate.forward(x))
    }
}
